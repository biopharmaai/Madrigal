""" 
Adapted from https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py
"""
import torch
import torch.nn as nn
from copy import deepcopy


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, init_batch, init_masks, init_mols, dim=256, mlp_dim=1024, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 1024)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        # self.all_molecules = all_molecules  # TODO: Move this outside of the class

        # build encoders
        self.base_encoder = base_encoder
        self.momentum_encoder = deepcopy(base_encoder)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)
        self._lazy_initialize(init_batch, init_masks, init_mols)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _lazy_initialize(self, init_batch, init_masks, init_mols):
        with torch.no_grad():
            self.base_encoder.train()
            self.momentum_encoder.train()
            self.base_encoder = self.base_encoder.to(init_masks.device)
            self.momentum_encoder = self.momentum_encoder.to(init_masks.device)
            init_batch = init_batch.to(init_masks.device)
            _ = self.base_encoder(init_batch, init_masks, init_mols)
            _ = self.momentum_encoder(init_batch, init_masks, init_mols)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T  # einsum is same as torch.mm(q, k.T)
        N = logits.shape[0]  # batch size
        labels = torch.arange(N, dtype=torch.long).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)


    def forward(self, drug_indices, batch_mask1, batch_mask2, batch_too_hard_neg_mask, batch_data, batch_extra_mols, batch_extra_masks):
        """
        Input:
            drug_indices
            mask1: first tensor of subsets of views of images
            mask2: second tensor of subsets of views of images
            m: moco momentum
        Output:
            loss
        """
        # compute features
        aug1 = self.predictor(self.base_encoder(drug_indices, batch_mask1, batch_mols, batch_kg, batch_cv, batch_tx_mcf7, batch_tx_pc3, batch_tx_vcap))
        aug2 = self.predictor(self.base_encoder(drug_indices, batch_mask2, batch_mols, batch_kg, batch_cv, batch_tx_mcf7, batch_tx_pc3, batch_tx_vcap))

        with torch.no_grad():  # no gradient
            if self.training:
                self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(drug_indices, mask1, batch_mols)
            k2 = self.momentum_encoder(drug_indices, mask2, batch_mols)
         
        return aug1, aug2, self.contrastive_loss(aug1, aug2, batch_too_hard_neg_mask)
        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo_NovelDDI(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.transformer.transformer_encoder.layers[-1].norm2.weight.shape[0]

        # projectors
        self.base_encoder.head = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

