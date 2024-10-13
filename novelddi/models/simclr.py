"""
Following the implementation in https://github.com/sthalles/SimCLR/blob/master/, Q and K are concatenated together rather than being separate as in MoCo.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR_NovelDDI(nn.Module):
    def __init__(self, base_encoder, dim=256, mlp_dim=1024, T=1.0, raw_encoder_output=False, shared_predictor=False):
        super(SimCLR_NovelDDI, self).__init__()
        self.base_encoder = base_encoder
        self.T = T
        self.raw_encoder_output = raw_encoder_output  # whether to use modality encoder outputs or the alreayd projected embeddings
        self.shared_predictor = shared_predictor  # whether to use the same predictor for both streams
        
        # add mlp projection head
        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        # lazy initialize
        # self._lazy_initialize(init_batch_drugs, init_masks, init_mols, init_cv, init_tx_mcf7, init_tx_pc3, init_tx_vcap)
    
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.uni_projector.fc[-1].weight.shape[0]  # get last dim of xW^T, same as self.base_encoder.transformer.latent2embed.weight.shape[0]
        assert hidden_dim == dim, f"Hidden dim of the encoder ({hidden_dim}) should be the same as the dim of the projection head ({dim})."
        
        # NOTE: projectors OMITTED
        # self.base_encoder.head = self._build_mlp(2, hidden_dim, mlp_dim, dim)  # In MoCo, you have separate encoders for Q and K, but here we have a single encoder, so there is intuitively no need to have separate projectors for Q and K.

        # predictor
        if self.shared_predictor:
            self.predictor = self._build_mlp(2, dim, mlp_dim, dim)
        else:
            self.predictor_1 = self._build_mlp(2, dim, mlp_dim, dim)
            self.predictor_2 = self._build_mlp(2, dim, mlp_dim, dim)

    # @torch.no_grad()
    # def _lazy_initialize(self, init_batch_drugs, init_masks, init_mols, init_batch_kg, init_batch_cv, init_batch_tx_mcf7, init_batch_tx_pc3, init_batch_tx_vcap):
    #     self.base_encoder.train()
    #     self.base_encoder = self.base_encoder.to(init_masks.device)
    #     init_batch_drugs = init_batch_drugs.to(init_masks.device)
    #     _ = self.base_encoder(init_batch_drugs, init_masks, init_mols, init_batch_kg, init_batch_cv, init_batch_tx_mcf7, init_batch_tx_pc3, init_batch_tx_vcap)

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

    # def contrastive_loss(self, q, k):
    #     # normalize
    #     q = nn.functional.normalize(q, dim=1)
    #     k = nn.functional.normalize(k, dim=1)
    #     # Einstein sum is more intuitive
    #     logits = torch.einsum('nc,mc->nm', [q, k]) / self.T  # same as torch.mm(q, k.T)
    #     N = logits.shape[0]  # batch size
    #     labels = torch.arange(N, dtype=torch.long).cuda()
    #     return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def contrastive_loss(self, aug1, aug2, batch_too_hard_neg_mask):
        assert aug1.shape[0] == aug2.shape[0]
        features = torch.cat([aug1, aug2], dim=0)
        labels = torch.cat([torch.arange(aug1.shape[0])] * 2, dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        
        # mask out the scores from the pairs that should not be negatives (too similar)
        if batch_too_hard_neg_mask is not None:
            similarity_matrix.masked_fill_(batch_too_hard_neg_mask.repeat(2, 2), -1e9)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1).to(features.device)
        assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        # positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        # negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # logits = torch.cat([positives, negatives], dim=1)  # reorganize so that positives are always the first column now
        # logits = logits.to(device)
        # logits = logits / self.T
        
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        
        logits = similarity_matrix / self.T

        return logits, labels, torch.nn.CrossEntropyLoss()(logits, labels)

    def forward(self, drug_indices, batch_mask_1, batch_mask_2, batch_too_hard_neg_mask, batch_data, batch_extra_mols, batch_extra_masks):
        """
        Input:
            drug_indices
            mask1: first tensor of subsets of views of images
            mask2: second tensor of subsets of views of images
            m: moco momentum
        Output:
            loss
        """
        batch_mols, batch_kg, batch_cv, batch_tx_dict = batch_data
        
        # compute features
        if self.shared_predictor:
            aug_1 = self.predictor(self.base_encoder(drug_indices, batch_mask_1, batch_mols, batch_kg, batch_cv, batch_tx_dict, raw_encoder_output=self.raw_encoder_output))  # raw encoder output is [batch_size, seq_len, hidden_dim], then select the corresponding modality output for each compound
            aug_2 = self.predictor(self.base_encoder(drug_indices, batch_mask_2, batch_mols, batch_kg, batch_cv, batch_tx_dict, raw_encoder_output=self.raw_encoder_output))
        else:
            aug_1 = self.predictor_1(self.base_encoder(drug_indices, batch_mask_1, batch_mols, batch_kg, batch_cv, batch_tx_dict, raw_encoder_output=self.raw_encoder_output))  # raw encoder output is [batch_size, seq_len, hidden_dim], then select the corresponding modality output for each compound
            aug_2 = self.predictor_2(self.base_encoder(drug_indices, batch_mask_2, batch_mols, batch_kg, batch_cv, batch_tx_dict, raw_encoder_output=self.raw_encoder_output))

        # TODO: extra neg mol features
        # if batch_extra_mols is not None:
        torch.cuda.empty_cache()

        return aug_1, aug_2, self.contrastive_loss(aug_1, aug_2, batch_too_hard_neg_mask)

