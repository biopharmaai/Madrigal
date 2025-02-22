import os, shutil, logging, random, math
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Union
from itertools import chain, combinations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import HeteroData, Data
from torchdrug import models
from torchdrug.data import PackedMolecule

from .chemcpa.chemcpa_config_utils import generate_configs, read_config

import plotly.express as px
from umap import UMAP

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


SEED = 42
MOL_DIM = 67  # NOTE: torchdrug default
MAX_DRUGS = 25000
CELL_LINES = ['a375', 'a549', 'asc', 'ha1e', 'hcc515', 'hec108', 'hela', 'hepg2', 'ht29', 'huvec', 'mcf7', 'npc', 'pc3', 'thp1', 'vcap', 'yapc']  # NOTE: This list is ORDERED
CELL_LINES_CAPITALIZED = [cell_line.upper() for cell_line in CELL_LINES]
NUM_NON_TX_MODALITIES = 3  # str, kg, cv
NUM_MODALITIES = NUM_NON_TX_MODALITIES + len(CELL_LINES)


load_dotenv()
PROJECT_DIR = os.getenv("PROJECT_DIR")
DATA_DIR = os.getenv("DATA_DIR")
BASE_DIR = os.getenv("BASE_DIR")
ENCODER_CKPT_DIR = os.getenv("ENCODER_CKPT_DIR")
CL_CKPT_DIR = os.getenv("CL_CKPT_DIR")


######
# Masking
######
def get_pretrain_masks(drugs, masks, pretrain_mode, pretrain_unbalanced, pretrain_tx_downsample_ratio):
    """ Get all train subset masks for pretraining and finetuning
    """
    # NOTE: In all cases, we first need to construct a dictionary of all possible subsets of all possible masks of drugs
    unique_subset_masks = {}  # {tuple(modalities):(subsets, subset_probs)}

    # compute sample-balanced modality sampling probabilities
    if not pretrain_unbalanced:
        mod_probs = (1 / (1 - masks).sum(axis=0))
        assert pretrain_tx_downsample_ratio <= 1
        mod_probs[-len(CELL_LINES):] = pretrain_tx_downsample_ratio * mod_probs[-len(CELL_LINES):]
        mod_probs =  np.array(mod_probs / mod_probs.sum())
    
    # get all possible subset masks for all samples
    if pretrain_mode == 'double_random' or pretrain_mode == 'str_kg':
        for mask in np.unique(masks, axis=0):
            # The first subset in the output of powerset is always (), so remove it
            unique_subset_masks[tuple(mask)] = torch.stack([from_indices_to_tensor(list(indices), masks.shape[1]) for indices in list(powerset(np.where(mask==0)[0].tolist()))[1:]])  # in the final mask, 0 means not masked, 1 means masked
        
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}  # in the final mask, 0 means not masked, 1 means masked
    
    elif pretrain_mode == 'str_center' and pretrain_unbalanced:
        for mask in np.unique(masks, axis=0):
            mask[0] = 1  # NOTE: we don't want str to appear in the other branch as that would lead to shortcut solution, so set str (0) always to True (masked)
            # The first subset in the output of powerset is always (), so remove it
            unique_subset_masks[tuple(mask)] = torch.stack([from_indices_to_tensor(list(indices), masks.shape[1]) for indices in list(powerset(np.where(mask==0)[0].tolist()))[1:]])  # in the final mask, 0 means not masked, 1 means masked
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}  # in the final mask, 0 means not masked, 1 means masked
    
    elif pretrain_mode == 'str_center' and not pretrain_unbalanced:
        for mask in np.unique(masks, axis=0):
            # same as above, set str always to masked and remove the first subset in the output of powerset, which is always ()
            mask[0] = 1
            all_subsets = [from_indices_to_tensor(list(indices), masks.shape[1]).numpy() for indices in list(powerset(np.where(mask==0)[0].tolist()))[1:]]
            
            # for each of the subset, get the probability of it getting sampled
            # P({m1, ..., not n1, ...}) = P(m1) * ... * (1-P(n1)) * ... * (#all choose #m)
            subset_probs = []
            for subset in all_subsets:
                subset_probs.append(np.concatenate([mod_probs[np.where(subset==0)[0]], (1-mod_probs)[np.where(subset==1)[0]]]).prod() * math.comb((1-mask).sum(), (1-subset).sum()))
            subset_probs = np.array(subset_probs) / sum(subset_probs)
            
            unique_subset_masks[tuple(mask)] = (all_subsets, subset_probs)
        
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}

    elif pretrain_mode == 'str_center_uni' and pretrain_unbalanced:
        for mask in np.unique(masks, axis=0):    
            # Different from all above, here we only want unimodal masks.  The first subset in the output of torch.where is always 0 (structure), so remove it.
            unique_subset_masks[tuple(mask)] = torch.stack([from_indices_to_tensor([mod_index], masks.shape[1]) for mod_index in np.where(mask==0)[0][1:]])
            
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}
        
    elif pretrain_mode == 'str_center_uni' and not pretrain_unbalanced:
        for mask in np.unique(masks, axis=0):
            # Same as above, but we also need to compute the sampling probabilities
            all_subsets = [from_indices_to_tensor([mod_index], masks.shape[1]).numpy() for mod_index in np.where(mask==0)[0][1:]]
            
            # for each of the subset, get the (unnormalized) probability of it getting sampled
            subset_probs = []
            for subset in all_subsets:
                subset_probs.extend(mod_probs[np.where(subset==0)[0]])  # because each time we only sample one modality
            subset_probs = np.array(subset_probs) / sum(subset_probs)
            
            unique_subset_masks[tuple(mask)] = (all_subsets, subset_probs)
        
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}
    
    elif pretrain_mode == 'str_center_comb' and pretrain_unbalanced:  # in this case, the other branch always contains more than one modalities
        for mask in np.unique(masks, axis=0):
            mask[0] = 1  # NOTE: we don't want str to appear in the other branch as that would lead to shortcut solution, so set str (0) always to True (masked)
            # The first subset in the output of powerset is always (), so remove it
            unique_subset_masks[tuple(mask)] = torch.stack([from_indices_to_tensor(list(indices), masks.shape[1]) for indices in list(powerset(np.where(mask==0)[0].tolist()))[1:] if len(indices) > 1])  # in the final mask, 0 means not masked, 1 means masked
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}  # in the final mask, 0 means not masked, 1 means masked

    elif pretrain_mode == 'str_center_comb' and not pretrain_unbalanced:
        for mask in np.unique(masks, axis=0):
            # same as above, set str always to masked and remove the first subset in the output of powerset, which is always ()
            mask[0] = 1
            all_subsets = [from_indices_to_tensor(list(indices), masks.shape[1]).numpy() for indices in list(powerset(np.where(mask==0)[0].tolist()))[1:] if len(indices) > 1]
            
            # for each of the subset, get the probability of it getting sampled
            subset_probs = []
            for subset in all_subsets:
                subset_probs.append(np.concatenate([mod_probs[np.where(subset==0)[0]], (1-mod_probs)[np.where(subset==1)[0]]]).prod())
            subset_probs = np.array(subset_probs) / sum(subset_probs)
            
            unique_subset_masks[tuple(mask)] = (all_subsets, subset_probs)
        
        all_subset_masks = {ind:unique_subset_masks[tuple(mask)] for ind, mask in zip(drugs, masks)}
    
    else:
        raise NotImplementedError
    
    return all_subset_masks


def get_train_masks(finetune_mode, train_batch_drugs, masks):
    # TODO: changing fintune modes for masks
    pass


####
# Model utils
####
def get_model(
    all_kg_data, 
    feature_dim,
    prediction_dim,
    str_encoder_name, 
    str_encoder_hparams, 
    kg_encoder_name, 
    kg_encoder_hparams, 
    cv_encoder_name,
    cv_encoder_hparams,
    tx_encoder_name,
    tx_encoder_hparams,
    num_attention_bottlenecks,
    pos_emb_type,
    pos_emb_dropout,
    transformer_fusion_hparams,
    proj_hparams, 
    fusion,
    normalize,
    decoder_normalize,
    checkpoint_path,
    frozen,
    device,
    modality_pretrain_path=None,
    encoder_only=False,
    finetune_mode=None,
    str_node_feat_dim=MOL_DIM,
    logger=None,
    use_modality_pretrain=True,
    use_tx_basal=False,
    adapt_before_fusion=False,
    use_pretrained_adaptor=False,
):
    from madrigal.models.models import NovelDDIEncoder, NovelDDIMultilabel
    
    # encoder
    if checkpoint_path is None:
        encoder = NovelDDIEncoder(
            all_kg_data=all_kg_data, 
            feat_dim=feature_dim, 
            str_encoder_name=str_encoder_name, 
            str_encoder_hparams=str_encoder_hparams, 
            kg_encoder_name=kg_encoder_name,
            kg_encoder_hparams=kg_encoder_hparams, 
            cv_encoder_name=cv_encoder_name, 
            cv_encoder_hparams=cv_encoder_hparams, 
            tx_encoder_name=tx_encoder_name, 
            tx_encoder_hparams=tx_encoder_hparams, 
            num_tx_bottlenecks=num_attention_bottlenecks,
            pos_emb_dropout=pos_emb_dropout, 
            transformer_fusion_hparams=transformer_fusion_hparams, 
            proj_hparams=proj_hparams,
            fusion=fusion,
            str_node_feat_dim=str_node_feat_dim,
            use_modality_pretrain=use_modality_pretrain,
            normalize=normalize,
            pos_emb_type=pos_emb_type,
            use_tx_basal=use_tx_basal,
            adapt_before_fusion=adapt_before_fusion,
        )
        
        # encoder configs
        encoder_configs = {
            'all_kg_data': all_kg_data, 
            'feat_dim': feature_dim, 
            'str_encoder_name': str_encoder_name, 
            'str_encoder_hparams': str_encoder_hparams, 
            'kg_encoder_name': kg_encoder_name,
            'kg_encoder_hparams': kg_encoder_hparams, 
            'cv_encoder_name': cv_encoder_name, 
            'cv_encoder_hparams': cv_encoder_hparams, 
            'tx_encoder_name': tx_encoder_name, 
            'tx_encoder_hparams': tx_encoder_hparams,
            'num_tx_bottlenecks': num_attention_bottlenecks,
            'pos_emb_type': pos_emb_type,
            'pos_emb_dropout': pos_emb_dropout, 
            'transformer_fusion_hparams': transformer_fusion_hparams, 
            'proj_hparams': proj_hparams,
            'fusion': fusion,
            'str_node_feat_dim': str_node_feat_dim,
            'use_modality_pretrain': use_modality_pretrain,
            'normalize': normalize,
            'adapt_before_fusion': adapt_before_fusion,
        }
        
    else:
        full_ckpt_path = CL_CKPT_DIR+checkpoint_path
        if os.path.isfile(full_ckpt_path):
            if logger is not None:
                logger.info("Loading checkpoint at '{}'".format(full_ckpt_path))
            else:
                print("Loading checkpoint at '{}'".format(full_ckpt_path))
            
            checkpoint = torch.load(full_ckpt_path, map_location="cpu")
            encoder_configs = checkpoint['encoder_configs']
            state_dict = checkpoint['state_dict']
            
            # replace the hyperparameters other than encoder-related ones in encoder_configs with the desired ones
            # assert (not use_tx_basal)
            for k, new_v in zip(
                ['num_tx_bottlenecks', 'pos_emb_type', 'pos_emb_dropout', 'transformer_fusion_hparams', 'proj_hparams', 'fusion', 'normalize', 'use_tx_basal', 'adapt_before_fusion'],
                [num_attention_bottlenecks, pos_emb_type, pos_emb_dropout, transformer_fusion_hparams, proj_hparams, fusion, normalize, use_tx_basal, adapt_before_fusion],
            ):
                if new_v is not None:
                    encoder_configs[k] = new_v
            
            # NOTE: Since we still want to do ablation studies with pretrained models, we need to revise the encoder_configs according to specified `finetune_mode`
            # encoder_configs['finetune_mode'] = finetune_mode
            encoder = NovelDDIEncoder(**encoder_configs)
            
            if 'epoch' in checkpoint.keys():
                start_epoch = checkpoint['epoch']
            
            # process state_dict
            # if finetune_mode is not None and 'ablation' in finetune_mode:  # NOTE: pe should be re-initialized in the ablation models
            #     del state_dict['base_encoder.pos_encoder.pe']
            for k in list(state_dict.keys()):
                # NOTE: retain only base_encoder up to before the embedding layer, excluding fusion-related modules
                if (
                    k.startswith('base_encoder') and
                    not k.startswith('base_encoder.head') and 
                    k not in {'base_encoder.tx_bottleneck_tokens', 'base_encoder.cls'} and
                    not k.startswith('base_encoder.pos_encoder') and
                    not k.startswith('base_encoder.transformer')
                ):
                    if (not use_pretrained_adaptor) and k.startswith('base_encoder.uni_projector'):
                        continue
                    # remove prefix
                    state_dict[k[len("base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
                
            msg = encoder.load_state_dict(state_dict, strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # assert msg.missing_keys == [] or msg.missing_keys[0] == 'pos_encoder.pe'
            if logger is not None:
                logger.info("=> Missing keys {}".format(msg.missing_keys))
                logger.info("=> Unexpected keys {}".format(msg.unexpected_keys))
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(full_ckpt_path, checkpoint['epoch']))
            else:
                print("=> Missing keys {}".format(msg.missing_keys))
                print("=> Unexpected keys {}".format(msg.unexpected_keys))
                print("=> loaded checkpoint '{}' (epoch {})".format(full_ckpt_path, checkpoint['epoch']))
                
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(full_ckpt_path))
    
    if encoder_only:
        if device is not None:
            encoder.to(device)
        return encoder, encoder_configs
    
    # full model
    model = NovelDDIMultilabel(
        encoder, 
        feat_dim=feature_dim, 
        prediction_dim=prediction_dim, 
        normalize=decoder_normalize,
    )
    if device is not None:
        model = model.to(device)  # TODO: deal with DDP case
    
    if frozen:
        for param in encoder.parameters():
            param.requires_grad = False
    
    # model configs (in addition to encoder configs)
    model_configs = {
        'feat_dim': feature_dim, 
        'prediction_dim': prediction_dim, 
        'normalize': normalize,
    }
    
    return model, encoder_configs, model_configs
    

####
# Plotting utils
####
@torch.no_grad()
def draw_umap_plot(embeds, encoder, drug_ids, drug_loader, masks, collator, plotname, wandb, device, logger, epoch, output_dir=None, other_labels=None, raw_encoder_output=False):
    str_mod_ind = 0
    kg_mod_ind = 1
    cv_mod_ind = 2
    tx_mcf7_mod_ind = 13
    tx_pc3_mod_ind = 15
    tx_vcap_mod_ind = 17
    
    if embeds is not None:
        str_embedding = embeds[str(str_mod_ind)]
        kg_embedding = embeds[str(kg_mod_ind)]
        cv_embedding = embeds[str(cv_mod_ind)]
        tx_mcf7_embedding = embeds[str(tx_mcf7_mod_ind)]  # TODO: Find a better way to do this...
        tx_pc3_embedding = embeds[str(tx_pc3_mod_ind)]
        tx_vcap_embedding = embeds[str(tx_vcap_mod_ind)]
        if drug_ids is not None:
            drug_ids = drug_ids[str(str_mod_ind)] + drug_ids[str(kg_mod_ind)] + drug_ids[str(cv_mod_ind)] + drug_ids[str(tx_mcf7_mod_ind)] + drug_ids[str(tx_pc3_mod_ind)] + drug_ids[str(tx_vcap_mod_ind)]
    
    elif drug_ids is not None:
        if isinstance(drug_ids, list):
            drug_ids = np.array(drug_ids)
        
        masks = masks[drug_ids]
        _, modality_data = to_device(collator([drug_ids]), device)
        mols, kgs, cvs, tx_all_cell_lines = modality_data
        assert drug_ids.shape[0] == masks.shape[0]
        mod_avail = 1 - masks
        
        keep_str_mask = torch.ones(masks.shape[1])
        keep_str_mask[str_mod_ind] = 0
        keep_str_mask = to_device(keep_str_mask.repeat(mod_avail.sum(axis=0)[str_mod_ind], 1).bool(), device)
        keep_kg_mask = torch.ones(masks.shape[1])
        keep_kg_mask[kg_mod_ind] = 0
        keep_kg_mask = to_device(keep_kg_mask.repeat(mod_avail.sum(axis=0)[kg_mod_ind], 1).bool(), device)
        keep_cv_mask = torch.ones(masks.shape[1])
        keep_cv_mask[cv_mod_ind] = 0
        keep_cv_mask = to_device(keep_cv_mask.repeat(mod_avail.sum(axis=0)[cv_mod_ind], 1).bool(), device)
        keep_tx_mcf7_mask = torch.ones(masks.shape[1])
        keep_tx_mcf7_mask[tx_mcf7_mod_ind] = 0
        keep_tx_mcf7_mask = to_device(keep_tx_mcf7_mask.repeat(mod_avail.sum(axis=0)[tx_mcf7_mod_ind], 1).bool(), device)
        keep_tx_pc3_mask = torch.ones(masks.shape[1])
        keep_tx_pc3_mask[tx_pc3_mod_ind] = 0
        keep_tx_pc3_mask = to_device(keep_tx_pc3_mask.repeat(mod_avail.sum(axis=0)[tx_pc3_mod_ind], 1).bool(), device)
        keep_tx_vcap_mask = torch.ones(masks.shape[1])
        keep_tx_vcap_mask[tx_vcap_mod_ind] = 0
        keep_tx_vcap_mask = to_device(keep_tx_vcap_mask.repeat(mod_avail.sum(axis=0)[tx_vcap_mod_ind], 1).bool(), device)

        encoder.eval()
        drug_ids = torch.from_numpy(drug_ids)
        str_indices_bool = torch.from_numpy(masks[:, str_mod_ind] == 0)
        str_embedding = encoder(drug_ids[str_indices_bool].to(device), keep_str_mask, mols[str_indices_bool], kgs, cvs[str_indices_bool], {cell_line: {k: v[str_indices_bool] for k, v in tx_cell_line.items()} for cell_line, tx_cell_line in tx_all_cell_lines.items()}, raw_encoder_output=raw_encoder_output).cpu().numpy()
        kg_indices_bool = torch.from_numpy(masks[:, kg_mod_ind] == 0)
        kg_embedding = encoder(drug_ids[kg_indices_bool].to(device), keep_kg_mask, mols[kg_indices_bool], kgs, cvs[kg_indices_bool], {cell_line: {k: v[kg_indices_bool] for k, v in tx_cell_line.items()} for cell_line, tx_cell_line in tx_all_cell_lines.items()}, raw_encoder_output=raw_encoder_output).cpu().numpy()
        cv_indices_bool = torch.from_numpy(masks[:, cv_mod_ind] == 0)
        cv_embedding = encoder(drug_ids[cv_indices_bool].to(device), keep_cv_mask, mols[cv_indices_bool], kgs, cvs[cv_indices_bool], {cell_line: {k: v[cv_indices_bool] for k, v in tx_cell_line.items()} for cell_line, tx_cell_line in tx_all_cell_lines.items()}, raw_encoder_output=raw_encoder_output).cpu().numpy()
        tx_mcf7_indices_bool = torch.from_numpy(masks[:, tx_mcf7_mod_ind] == 0)
        tx_mcf7_embedding = encoder(drug_ids[tx_mcf7_indices_bool].to(device), keep_tx_mcf7_mask, mols[tx_mcf7_indices_bool], kgs, cvs[tx_mcf7_indices_bool], {cell_line: {k: v[tx_mcf7_indices_bool] for k, v in tx_cell_line.items()} for cell_line, tx_cell_line in tx_all_cell_lines.items()}, raw_encoder_output=raw_encoder_output).cpu().numpy()
        tx_pc3_indices_bool = torch.from_numpy(masks[:, tx_pc3_mod_ind] == 0)
        tx_pc3_embedding = encoder(drug_ids[tx_pc3_indices_bool].to(device), keep_tx_pc3_mask, mols[tx_pc3_indices_bool], kgs, cvs[tx_pc3_indices_bool], {cell_line: {k: v[tx_pc3_indices_bool] for k, v in tx_cell_line.items()} for cell_line, tx_cell_line in tx_all_cell_lines.items()}, raw_encoder_output=raw_encoder_output).cpu().numpy()
        tx_vcap_indices_bool = torch.from_numpy(masks[:, tx_vcap_mod_ind] == 0)
        tx_vcap_embedding = encoder(drug_ids[tx_vcap_indices_bool].to(device), keep_tx_vcap_mask, mols[tx_vcap_indices_bool], kgs, cvs[tx_vcap_indices_bool], {cell_line: {k: v[tx_vcap_indices_bool] for k, v in tx_cell_line.items()} for cell_line, tx_cell_line in tx_all_cell_lines.items()}, raw_encoder_output=raw_encoder_output).cpu().numpy()

        drug_ids = drug_ids[str_indices_bool].tolist() + drug_ids[kg_indices_bool].tolist() + drug_ids[cv_indices_bool].tolist() + drug_ids[tx_mcf7_indices_bool].tolist() + drug_ids[tx_pc3_indices_bool].tolist() + drug_ids[tx_vcap_indices_bool].tolist()
    
    # if full batch inference is not feasible (which will likely not be the case for this project)
    elif drug_loader is not None:
        raise NotImplementedError
        # for i, batch_drug_indices in enumerate(train_loader):
        #     # measure data loading time
        #     data_time.update(time.time() - end)
        #     batch_drug_indices = batch_drug_indices[0]  # NOTE: Because of the way TensorDataset loads data, [tensor([xxx])]
        #     batch_mols = all_molecules[batch_drug_indices].to(device)
        #     batch_cv = cv_store[batch_drug_indices].to(device)
        #     batch_tx_mcf7 = tx_mcf7_store[batch_drug_indices].to(device)
        #     batch_tx_pc3 = tx_pc3_store[batch_drug_indices].to(device)
        #     batch_tx_vcap = tx_vcap_store[batch_drug_indices].to(device)

        #     # adjust learning rate and momentum coefficient per iteration
        #     lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, hparams)
        #     learning_rates.update(lr)
        #     # if hparams['moco_m_cos']:
        #     #     moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, hparams)
            
        #     # Get two (subset sampling) masks for each drug in the batch.  Note that in `MoCo_NovelDDI`, we directly feed the input mask to the encoder, so they must already be aligned with the drugs (rather than being aligned in `NovelDDI`).
        #     batch_mask1, batch_mask2 = pretrain_modality_subset_sampler([all_train_subset_masks[drug_ind.item()] for drug_ind in batch_drug_indices], pretrain_mode=pretrain_mode, unbalanced=unbalanced)
        #     batch_mask1, batch_mask2 = batch_mask1.to(device), batch_mask2.to(device)
        #     if args.too_hard_neg_mask:
        #         batch_hard_negative_mask = hard_negative_mask[batch_drug_indices.tolist(), :][: , batch_drug_indices.tolist()].to(device)
        #     else:
        #         batch_hard_negative_mask = None
                
        #     # Get extra negative molecules for the batch (if applicable)
        #     if all_extra_molecules is not None and extra_mol_num > 0:
        #         batch_extra_mols = all_extra_molecules[np.random.choice(all_extra_molecules.shape[0], extra_mol_num, replace=False)]
        #         batch_extra_mols = batch_extra_mols.to(device)
        #     else:
        #         batch_extra_mols = None
            
        #     optimizer.zero_grad()
            
        #     # compute output
        #     # loss = model(batch_drug_indices, batch_mask1, batch_mask2, batch_mols, moco_m)
        #     _, _, (logits, labels, loss) = model(batch_drug_indices, batch_mask1, batch_mask2, batch_hard_negative_mask, batch_mols, batch_cv, batch_tx_mcf7, batch_tx_pc3, batch_tx_vcap, batch_extra_mols, extra_mol_str_masks)

    full_embeddings = np.concatenate((str_embedding, kg_embedding, cv_embedding, tx_mcf7_embedding, tx_pc3_embedding, tx_vcap_embedding))
    modality_labels = ['str'] * str_embedding.shape[0] + ['kg'] * kg_embedding.shape[0] + ['cv'] * cv_embedding.shape[0] + ['tx_mcf7'] * tx_mcf7_embedding.shape[0] + ['tx_pc3'] * tx_pc3_embedding.shape[0] + ['tx_vcap'] * tx_vcap_embedding.shape[0]
    if other_labels is not None:
        other_labels = other_labels[masks[:, str_mod_ind] == 0].tolist() + other_labels[masks[:, kg_mod_ind] == 0].tolist() + other_labels[masks[:, cv_mod_ind] == 0].tolist() + other_labels[masks[:, tx_mcf7_mod_ind] == 0].tolist() + other_labels[masks[:, tx_pc3_mod_ind] == 0].tolist() + other_labels[masks[:, tx_vcap_mod_ind] == 0].tolist()
    
    logger.info("=> Fitting UMAP")
    
    # NOTE: 'cosine' can lead to Segmentation Fault
    dat = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42).fit_transform(full_embeddings)  # Use 'euclidean' if 'cosine' leads to Segmentation Fault
    
    if other_labels is None:
        if drug_ids is not None:
            fig = px.scatter(x=dat[:,0], y=dat[:,1], color=modality_labels, hover_data={'drug_id':drug_ids})
        else:
            fig = px.scatter(x=dat[:,0], y=dat[:,1], color=modality_labels)
    else:
        if drug_ids is not None:
            fig = px.scatter(x=dat[:,0], y=dat[:,1], color=other_labels, hover_data={'drug_id':drug_ids})
        else:
            fig = px.scatter(x=dat[:,0], y=dat[:,1], color=other_labels)

    if wandb is not None:
        wandb.log({plotname:fig}, step=epoch)
    if output_dir is not None:
        fig.write_image(output_dir+plotname)


######
# Hook utils
######
def get_activation(name, activation):
    def hook(module, input, output):
        if name in activation.keys():
            activation[name].append(output)
        else:
            activation[name] = [output]
    return hook


######
# Pretraining modality sampler
######
# First derive a "modality subset bank", then in each epoch sample from the bank
def pretrain_modality_subset_sampler(all_subset_masks: Iterable, pretrain_mode: str = 'str_center_uni', unbalanced: bool = False):
    if pretrain_mode in {'str_center', 'str_center_uni', 'str_center_comb'}:
        if not unbalanced:  # In this case the all_subset_masks is a list of tuples of (subset_masks, subset_probs)
            aug1 = torch.ones_like(torch.from_numpy(all_subset_masks[0][0][0]))  # first mask list & prob tuple -> first mask list -> first mask
            aug1[0] = 0
            aug1 = aug1.repeat(len(all_subset_masks), 1).bool()
            aug2 = torch.stack([torch.from_numpy(subset_masks[np.random.choice(np.arange(len(subset_masks)), size=1, p=subset_probs)[0]]) for subset_masks, subset_probs in all_subset_masks], dim=0).bool()
            
        else:
            aug1 = torch.ones_like(all_subset_masks[0][0])  # first mask list -> first mask (subset)
            aug1[0] = 0
            aug1 = aug1.repeat(len(all_subset_masks), 1).bool()
            aug2 = torch.stack([subset_masks[torch.randint(len(subset_masks), (1,))[0].item()] for subset_masks in all_subset_masks], dim=0).bool()

    elif pretrain_mode == 'double_random':
        # Scale up for all possible modality subsets for arbitrary number of views
        sampled_masks = torch.stack([subset_masks[[torch.randperm(len(subset_masks))[:2]]] for subset_masks in all_subset_masks], dim=0)
        aug1 = sampled_masks[:, 0, :].bool()
        aug2 = sampled_masks[:, 1, :].bool()
    
    elif pretrain_mode == 'str_kg':
        aug1, aug2 = torch.ones_like(all_subset_masks[0][0]), torch.ones_like(all_subset_masks[0][0])
        aug1[0] = 0
        aug2[1] = 0
        aug1 = aug1.repeat(len(all_subset_masks), 1).bool()
        aug2 = aug2.repeat(len(all_subset_masks), 1).bool()
        
    else:
        raise NotImplementedError

    return aug1, aug2


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def from_indices_to_tensor(indices: Iterable, size: Union[tuple, int], value: Union[int, float] = 0, base_value: Union[int, float] = 1, dim: int = -1):
    """ Transform a tensor of indices (where the dimensions except for the `dim` dimension are the same as expected output `size`) to a tensor of size `size` with value at the `indices` (over dim `dim`) and `base_value` elsewhere
    E.g., from_indices_to_tensor([[0, 1], [1, 2]], (2, 3), value=1, base_value=0, dim=1) -> [[1, 1, 0], [0, 1, 1]]
    """
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices)
    if len(indices.shape)>1:
        assert len(indices.shape) == len(size)
        assert indices.shape[0] == size[0]
    out = torch.ones(size) * base_value
    # out[indices] = 0
    return out.scatter(dim=dim, index=indices, value=value)


#######
# Saving utils
#######
def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt')


# TODO: Update
@torch.no_grad()
def save_embeds(encoder, train_drugs, val_drugs, masks, collator, save_dir, device, raw_encoder_output=False):
    encoder.eval()
    # full_mask = torch.zeros_like(masks[0]).bool()

    # TODO: Save attention weights (see `predict.py`)
    # activation = {}
    # activation_order = []
    # encoder.transformer.transformer_encoder.layers[-1].self_attn.register_forward_hook(get_activation('last-layer-attention', activation))

    train_outputs = {}
    val_outputs = {}
    
    for index, subset_mask in enumerate(torch.eye(masks.shape[1])):
        # NOTE: only save for str, kg, cv, tx_mcf7, tx_pc3, tx_vcap
        if index not in {0, 1, 2, 13, 15, 17}:
            continue
        # indices = torch.where(subset_mask==0)[0]
        # val_valid_drugs = val_drugs[(1 - masks[val_drugs, :][:, indices]).sum(axis=1) == len(indices)]  # Get val drugs that have the required modalities (subset1 and subset2)
        val_valid_drugs = val_drugs[masks[val_drugs, :][:, index] == 0]  # Get val drugs that have the required modality
        val_valid_drugs, val_valid_data = to_device(collator([val_valid_drugs]), device)
        val_valid_mols, val_valid_kgs, val_valid_cvs, val_valid_tx_all_cell_lines = val_valid_data
        val_valid_masks = to_device(~(subset_mask.repeat(val_valid_drugs.shape[0], 1).bool()), device)
        
        val_valid_embeds = encoder(val_valid_drugs, val_valid_masks, val_valid_mols, val_valid_kgs, val_valid_cvs, val_valid_tx_all_cell_lines, raw_encoder_output=raw_encoder_output).cpu()
        
        # indices_str = ''.join(np.array(indices.tolist()).astype(str))
        # if len(indices) > 1:
        #     activation_order.append(f'val_{indices_str}')
        if save_dir is not None:
            torch.save({'drugs':val_valid_drugs.detach().cpu().numpy(), 'embeds':val_valid_embeds, 'masks':masks[val_valid_drugs.detach().cpu().numpy()]}, save_dir + f'/val_embeds_{index}.pt')
        
        # val_outputs[indices_str] = {}
        # val_outputs[indices_str]['embeds'] = val_valid_embeds
        # val_outputs[indices_str]['drugs'] = val_valid_drugs
        val_outputs[str(index)] = {}
        val_outputs[str(index)]['embeds'] = val_valid_embeds
        val_outputs[str(index)]['drugs'] = val_valid_drugs.detach().cpu().numpy()

        # train_valid_drugs = train_drugs[(1 - masks[train_drugs, :][:, indices]).sum(axis=1) == len(indices)]  # Get train drugs that have the required modalities (subset1 and subset2)
        train_valid_drugs = train_drugs[masks[train_drugs, :][:, index] == 0]  # Get train drugs that have the required modality
        train_valid_drugs, train_valid_data = to_device(collator([train_valid_drugs]), device)
        train_valid_mols, train_valid_kgs, train_valid_cvs, train_valid_tx_all_cell_lines = train_valid_data
        train_valid_masks = to_device(~(subset_mask.repeat(train_valid_drugs.shape[0], 1).bool()), device)
        
        train_valid_embeds = encoder(train_valid_drugs, train_valid_masks, train_valid_mols, train_valid_kgs, train_valid_cvs, train_valid_tx_all_cell_lines, raw_encoder_output=raw_encoder_output).cpu()
        
        # if len(indices) > 1:
            # activation_order.append(f'train_{indices_str}')
        if save_dir is not None:
            torch.save({'drugs':train_valid_drugs.detach().cpu().numpy(), 'embeds':train_valid_embeds, 'masks':masks[train_valid_drugs.detach().cpu().numpy()]}, save_dir + f'/train_embeds_{index}.pt')

        train_outputs[str(index)] = {}
        train_outputs[str(index)]['embeds'] = train_valid_embeds
        train_outputs[str(index)]['drugs'] = train_valid_drugs.detach().cpu().numpy()
    
    # if save_dir is not None:
        # torch.save(activation.update({'order':activation_order}), save_dir + f'/attention_weights_{epoch}.pt')

    return train_outputs, val_outputs


######
# Optimization utils
######
def to_device(input, device):
    if isinstance(input, dict):
        for k, v in input.items():
            input[k] = to_device(v, device)
        return input
    elif isinstance(input, (list, tuple)):
        return [to_device(inp, device) for inp in input]
    elif isinstance(input, (torch.Tensor, PackedMolecule, HeteroData, Data)):  # NOTE: don't put some random stuff in the batch
        return input.to(device)
    else:  # np.ndarray, pd.DataFrame, int, float, str, bool, etc.
        return input
    
    
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def get_parameter_names(model, forbidden_layer_types):
    """
    Adapted from transformers.trainer_pt_utils.get_parameter_names (change the logic of nn.Parameter)
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(name for name in model._parameters.keys() if 'cls' not in name and 'bottleneck_tokens' not in name)
    return result


def create_optimizer(model, hparams):
    """
    Enable independent learning rates for different parts of the model.
    """
    from madrigal.models.models import HAN, HGT, RGCN, MLPEncoder, TransformerFusion, PositionEncodingSinusoidal, PositionEncodingLearnable, PositionEncodingRotary, MLPAdaptor, BilinearDDIScorer, NovelDDIEncoder, NovelDDIMultilabel
    from madrigal.chemcpa.chemCPA.model import TxAdaptingComPert
    
    decay_params = get_parameter_names(model, [nn.LayerNorm])
    decay_params = [name for name in decay_params if "bias" not in name]
    
    str_encoder_params = get_parameter_names(model, [HAN, HGT, RGCN, MLPEncoder, TxAdaptingComPert, MLPAdaptor, PositionEncodingSinusoidal, PositionEncodingLearnable, PositionEncodingRotary, TransformerFusion, BilinearDDIScorer])  # Either GIN or GAT
    kg_encoder_params = get_parameter_names(model, [models.GraphAttentionNetwork, models.GraphIsomorphismNetwork, MLPEncoder, TxAdaptingComPert, MLPAdaptor, PositionEncodingSinusoidal, PositionEncodingLearnable, PositionEncodingRotary, TransformerFusion, BilinearDDIScorer])  # Either HGT, HAN, or RGCN
    cv_encoders_params = get_parameter_names(model, [HAN, HGT, RGCN, models.GraphAttentionNetwork, models.GraphIsomorphismNetwork, TxAdaptingComPert, MLPAdaptor, PositionEncodingSinusoidal, PositionEncodingLearnable, PositionEncodingRotary, TransformerFusion, BilinearDDIScorer])
    tx_encoders_params = get_parameter_names(model, [HAN, HGT, RGCN, models.GraphAttentionNetwork, models.GraphIsomorphismNetwork, MLPEncoder, MLPAdaptor, PositionEncodingSinusoidal, PositionEncodingLearnable, PositionEncodingRotary, TransformerFusion, BilinearDDIScorer])
    fusion_params = get_parameter_names(model, [HAN, HGT, RGCN, models.GraphAttentionNetwork, models.GraphIsomorphismNetwork, MLPEncoder, TxAdaptingComPert, BilinearDDIScorer])  # MLP, TransformerFusion, PositionEncoding
    fusion_params += list(model._parameters.keys())  # NOTE: Add [CLS] and [TX_BOTTLENECK] into fusion_params
    decoder_params = get_parameter_names(model, [NovelDDIEncoder])  # just the decoder
    
    str_encoder_no_decay_params = list(
        set(str_encoder_params) - set(decay_params)
    )
    str_encoder_decay_params = list(
        set(str_encoder_params) & set(decay_params)
    )

    kg_encoder_no_decay_params = list(set(kg_encoder_params) - set(decay_params))
    kg_encoder_decay_params = list(set(kg_encoder_params) & set(decay_params))
    
    cv_encoders_no_decay_params = list(set(cv_encoders_params) - set(decay_params))
    cv_encoders_decay_params = list(set(cv_encoders_params) & set(decay_params))

    tx_encoders_no_decay_params = list(set(tx_encoders_params) - set(decay_params))
    tx_encoders_decay_params = list(set(tx_encoders_params) & set(decay_params))
    
    fusion_no_decay_params = list(set(fusion_params) - set(decay_params))
    fusion_decay_params = list(set(fusion_params) & set(decay_params))

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in str_encoder_no_decay_params
            ],
            "weight_decay": 0.0,
            "lr": hparams['structure_encoder_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in str_encoder_decay_params
            ],
            "weight_decay": hparams['wd'],
            "lr": hparams['structure_encoder_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in kg_encoder_no_decay_params
            ],
            "weight_decay": 0.0,
            "lr": hparams['kg_encoder_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in kg_encoder_decay_params
            ],
            "weight_decay": hparams['wd'],
            "lr": hparams['kg_encoder_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in cv_encoders_no_decay_params
            ],
            "weight_decay": 0.0,
            "lr": hparams['perturb_encoders_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in cv_encoders_decay_params
            ],
            "weight_decay": hparams['wd'],
            "lr": hparams['perturb_encoders_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in tx_encoders_no_decay_params
            ],
            "weight_decay": 0.0,
            "lr": hparams['perturb_encoders_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in tx_encoders_decay_params
            ],
            "weight_decay": hparams['wd'],
            "lr": hparams['perturb_encoders_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in fusion_no_decay_params
            ],
            "weight_decay": 0.0,
            "lr": hparams['fusion_lr'],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in fusion_decay_params
            ],
            "weight_decay": hparams['wd'],
            "lr": hparams['fusion_lr'],
        },
        {
            "params": [
                p 
                for n, p in model.named_parameters() 
                if n in decoder_params
            ],
            "weight_decay": hparams['wd'],
            "lr": hparams['decoder_lr'],
        },
    ]

    OPTIMIZER_CLASSES = {"radam": torch.optim.RAdam, "adamw": torch.optim.AdamW}

    # TODO: add support for Adafactor and LARS
    optimizer_class = OPTIMIZER_CLASSES[hparams['optimizer']]
    optimizer_kwargs = {
        "betas": (hparams['beta1'], hparams['beta2']),
        "eps": hparams['eps'],
    }

    return optimizer_class(
        optimizer_grouped_parameters, **optimizer_kwargs
    )


def get_loss_fn(loss_fn_name, task, loss_readout):
    if loss_fn_name == 'bce':
        # NOTE: note that multiclass task can also use BCE loss with negative sampling
        loss_fn = nn.BCELoss(reduction=loss_readout)
    elif loss_fn_name == 'ce' and task == 'multiclass':
        loss_fn = nn.CrossEntropyLoss(reduction=loss_readout)
    else:
        raise NotImplementedError("Loss function {} not implemented for task {}".format(loss_fn_name, task))

    return loss_fn


class LARS(torch.optim.Optimizer):
    """
    Copied from https://github.com/facebookresearch/moco-v3/blob/main/moco/optimizer.py
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])


class LinearWarmupCosineDecaySchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, num_cycles=1., last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.num_cycles = num_cycles
        super(LinearWarmupCosineDecaySchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cur_epoch = self.last_epoch
        if cur_epoch < self.warmup_epochs:
            return [base_lr * cur_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            t = (cur_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [base_lr * (1 + math.cos(math.pi * self.num_cycles * t)) / 2 \
                for base_lr in self.base_lrs]


def adjust_learning_rate(optimizer, cur_epoch, lr, warmup_epochs, num_epochs):
    """
    During warmup, increase lr linearly. Then decays the learning rate with half-cycle cosine after warmup.
    """
    # TODO: Update code so that we have separate lr for separate param groups
    if cur_epoch < warmup_epochs:
        lr = lr * cur_epoch / warmup_epochs
    else:
        lr = lr * 0.5 * (1. + math.cos(math.pi * (cur_epoch - warmup_epochs) / (num_epochs - warmup_epochs)))  # cosine decay schedule
        # lr = lr * (1 - (cur_epoch - warmup_epochs) / (num_epochs - warmup_epochs))  # linear decay schedule
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


######
# Logging utils
######
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_root_logger(fname=None, file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("[%(asctime)-10s] %(message)s", '%m/%d/%Y %H:%M:%S')

    if file:
        handler = logging.FileHandler(fname, mode='w')
        handler.setFormatter(format)
        logger.addHandler(handler)

    logger.addHandler(logging.StreamHandler())
    
    return logger


######
# Load model hparams
######
def get_str_encoder_hparams(args, hparams):
    if args.str_encoder == 'gat':
        structural_encoder_hparams = {hparam_name:hparams[hparam_name] for hparam_name in ['gat_hidden_dims', 'gat_edge_input_dim', 'gat_att_heads', 'gat_negative_slope', 'gat_batch_norm', 'gat_actn', 'gat_readout']}

    elif args.str_encoder == 'gin':
        structural_encoder_hparams = {hparam_name:hparams[hparam_name] for hparam_name in ['gin_hidden_dims', 'gin_edge_input_dim', 'gin_num_mlp_layer', 'gin_eps', 'gin_batch_norm', 'gin_actn', 'gin_readout']}

    else:
        raise NotImplementedError
    
    return structural_encoder_hparams


def get_kg_encoder_hparams(args, hparams):
    if 'han' in args.kg_encoder:
        kg_encoder_hparams = {hparam_name:hparams[hparam_name] for hparam_name in ['han_att_heads', 'han_hidden_dim', 'han_num_layers', 'han_negative_slope', 'han_dropout']}
    elif 'hgt' in args.kg_encoder:
        kg_encoder_hparams = {hparam_name:hparams[hparam_name] for hparam_name in ['hgt_hidden_dim', 'hgt_num_layers', 'hgt_att_heads', 'hgt_group']}
    # elif 'rgcn' in args.kg_encoder:
    #     kg_encoder_hparams = (hparams['rgcn_hidden_dim'], hparams['rgcn_num_layers'], hparams['rgcn_num_bases'], hparams['rgcn_aggr'], hparams['rgcn_actn'])
    else:
        raise NotImplementedError
    return kg_encoder_hparams


def get_cv_encoder_hparams(args, hparams, cv_input_dim):
    if args.cv_encoder == 'mlp':
        cv_encoder_hparams = {'cv_input_dim': cv_input_dim}
        cv_encoder_hparams.update({hparam_name:hparams[hparam_name] for hparam_name in ['cv_mlp_hidden_dims', 'cv_mlp_dropout', 'cv_mlp_norm', 'cv_mlp_actn', 'cv_mlp_order']})
    else:
        raise NotImplementedError
    return cv_encoder_hparams


def get_tx_encoder_hparams(args, hparams, tx_input_dim):
    if args.tx_encoder == 'mlp':
        tx_encoder_hparams = {'tx_input_dim': tx_input_dim}
        tx_encoder_hparams.update({hparam_name:hparams[hparam_name] for hparam_name in ['tx_mlp_hidden_dims', 'tx_mlp_dropout', 'tx_mlp_norm', 'tx_mlp_actn', 'tx_mlp_order']})
    elif args.tx_encoder == 'chemcpa':
        _, _, experiment_config = read_config(PROJECT_DIR + hparams['tx_chemcpa_config_path'])
        configs = generate_configs(experiment_config)
        assert len(configs) == 1
        tx_encoder_hparams = configs[0]
    else:
        raise NotImplementedError
    return tx_encoder_hparams


def get_proj_hparams(hparams):
    proj_hparams = {hparam_name:hparams[hparam_name] for hparam_name in ['proj_hidden_dims', 'proj_dropout', 'proj_norm', 'proj_actn', 'proj_order']}
    return proj_hparams


def get_transformer_fusion_hparams(args, hparams):
    transformer_hparams =  {hparam_name:hparams[hparam_name] for hparam_name in ['transformer_num_layers', 'transformer_att_heads', 'transformer_head_dim', 'transformer_ffn_dim', 'transformer_dropout', 'transformer_actn', 'transformer_norm_first']}
    transformer_hparams['transformer_batch_first'] = args.transformer_batch_first
    transformer_hparams['transformer_agg'] = args.transformer_agg
    return transformer_hparams
