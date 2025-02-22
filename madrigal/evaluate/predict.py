import pandas as pd
import numpy as np
import torch, json, wandb

from ..utils import get_activation
from ..models.models import NovelDDIEncoder, NovelDDIMultilabel
from .evaluate import evaluate_ft
from .eval_utils import K
from .metrics import get_metrics


@torch.no_grad()
def test(best_epoch, best_within_epoch, test_loaders, loss_fn, task, finetune_mode, output_dir, split_method, label_map, device, logger, wandb=None):
    # NOTE: Work-around for full-batch training
    test_batches = [next(iter(test_loader)) for test_loader in test_loaders]
    
    # first test "test between" or "test" split
    checkpoint = torch.load(output_dir+'best_model.pt')
    encoder = NovelDDIEncoder(**checkpoint['encoder_configs'])
    best_model = NovelDDIMultilabel(encoder, **checkpoint['model_configs'])
    best_model.load_state_dict(checkpoint['state_dict'])
    best_model.eval()
    
    test_metrics = {}
    if split_method in {'split_by_triplets', 'split_by_pairs'}:
        logger.info("Test:")
        test_key_metric = evaluate_ft(best_model, test_batches[0], loss_fn, k=K, task=task, split='test', finetune_mode=finetune_mode, best_metrics=test_metrics, subgroup=False, verbose=True, device=device, logger=logger, wandb=wandb, save_scores=True, output_dir=output_dir, label_map=label_map)
        
    else:
        within_checkpoint = torch.load(output_dir+'best_within_model.pt')
        within_encoder = NovelDDIEncoder(**within_checkpoint['encoder_configs'])
        best_within_model = NovelDDIMultilabel(within_encoder, **within_checkpoint['model_configs'])
        best_within_model.load_state_dict(within_checkpoint['state_dict'])
        best_within_model.eval()
        
        if 'easy' in split_method:  # NOTE: the case where val/test drugs are all full modality and we want to evaluate more comprehensively
            logger.info("Test (between):")
            test_key_metric = evaluate_ft(best_model, test_batches[0], loss_fn, k=K, task=task, split='test_between_easy', finetune_mode=finetune_mode, best_metrics=test_metrics, subgroup=False, verbose=True, device=device, logger=logger, wandb=wandb, save_scores=True, output_dir=output_dir, label_map=label_map)

            logger.info("Test (within):")
            test_within_key_metric = evaluate_ft(best_within_model, test_batches[1], loss_fn, k=K, task=task, split='test_within_easy', finetune_mode=finetune_mode, best_metrics=test_metrics, subgroup=False, verbose=True, device=device, logger=logger, wandb=wandb, save_scores=True, output_dir=output_dir, label_map=label_map)
        
        else:  # both random and hard drug-splits
            logger.info("Test (between):")
            test_key_metric = evaluate_ft(best_model, test_batches[0], loss_fn, k=K, task=task, split='test_between', finetune_mode=finetune_mode, best_metrics=test_metrics, subgroup=False, verbose=True, device=device, logger=logger, wandb=wandb, save_scores=True, output_dir=output_dir, label_map=label_map)

            logger.info("Test (within):")
            test_within_key_metric = evaluate_ft(best_within_model, test_batches[1], loss_fn, k=K, task=task, split='test_within', finetune_mode=finetune_mode, best_metrics=test_metrics, subgroup=False, verbose=True, device=device, logger=logger, wandb=wandb, save_scores=True, output_dir=output_dir, label_map=label_map)
            

def get_data_for_analysis(data_source: str, kg_encoder: str, split_method: str, repeat: str, path_base: str, checkpoint_dir: str):
    from madrigal.data.data import get_train_data

    checkpoint = checkpoint_dir + 'best_model.pt'
    class Args:
        def __init__(self):
            self.path_base = path_base
            self.checkpoint = checkpoint
            self.kg_encoder = kg_encoder
            self.split_method = split_method
            self.batch_size = None
            self.data_source = data_source
            self.repeat = repeat
            self.kg_sampling_num_neighbors = None
            self.kg_sampling_num_layers = None
            self.num_negative_samples_per_pair = None
            self.negative_sampling_probs_type = None
            self.num_workers = 2
        
    args = Args()
    _, train_loader, val_loaders, test_loaders, _, _, _, _, label_map = get_train_data(args, eval_mode=True)
    train_batch = next(iter(train_loader))
    val_batches = [next(iter(val_loader)) for val_loader in val_loaders]
    test_batches = [next(iter(test_loader)) for test_loader in test_loaders]

    return train_batch, val_batches, test_batches, label_map


# TODO: Merge this with the above function
def get_data_for_analysis_copy(data_source: str, kg_encoder: str, split_method: str, repeat: str, path_base: str, checkpoint: str, first_num_drugs: int, add_specific_drugs: str = None):
    from madrigal.data.data import get_all_drugs_data
    assert first_num_drugs is not None
    class Args:
        def __init__(self):
            self.path_base = path_base
            self.checkpoint = checkpoint
            self.kg_encoder = kg_encoder
            self.split_method = split_method
            self.batch_size = None
            self.data_source = data_source
            self.repeat = repeat
            self.kg_sampling_num_neighbors = None
            self.kg_sampling_num_layers = None
            self.num_negative_samples_per_pair = None
            self.negative_sampling_probs_type = None
            self.num_workers = 2
            self.first_num_drugs = first_num_drugs
        
    args = Args()
    _, test_loader, _, _, label_map = get_all_drugs_data(args, add_specific_drugs=add_specific_drugs)
    eval_batch = next(iter(test_loader))

    return None, None, eval_batch, label_map
    

def get_data_for_analysis_all_train(data_source: str, kg_encoder: str, split_method: str, repeat: str, path_base: str, ckpt: str):
    from madrigal.data.data import get_train_data_for_all_train

    assert split_method == "split_by_pairs"
    class Args:
        def __init__(self):
            self.path_base = path_base
            self.checkpoint = ckpt
            self.kg_encoder = kg_encoder
            self.split_method = split_method
            self.batch_size = None
            self.data_source = data_source
            self.repeat = repeat
            self.kg_sampling_num_neighbors = None
            self.kg_sampling_num_layers = None
            self.num_negative_samples_per_pair = None
            self.negative_sampling_probs_type = None
            self.num_workers = 2
        
    args = Args()
    _, train_loader, _, _, label_map = get_train_data_for_all_train(args, eval_mode=True)
    train_batch = next(iter(train_loader))

    return train_batch, label_map


@torch.no_grad()
def make_predictions(ckpt_or_checkpoint_dir: str, batch: dict, eval_type: str, finetune_mode: str, device: torch.device):
    from madrigal.models.models import NovelDDIEncoder, NovelDDIMultilabel
    from madrigal.utils import to_device
    from madrigal.evaluate.eval_utils import get_evaluate_masks

    if ckpt_or_checkpoint_dir.endswith(".pt"):
        checkpoint = torch.load(ckpt_or_checkpoint_dir, map_location=device)
    else:
        checkpoint = torch.load(ckpt_or_checkpoint_dir + "best_model.pt", map_location=device)
    encoder = NovelDDIEncoder(**checkpoint['encoder_configs'])
    model = NovelDDIMultilabel(encoder, **checkpoint['model_configs'])
    incomp_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
    incomp_keys_info = {
        "Missing keys": incomp_keys.missing_keys,
        "Unexpected_keys": incomp_keys.unexpected_keys,
    }
    print(incomp_keys_info)
    
    model.eval()
    model.to(device)

    batch_head = to_device(batch['head'], device)  # dict
    batch_tail = to_device(batch['tail'], device)
    batch_kg = to_device(batch['kg'], device)
    head_masks_base = batch['head']['masks']
    tail_masks_base = batch['tail']['masks']
    ddi_head_indices = batch['edge_indices']['head']
    ddi_tail_indices = batch['edge_indices']['tail']
    ddi_labels = batch['edge_indices']['label']
    
    masks_head, masks_tail = get_evaluate_masks(head_masks_base, tail_masks_base, eval_type, finetune_mode, device)
    pred_ddis = torch.sigmoid(model(batch_head, batch_tail, to_device(masks_head, device), to_device(masks_tail, device), batch_kg)).detach().cpu()
    pred_ddis = pred_ddis[ddi_labels, ddi_head_indices, ddi_tail_indices]  # in place to reduce GPU memory cost
    
    return pred_ddis


# TODO: Merge this with the above function
@torch.no_grad()
def make_predictions_copy(checkpoint_dir: str, batch: dict, eval_type: str, finetune_mode: str, device: torch.device):
    from madrigal.models.models import NovelDDIEncoder, NovelDDIMultilabel
    from madrigal.utils import to_device
    from madrigal.evaluate.eval_utils import get_evaluate_masks

    checkpoint = torch.load(checkpoint_dir + 'best_model.pt', map_location=device)
    encoder = NovelDDIEncoder(**checkpoint['encoder_configs'])
    model = NovelDDIMultilabel(encoder, **checkpoint['model_configs'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)

    batch_head = to_device(batch['head'], device)  # dict
    batch_tail = to_device(batch['tail'], device)
    batch_kg = to_device(batch['kg'], device)
    head_masks_base = batch['head']['masks']
    tail_masks_base = batch['tail']['masks']
    # ddi_head_indices = batch['edge_indices']['head']
    # ddi_tail_indices = batch['edge_indices']['tail']
    # ddi_labels = batch['edge_indices']['label']
    
    masks_head, masks_tail = get_evaluate_masks(head_masks_base, tail_masks_base, eval_type, finetune_mode, device)
    pred_ddis = torch.sigmoid(model(batch_head, batch_tail, to_device(masks_head, device), to_device(masks_tail, device), batch_kg)).detach().cpu()
    #pred_ddis = pred_ddis[ddi_labels, ddi_head_indices, ddi_tail_indices]  # in place to reduce GPU memory cost
    
    return pred_ddis


def get_drug_specific_scores(checkpoint_dir, batch, eval_type, finetune_mode, device, mode='test_between'):
    ddi_head_indices = batch['edge_indices']['head']
    ddi_tail_indices = batch['edge_indices']['tail']
    ddi_labels = batch['edge_indices']['label']
    ddi_pos_neg_samples = batch['edge_indices']['pos_neg'].float()
    true_ddis = ddi_pos_neg_samples
    
    pred_ddis = make_predictions(checkpoint_dir, batch, eval_type, finetune_mode, device)

    ddi_head_indices_pos = ddi_head_indices[true_ddis.bool()]
    ddi_tail_indices_pos = ddi_tail_indices[true_ddis.bool()]
    pred_ddis_pos = pred_ddis[true_ddis.bool()]
    ddi_labels_pos = ddi_labels[true_ddis.bool()]
    true_ddis_pos = true_ddis[true_ddis.bool()]

    ddi_head_indices_neg = ddi_head_indices[~true_ddis.bool()]
    ddi_tail_indices_neg = ddi_tail_indices[~true_ddis.bool()]
    pred_ddis_neg = pred_ddis[~true_ddis.bool()]
    ddi_labels_neg = ddi_labels[~true_ddis.bool()]
    true_ddis_neg = true_ddis[~true_ddis.bool()]

    if mode == 'test_between':
        drugs_of_interest = batch['head']['drugs'].cpu()  # between test test drugs
        drugs_of_interest_new_indices = np.arange(len(drugs_of_interest))
    elif mode == 'test_between_train':
        drugs_of_interest = batch['tail']['drugs'][np.unique(ddi_tail_indices_pos)].cpu()  # between test train drugs
        drugs_of_interest_new_indices = np.unique(ddi_tail_indices_pos)
    else:
        raise NotImplementedError

    metrics_for_all_drugs = []
    pos_samples_for_all_drugs = []
    for new_drug_index in drugs_of_interest_new_indices:
        # indices in the ddi (positive) list where this drug is involved
        if mode == 'test_between':
            drug_specific_ddi_indices_pos = torch.where(ddi_head_indices_pos == new_drug_index)[0]
        elif mode == 'test_between_train':
            drug_specific_ddi_indices_pos = torch.where(ddi_tail_indices_pos == new_drug_index)[0]

        # the scores & other stuff of this drug's positive ddis
        drug_specific_pred_ddis_pos = pred_ddis_pos[drug_specific_ddi_indices_pos]
        drug_specific_ddi_labels_pos = ddi_labels_pos[drug_specific_ddi_indices_pos]
        drug_specific_true_ddis_pos = true_ddis_pos[drug_specific_ddi_indices_pos]

        # indices in the ddi (negative) list where this drug is involved
        drug_specific_ddi_indices_neg = torch.cat([
            drug_specific_ddi_indices_pos,
            drug_specific_ddi_indices_pos + len(ddi_head_indices_pos)
        ])

        # the scores & other stuff of this drug's negative ddis
        drug_specific_pred_ddis_neg = pred_ddis_neg[drug_specific_ddi_indices_neg]
        drug_specific_ddi_labels_neg = ddi_labels_neg[drug_specific_ddi_indices_neg]
        drug_specific_true_ddis_neg = true_ddis_neg[drug_specific_ddi_indices_neg]

        drug_specific_pred_ddis = torch.cat([drug_specific_pred_ddis_pos, drug_specific_pred_ddis_neg])
        drug_specific_ddi_labels = torch.cat([drug_specific_ddi_labels_pos, drug_specific_ddi_labels_neg])
        drug_specific_true_ddis = torch.cat([drug_specific_true_ddis_pos, drug_specific_true_ddis_neg])

        metrics, pos_samples = get_metrics(
            drug_specific_pred_ddis.numpy(), 
            drug_specific_true_ddis.numpy(), 
            drug_specific_ddi_labels.numpy(), 
            k=50, 
            task='multiclass',
            average='macro',
            logger=None, 
            verbose=False,
        )

        metrics_for_all_drugs.append(metrics)
        pos_samples_for_all_drugs.append(pos_samples)

    all_metrics = {}
    for metrics in metrics_for_all_drugs:
        for metric_name, metric in metrics.items():
            all_metrics.setdefault(metric_name, []).append(metric)
            
    return all_metrics, drugs_of_interest


@torch.no_grad()
def get_all_preds(best_model, train_edgelist, val_edgelist, test_edgelist, masks, all_molecules, adjmat, batch_size, model_save_dir, split, wandb):
    """ Get predictions for all DDIs in the dataset. 
    Note that when batch_first=True (and activation in Transformer is either relu or gelu), the hook does not work.
    Also note that unlike in train.py, here `masks` contains the CLS mask.
    """
    activation = {}

    if split == 'ddi':
        train_adjmat = adjmat[0]
        val_adjmat = adjmat[1]
        test_adjmat = adjmat[2]
        val_batch = val_edgelist
        test_batch = test_edgelist
    elif split == 'drug':
        train_adjmat = adjmat[0]
        val_adjmat_between = adjmat[1]
        val_adjmat_within = adjmat[2]
        test_adjmat_between = adjmat[3]
        test_adjmat_within = adjmat[4]
        val_batch_between = val_edgelist[0]
        val_batch_within = val_edgelist[1]
        test_batch_between = test_edgelist[0]
        test_batch_within = test_edgelist[1]

    # TODO: Add an additional index for multimodal drugs (we don't record attentions for unimodal ones)
    best_model.encoder.transformer.transformer_encoder.layers[-1].self_attn.register_forward_hook(get_activation('last-layer-attention', activation))
    train_batch = train_edgelist
    
    # TODO: batching
    
    # Train predictions 
    _, train_pred_ddis_tri, train_true_ddis_tri, train_unique_drugs_in_batch, train_ddi_indices = evaluate_ft(best_model, train_batch, train_adjmat, masks, all_molecules, split='train', k=0.01, verbose=False)

    torch.save(train_pred_ddis_tri, model_save_dir+'/train_pred_ddi_scores.pt')
    torch.save(train_true_ddis_tri, model_save_dir+'/train_true_ddi_labels.pt')
    torch.save(train_ddi_indices, model_save_dir+'/train_ddi_indices.pt')
    torch.save(masks[train_unique_drugs_in_batch], model_save_dir+'/train_masks.pt')

    # Note that we only extract attention weights when test=True to avoid conflicts in batch_first.
    train_fin_embeds = activation['last-layer-attention'][0][0].detach()
    train_att_weights = activation['last-layer-attention'][0][1].detach()
    torch.save(train_att_weights, model_save_dir+'/train_last_layer_att_weights.pt')
    torch.save(train_fin_embeds, model_save_dir+'/train_final_embeds.pt')

    # Val predictions    
    if split == 'ddi':
        print("\nVal metrics:")
        val_metrics, val_pred_ddis_tri, val_true_ddis_tri, val_unique_drugs_in_batch, val_ddi_indices, val_subgroup_metrics = evaluate_ft(best_model, val_batch, val_adjmat, masks, split='val', k=0.01, subgroup=True, verbose=True)
        
        for subgroup, metrics in val_subgroup_metrics.items():
            val_fmax, val_recall_1_percent, val_precision_1_percent, val_ap_1_percent, val_auroc, val_auprc, val_accuracy, val_precision, val_recall, val_f1 = metrics
            wandb.log({f'val {subgroup} Fmax':val_fmax, f'val {subgroup} AUROC':val_auroc, f'val {subgroup} AUPRC':val_auprc})

        torch.save(val_pred_ddis_tri, model_save_dir+'/val_pred_ddi_scores.pt')
        torch.save(val_true_ddis_tri, model_save_dir+'/val_true_ddi_labels.pt')
        torch.save(val_ddi_indices, model_save_dir+'/val_ddi_indices.pt')
        torch.save(masks[val_unique_drugs_in_batch], model_save_dir+'/val_masks.pt')

        with open(model_save_dir+'/val_subgroup_results.json', 'w') as f:
            json.dump(val_subgroup_metrics, f)
        with open(model_save_dir+'/val_results.json', 'w') as f:
            json.dump(val_metrics, f)

        val_fin_embeds = activation['last-layer-attention'][1][0].detach()
        val_att_weights = activation['last-layer-attention'][1][1].detach()
        torch.save(val_att_weights, model_save_dir+'/val_last_layer_att_weights.pt')
        torch.save(val_fin_embeds, model_save_dir+'/val_final_embeds.pt')
    
    elif split == 'drug':
        print("\nVal metrics (between):")
        val_metrics, val_pred_ddis_tri, val_true_ddis_tri, val_unique_drugs_in_batch, val_ddi_indices, val_subgroup_metrics = evaluate_ft(best_model, val_batch_between, val_adjmat_between, masks, all_molecules, split='val', between_within='between', k=0.01, subgroup=True, verbose=True)

        for subgroup, metrics in val_subgroup_metrics.items():
            val_fmax, val_recall_1_percent, val_precision_1_percent, val_ap_1_percent, val_auroc, val_auprc, val_accuracy, val_precision, val_recall, val_f1 = metrics
            wandb.log({f'val {subgroup} Fmax between':val_fmax, f'val {subgroup} AUROC between':val_auroc, f'val {subgroup} AUPRC between':val_auprc})
        
        torch.save(val_pred_ddis_tri, model_save_dir+'/val_pred_ddi_scores_between.pt')
        torch.save(val_true_ddis_tri, model_save_dir+'/val_true_ddi_labels_between.pt')
        torch.save(val_ddi_indices, model_save_dir+'/val_ddi_indices_between.pt')
        torch.save(masks[val_unique_drugs_in_batch], model_save_dir+'/val_masks_between.pt')

        with open(model_save_dir+'/val_subgroup_results_between.json', 'w') as f:
            json.dump(val_subgroup_metrics, f)
        with open(model_save_dir+'/val_results_between.json', 'w') as f:
            json.dump(val_metrics, f)
        
        print("\nVal metrics (within):")
        val_metrics, val_pred_ddis_tri, val_true_ddis_tri, val_unique_drugs_in_batch, val_ddi_indices, val_subgroup_metrics = evaluate_ft(best_model, val_batch_within, val_adjmat_within, masks, all_molecules, split='val', between_within='within', k=0.01, subgroup=True, verbose=True)

        for subgroup, metrics in val_subgroup_metrics.items():
            val_fmax, val_recall_1_percent, val_precision_1_percent, val_ap_1_percent, val_auroc, val_auprc, val_accuracy, val_precision, val_recall, val_f1 = metrics
            wandb.log({f'val {subgroup} Fmax within':val_fmax, f'val {subgroup} AUROC within':val_auroc, f'val {subgroup} AUPRC within':val_auprc})

        torch.save(val_pred_ddis_tri, model_save_dir+'/val_pred_ddi_scores_within.pt')
        torch.save(val_true_ddis_tri, model_save_dir+'/val_true_ddi_labels_within.pt')
        torch.save(val_ddi_indices, model_save_dir+'/val_ddi_indices_within.pt')
        torch.save(masks[val_unique_drugs_in_batch], model_save_dir+'/val_masks_within.pt')

        with open(model_save_dir+'/val_subgroup_results_within.json', 'w') as f:
            json.dump(val_subgroup_metrics, f)
        with open(model_save_dir+'/val_results_within.json', 'w') as f:
            json.dump(val_metrics, f)

        val_fin_embeds_between = activation['last-layer-attention'][1][0].detach()
        val_att_weights_between = activation['last-layer-attention'][1][1].detach()
        val_fin_embeds_within = activation['last-layer-attention'][2][0].detach()
        val_att_weights_within = activation['last-layer-attention'][2][1].detach()
        torch.save(val_att_weights_between, model_save_dir+'/val_last_layer_att_weights_between.pt')
        torch.save(val_fin_embeds_between, model_save_dir+'/val_final_embeds_between.pt')
        torch.save(val_att_weights_within, model_save_dir+'/val_last_layer_att_weights_within.pt')
        torch.save(val_fin_embeds_within, model_save_dir+'/val_final_embeds_within.pt')
    
    # Test predictions
    if split == 'ddi':
        print("\nTest metrics:")
        test_metrics, test_pred_ddis_tri, test_true_ddis_tri, test_unique_drugs_in_batch, test_ddi_indices, test_subgroup_metrics = evaluate_ft(best_model, test_batch, test_adjmat, masks, all_molecules, split='test', k=0.01, subgroup=True, verbose=True)
        
        test_fmax, test_recall_1_percent, test_precision_1_percent, test_ap_1_percent, test_auroc, test_auprc, test_accuracy, test_precision, test_recall, test_f1 = test_metrics
        wandb.log({'test_fmax':test_fmax, 'test_auprc':test_auprc, 'test_recall_1_percent':test_recall_1_percent, 'test_precision_1_percent':test_precision_1_percent, 'test_ap_1_percent':test_ap_1_percent, 'test_auroc':test_auroc})

        for subgroup, metrics in test_subgroup_metrics.items():
            test_fmax, test_recall_1_percent, test_precision_1_percent, test_ap_1_percent, test_auroc, test_auprc, test_accuracy, test_precision, test_recall, test_f1 = metrics
            wandb.log({f'test {subgroup} Fmax':test_fmax, f'test {subgroup} AUROC':test_auroc, f'test {subgroup} AUPRC':test_auprc})

        torch.save(test_pred_ddis_tri, model_save_dir+'/test_pred_ddi_scores.pt')
        torch.save(test_true_ddis_tri, model_save_dir+'/test_true_ddi_labels.pt')
        torch.save(test_ddi_indices, model_save_dir+'/test_ddi_indices.pt')
        torch.save(masks[val_unique_drugs_in_batch], model_save_dir+'/test_masks.pt')
        
        with open(model_save_dir+'/test_subgroup_results.json', 'w') as f:
            json.dump(test_subgroup_metrics, f)
        with open(model_save_dir+'/test_results.json', 'w') as f:
            json.dump(test_metrics, f)

        test_fin_embeds = activation['last-layer-attention'][2][0].detach()
        test_att_weights = activation['last-layer-attention'][2][1].detach()
        torch.save(test_att_weights, model_save_dir+'/test_last_layer_att_weights.pt')
        torch.save(test_fin_embeds, model_save_dir+'/test_final_embeds.pt')

    elif split == 'drug':        
        print("\nTest metrics (between):")
        test_metrics, test_pred_ddis_tri, test_true_ddis_tri, test_unique_drugs_in_batch, test_ddi_indices, test_subgroup_metrics = evaluate_ft(best_model, test_batch_between, test_adjmat_between, masks, all_molecules, split='test', between_within='between', k=0.01, subgroup=True, verbose=True)

        for subgroup, metrics in test_subgroup_metrics.items():
            test_fmax, test_recall_1_percent, test_precision_1_percent, test_ap_1_percent, test_auroc, test_auprc, test_accuracy, test_precision, test_recall, test_f1 = metrics
            wandb.log({f'test {subgroup} Fmax between':test_fmax, f'test {subgroup} AUROC between':test_auroc, f'val {subgroup} AUPRC between':test_auprc})
        
        torch.save(test_pred_ddis_tri, model_save_dir+'/test_pred_ddi_scores_between.pt')
        torch.save(test_true_ddis_tri, model_save_dir+'/test_true_ddi_labels_between.pt')
        torch.save(test_ddi_indices, model_save_dir+'/test_ddi_indices_between.pt')
        torch.save(masks[test_unique_drugs_in_batch], model_save_dir+'/test_masks_between.pt')

        with open(model_save_dir+'/test_subgroup_results_between.json', 'w') as f:
            json.dump(test_subgroup_metrics, f)
        with open(model_save_dir+'/test_results_between.json', 'w') as f:
            json.dump(test_metrics, f)
        
        print("\nTest metrics (within):")
        test_metrics, test_pred_ddis_tri, test_true_ddis_tri, test_unique_drugs_in_batch, test_ddi_indices, test_subgroup_metrics = evaluate_ft(best_model, test_batch_within, test_adjmat_within, masks, all_molecules, split='test', between_within='within', k=0.01, subgroup=True, verbose=True)

        for subgroup, metrics in test_subgroup_metrics.items():
            test_fmax, test_recall_1_percent, test_precision_1_percent, test_ap_1_percent, test_auroc, test_auprc, test_accuracy, test_precision, test_recall, test_f1 = metrics
            wandb.log({f'test {subgroup} Fmax within':test_fmax, f'test {subgroup} AUROC within':test_auroc, f'val {subgroup} AUPRC within':test_auprc})

        torch.save(test_pred_ddis_tri, model_save_dir+'/test_pred_ddi_scores_within.pt')
        torch.save(test_true_ddis_tri, model_save_dir+'/test_true_ddi_labels_within.pt')
        torch.save(test_ddi_indices, model_save_dir+'/test_ddi_indices_within.pt')
        torch.save(masks[test_unique_drugs_in_batch], model_save_dir+'/test_masks_within.pt')

        with open(model_save_dir+'/test_subgroup_results_within.json', 'w') as f:
            json.dump(test_subgroup_metrics, f)
        with open(model_save_dir+'/test_results_within.json', 'w') as f:
            json.dump(test_metrics, f)

        test_fin_embeds_between = activation['last-layer-attention'][3][0].detach()
        test_att_weights_between = activation['last-layer-attention'][3][1].detach()
        test_fin_embeds_within = activation['last-layer-attention'][4][0].detach()
        test_att_weights_within = activation['last-layer-attention'][4][1].detach()
        torch.save(test_att_weights_between, model_save_dir+'/test_last_layer_att_weights_between.pt')
        torch.save(test_fin_embeds_between, model_save_dir+'/test_final_embeds_between.pt')
        torch.save(test_att_weights_within, model_save_dir+'/test_last_layer_att_weights_within.pt')
        torch.save(test_fin_embeds_within, model_save_dir+'/test_final_embeds_within.pt')
