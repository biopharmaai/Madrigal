import os
import gc
from datetime import datetime

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

import torch
import wandb
import torch.nn as nn
# import torch.multiprocessing as mp
# from torch.distributed import init_process_group, get_rank, get_world_size

## importing files
from madrigal.evaluate.metrics import get_metrics
from madrigal.evaluate.eval_utils import K, AVERAGE, FINETUNE_MODE_ABLATION_FULL_UNAVAIL_MAP
from madrigal.parse_args import create_parser, get_hparams
from madrigal.data.data import get_train_data_for_all_train
from madrigal.utils import (
    NON_TX_MODALITIES,
    get_model,
    get_loss_fn,
    create_optimizer,
    to_device,
    from_indices_to_tensor,
    powerset,
    get_root_logger,
    get_str_encoder_hparams,
    get_kg_encoder_hparams,
    get_cv_encoder_hparams,
    get_tx_encoder_hparams,
    get_transformer_fusion_hparams,
    get_proj_hparams,
    LinearWarmupCosineDecaySchedule,
    set_seed,
)

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(train_loader, task, all_kg_data, num_labels, num_epochs, loss_fn_name, feature_dim, str_encoder, str_encoder_hparams, str_node_feat_dim, kg_encoder, kg_encoder_hparams, cv_encoder, cv_encoder_hparams, tx_encoder, tx_encoder_hparams, transformer_fusion_hparams, proj_hparams, hparams, save_dir, finetune_mode, device, logger, frozen=False, tab_mod_encoder_hparams_dict=None):
    """ Main training function
    """
    model, encoder_configs, model_configs = get_model(
        all_kg_data, 
        feature_dim,
        num_labels,
        str_encoder, 
        str_encoder_hparams, 
        kg_encoder, 
        kg_encoder_hparams, 
        cv_encoder,
        cv_encoder_hparams,
        tx_encoder,
        tx_encoder_hparams,
        hparams["num_attention_bottlenecks"],
        hparams["pos_emb_type"],
        hparams["pos_emb_dropout"],
        transformer_fusion_hparams,
        proj_hparams, 
        hparams["fusion"],
        hparams["normalize"],
        hparams["decoder_normalize"],
        hparams["checkpoint"],
        frozen,
        device,
        encoder_only=False,
        finetune_mode=finetune_mode,
        str_node_feat_dim=str_node_feat_dim,
        logger=logger,
        use_modality_pretrain=hparams["use_modality_pretrain"],
        adapt_before_fusion=hparams["adapt_before_fusion"],
        use_pretrained_adaptor=hparams["use_pretrained_adaptor"],
        tab_mod_encoder_hparams_dict=tab_mod_encoder_hparams_dict,
    )
    
    if hparams["checkpoint"] is not None:
        encoder_hparams = wandb.config
        for k, v in encoder_configs.items():  # NOTE: Replace wandb displayed hyperparameters with the ones actually used from the checkpoint (exclusion of those not used are already done in `get_model`)
            if k in encoder_hparams.keys():
                encoder_hparams[k] = v
            elif "encoder_name" in k:  # In hparams (wandb.config), it is "*_encoder", while in encoder_configs, it is "*_encoder_name"
                encoder_hparams[k[:-5]] = v
            elif "hparams" in k:  # 
                for kk, vv in v.items():
                    if kk in encoder_hparams.keys():
                        encoder_hparams[kk] = vv
            elif k == "feat_dim":
                encoder_hparams["feature_dim"] = v
            elif k == "num_tx_bottlenecks":
                encoder_hparams["num_attention_bottlenecks"] = v
        wandb.config.update(encoder_hparams)
    
    loss_fn = get_loss_fn(loss_fn_name, task, hparams["loss_readout"])
    optimizer = create_optimizer(model, hparams)
    
    if hparams["warmup_epochs"] > 0:
        scheduler = LinearWarmupCosineDecaySchedule(optimizer, warmup_epochs=hparams["warmup_epochs"], total_epochs=num_epochs, num_cycles=1.)
    else:
        scheduler = None

    # count parameters of the model
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    wandb.log({"num_all_params":sum(p.numel() for p in model.parameters())})
    wandb.log({"num_trainable_params":sum(p.numel() for p in model.parameters() if p.requires_grad)})
    
    train_batch = next(iter(train_loader))
    
    batch_head = train_batch["head"]  # dict
    batch_tail = train_batch["tail"]
    batch_kg = train_batch["kg"]
    head_masks_base = train_batch["head"]["masks"]  # to device later
    tail_masks_base = train_batch["tail"]["masks"]
    ddi_head_indices = train_batch["edge_indices"]["head"]
    ddi_tail_indices = train_batch["edge_indices"]["tail"]
    ddi_labels = train_batch["edge_indices"]["label"]
    
    if isinstance(loss_fn, (nn.BCEWithLogitsLoss, nn.BCELoss)):
        ddi_pos_neg_samples = train_batch["edge_indices"]["pos_neg"].float()
    elif isinstance(loss_fn, nn.CrossEntropyLoss):
        ddi_pos_neg_samples = train_batch["edge_indices"]["pos_neg"].long()
    else:
        raise NotImplementedError
    
    if finetune_mode == "full_full":
        # get masks
        head_masks_base = head_masks_base
        tail_masks_base = tail_masks_base
        
        # make ddi directed
        directed_indices_bool = ddi_head_indices < ddi_tail_indices
        ddi_head_indices = ddi_head_indices[directed_indices_bool]
        ddi_tail_indices = ddi_tail_indices[directed_indices_bool]
        ddi_labels = ddi_labels[directed_indices_bool]
        ddi_pos_neg_samples = ddi_pos_neg_samples[directed_indices_bool]
    
    elif finetune_mode == "ablation_str_str" or "padded" in finetune_mode:  # all "ablation_x_x_padded" runs
        # get masks
        head_masks_base = torch.zeros_like(head_masks_base)
        unavail_mod_indices = FINETUNE_MODE_ABLATION_FULL_UNAVAIL_MAP[finetune_mode]
        head_masks_base[:, unavail_mod_indices] = 1
        head_masks_base = head_masks_base.bool()
        tail_masks_base = head_masks_base
        
        # make ddi directed
        directed_indices_bool = ddi_head_indices < ddi_tail_indices
        ddi_head_indices = ddi_head_indices[directed_indices_bool]
        ddi_tail_indices = ddi_tail_indices[directed_indices_bool]
        ddi_labels = ddi_labels[directed_indices_bool]
        ddi_pos_neg_samples = ddi_pos_neg_samples[directed_indices_bool]
    
    # here we want to avoid reindexing the valid indices
    elif finetune_mode == "ablation_kg_kg_subset":
        # remove the head/tail ddi indices that has no kg modality
        head_valid_indices = torch.where(head_masks_base[:, 1]==0)[0]  # subset of heads that have kg modality
        tail_valid_indices = torch.where(tail_masks_base[:, 1]==0)[0]  # subset of tails that have kg modality
        
        valid_indices_bool = (torch.isin(ddi_head_indices, head_valid_indices) & torch.isin(ddi_tail_indices, tail_valid_indices))  # for the edge list, select only edges where heads and tails are both valid.
        ddi_head_indices = ddi_head_indices[valid_indices_bool]
        ddi_tail_indices = ddi_tail_indices[valid_indices_bool]
        ddi_labels = ddi_labels[valid_indices_bool]
        ddi_pos_neg_samples = ddi_pos_neg_samples[valid_indices_bool]
        
        # get masks 
        head_masks_base = torch.ones_like(head_masks_base)
        head_masks_base[:, 1] = 0
        head_masks_base = head_masks_base.bool()
        tail_masks_base = head_masks_base
        
        # make ddi directed
        directed_indices_bool = ddi_head_indices < ddi_tail_indices
        ddi_head_indices = ddi_head_indices[directed_indices_bool]
        ddi_tail_indices = ddi_tail_indices[directed_indices_bool]
        ddi_labels = ddi_labels[directed_indices_bool]
        ddi_pos_neg_samples = ddi_pos_neg_samples[directed_indices_bool]
    
    elif finetune_mode == "str_full":  # NOTE: will be used as str-str (directed) + str-full (undirected) + full-full (directed)
        head_masks_base = torch.ones_like(head_masks_base)
        head_masks_base[:, 0] = 0
        head_masks_base = head_masks_base.bool()
        tail_masks_base = tail_masks_base
        
    # NOTE: For efficiency in full batch training, we use the fact that head and tail drugs are the same.
    elif finetune_mode == "str_str+random_sample":  # NOTE: will be used as str-str (directed) + str-str+random (undirected) + str+random-str+random (directed)
        head_all_subset_masks = [torch.stack([from_indices_to_tensor(list(indices), head_masks_base.shape[1]) for indices in list(powerset(torch.where(mask==0)[0].tolist()))[1:] if 0 in indices]) for mask in head_masks_base.int()]  # generate only subset masks that contain structure modality
    
    elif finetune_mode in {"str_random_sample", "double_random"}:  # NOTE: "str_random_sample" will be used as str-str (directed) + str-random (undirected) + random-random (directed); while "double_random" will be random-random (undirected)
        head_all_subset_masks = [torch.stack([from_indices_to_tensor(list(indices), head_masks_base.shape[1]) for indices in list(powerset(torch.where(mask==0)[0].tolist()))[1:]]) for mask in head_masks_base.int()]
        
    elif finetune_mode in {
        "ablation_str_random_str+kg_full_sample", 
        "ablation_str_random_str+cv_full_sample", 
        "ablation_str_random_str+tx_full_sample",
        "ablation_str_random_str+kg+cv_full_sample",
        "ablation_str_random_str+kg+tx_full_sample",
        "ablation_str_random_str+cv+tx_full_sample",
    }:
        unavail_mod_indices = FINETUNE_MODE_ABLATION_FULL_UNAVAIL_MAP[finetune_mode]
        head_masks_base[:, unavail_mod_indices] = True
        head_all_subset_masks = [torch.stack([from_indices_to_tensor(list(indices), head_masks_base.shape[1]) for indices in list(powerset(torch.where(mask==0)[0].tolist()))[1:]]) for mask in head_masks_base.int()]
        
    else:
        raise NotImplementedError

    assert len(head_masks_base) == len(tail_masks_base)

    wandb.watch(model, log="all", log_freq=200)
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        # random sampling cases need additional data processing in each epoch/batch
        if finetune_mode == "double_random":
            # NOTE: make sure the two sampled modality masks are different whenever possible
            masks_random_head, masks_random_tail = torch.stack([subset_masks[torch.randperm(len(subset_masks))[:2] if len(subset_masks)>1 else torch.tensor([0, 0])] for subset_masks in head_all_subset_masks], dim=0).bool().unbind(1)  # Some drugs only have one subset mask (structure). 
            
        elif finetune_mode in {
            "str_str+random_sample", 
            "str_random_sample", 
            "ablation_str_random_str+kg_full_sample", 
            "ablation_str_random_str+cv_full_sample", 
            "ablation_str_random_str+tx_full_sample",
            "ablation_str_random_str+kg+cv_full_sample",
            "ablation_str_random_str+kg+tx_full_sample",
            "ablation_str_random_str+cv+tx_full_sample"
        }:
            masks_str = torch.ones_like(head_masks_base)
            masks_str[:, 0] = 0
            masks_str = masks_str.bool()
            masks_X = torch.stack([subset_masks[torch.randperm(len(subset_masks)-1)[0] + 1] if len(subset_masks)>1 else subset_masks[0] for subset_masks in head_all_subset_masks], dim=0).bool()  # NOTE: We don"t want to retrieve the structure-only mask, so we used the +1 offset ([0, 1, 1, ...] is always the first among all subset masks). Still need to consider adding dummy 0"s for those str_only drugs.
            
        elif finetune_mode in {"full_full", "ablation_str_str", "ablation_kg_kg_subset"} or "padded" in finetune_mode:
            masks_X = head_masks_base
        
        # Start training
        model.train()
        batch_head = to_device(batch_head, device)
        batch_tail = to_device(batch_tail, device)
        batch_kg = to_device(batch_kg, device)
        ddi_labels = to_device(ddi_labels, device)
        ddi_head_indices = to_device(ddi_head_indices, device)
        ddi_tail_indices = to_device(ddi_tail_indices, device)
        ddi_pos_neg_samples = to_device(ddi_pos_neg_samples, device)
        
        optimizer.zero_grad()
        
        if finetune_mode in {
            "full_full", 
            "ablation_str_str", 
            "ablation_kg_kg_subset", 
        } or "padded" in finetune_mode:
            pred_ddis = torch.sigmoid(model(batch_head, batch_tail, to_device(masks_X, device), to_device(masks_X, device), batch_kg))
            pred_ddis = pred_ddis[ddi_labels, ddi_head_indices, ddi_tail_indices]  # NOTE: ddi indices are already made directed for these cases; in place to reduce GPU memory cost
            true_ddis = ddi_pos_neg_samples
            loss = loss_fn(pred_ddis, true_ddis)
            
            loss.backward()
            logger.info(f"Train {epoch+1}: loss = {loss.item()}")
            wandb.log({"train_loss": loss.item()}, step=epoch)
        
        elif finetune_mode == "double_random":  # effectively same code as above, but separated for clarity
            pred_ddis = torch.sigmoid(model(batch_head, batch_tail, to_device(masks_random_head, device), to_device(masks_random_tail, device), batch_kg))
            pred_ddis = pred_ddis[ddi_labels, ddi_head_indices, ddi_tail_indices]  # in place to reduce GPU memory cost
            true_ddis = ddi_pos_neg_samples
            loss = loss_fn(pred_ddis, true_ddis)

            loss.backward()
            logger.info(f"Train {epoch+1}: loss = {loss.item()}")
            wandb.log({"train_loss": loss.item()}, step=epoch)
        
        elif finetune_mode in {
            "str_str+random_sample", 
            "str_random_sample", 
            "str_full", 
            "ablation_str_random_str+kg_full_sample", 
            "ablation_str_random_str+cv_full_sample", 
            "ablation_str_random_str+tx_full_sample",
            "ablation_str_random_str+kg+cv_full_sample",
            "ablation_str_random_str+kg+tx_full_sample",
            "ablation_str_random_str+cv+tx_full_sample",
        }:
            directed_indices_bool = ddi_head_indices < ddi_tail_indices
            ddi_head_indices_directed = ddi_head_indices[directed_indices_bool]
            ddi_tail_indices_directed = ddi_tail_indices[directed_indices_bool]
            ddi_labels_directed = ddi_labels[directed_indices_bool]
            ddi_pos_neg_samples_directed = ddi_pos_neg_samples[directed_indices_bool]
            
            # str-str (directed)
            if hparams["train_with_str_str"]:
                pred_ddis = torch.sigmoid(model(batch_head, batch_tail, to_device(masks_str, device), to_device(masks_str, device), batch_kg))
                pred_ddis = pred_ddis[ddi_labels_directed, ddi_head_indices_directed, ddi_tail_indices_directed]  # in place to reduce GPU memory cost
                true_ddis = ddi_pos_neg_samples_directed
                loss_str_str = loss_fn(pred_ddis, true_ddis)
                loss_str_str.backward()
            else:
                loss_str_str = torch.zeros(1).to(device)
            
            # X-X (directed)
            pred_ddis = torch.sigmoid(model(batch_head, batch_tail, to_device(masks_X, device), to_device(masks_X, device), batch_kg))
            pred_ddis = pred_ddis[ddi_labels_directed, ddi_head_indices_directed, ddi_tail_indices_directed]  # in place to reduce GPU memory cost
            true_ddis = ddi_pos_neg_samples_directed
            loss_X_X = loss_fn(pred_ddis, true_ddis)
            
            loss_X_X.backward()
            
            # str-X (undirected)
            pred_ddis = torch.sigmoid(model(batch_head, batch_tail, to_device(masks_str, device), to_device(masks_X, device), batch_kg))
            pred_ddis = pred_ddis[ddi_labels, ddi_head_indices, ddi_tail_indices]  # in place to reduce GPU memory cost
            true_ddis = ddi_pos_neg_samples
            loss_str_X = loss_fn(pred_ddis, true_ddis)
            
            loss_str_X.backward()
            
            loss = (loss_str_str + loss_str_X + loss_X_X).item()
            logger.info(f"Train {epoch+1}: loss = {loss}, loss_str_str = {loss_str_str.item()}, loss_str_X = {loss_str_X.item()}, loss_X_X = {loss_X_X.item()}")
            wandb.log({"train_loss": loss, "train_loss_str_str": loss_str_str.item(), "train_loss_str_X": loss_str_X.item(), "train_loss_X_X": loss_X_X.item()}, step=epoch)
        
        optimizer.step()
        wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]}, step=epoch)
        if scheduler is not None:
            scheduler.step()

        if epoch % hparams["evaluate_interval"] == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
            model.eval()
            logger.info("Computing train metrics:")
            
            train_metrics, _ = get_metrics(
                pred_ddis.detach().cpu().numpy(), 
                true_ddis.detach().cpu().numpy(), 
                ddi_labels.detach().cpu().numpy(), 
                k=K, 
                task=task,
                logger=logger, 
                average=AVERAGE,
                verbose=True,
            )  # NOTE: For the str_X cases, this is only calculating the batch metrics for the str_X case, not the str_str or X_X cases
            wandb.log({f"train_batch_{metric_name}": metric_value for metric_name, metric_value in train_metrics.items()}, step=epoch)
        
        if (epoch+1) % 100 == 0:
            torch.save(
                {
                    "state_dict":model.state_dict(), 
                    "encoder_configs":encoder_configs,
                    "model_configs":model_configs,
                }, 
                save_dir+f"checkpoint_{epoch+1}.pt")


def main():
    args = create_parser("train")
    hparams = get_hparams(args, "train")
    
    if args.seed is not None:
        seed = args.seed
    else:
        seed = SEED
    set_seed(seed)

    project_name = f"{args.data_source}_all_train"
    
    wandb.init(
        project=project_name, 
        entity="noveldrugdrug",
        dir=args.save_dir,
        mode="offline" if args.debug else "online",
        config=hparams, 
    )
    wandb.run.name = args.run_name if args.run_name is not None else wandb.run.name
    cur_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
    output_dir = f"{args.save_dir}/{cur_time}_{wandb.run.name}/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = get_root_logger(output_dir+"log.txt")
    
    logger.info("Args: {}".format(args))
    logger.info("hparams: {}".format(hparams))
    logger.info("wandb: {}".format(wandb.run.name))
    logger.info("log_dir_path: {}".format(output_dir))
    
    logger.info("Loading data...")
    all_kg_data, train_loader, train_collator, train_dataset, label_map = get_train_data_for_all_train(args, logger)
    logger.info("Training positive samples: {}".format(len(train_dataset)))

    ## Train model
    logger.info("Training starting...")
    
    # Collate hidden dims for structural encoder. Same should be done for Cv/Ts MLPs, maybe wrap in a function
    str_encoder_hparams = get_str_encoder_hparams(args, hparams)
    kg_encoder_hparams = get_kg_encoder_hparams(args, hparams)  # hparams["han_att_heads"], hparams["han_hidden_dim"]
    cv_encoder_hparams = get_cv_encoder_hparams(args, hparams, train_collator.tabular_mod_dfs["cv"].shape[0])
    tab_mod_encoder_hparams_dict = {}
    for mod in NON_TX_MODALITIES[2:]:
        tab_mod_encoder_hparams_dict[mod] = get_cv_encoder_hparams(args, hparams, train_collator.tabular_mod_dfs[mod].shape[0])
    tx_encoder_hparams = get_tx_encoder_hparams(args, hparams, train_collator.tx_df.shape[0])
    proj_hparams = get_proj_hparams(hparams)
    transformer_fusion_hparams = get_transformer_fusion_hparams(args, hparams)

    train(
        train_loader, 
        args.task, 
        all_kg_data, 
        train_dataset.num_labels, 
        args.num_epochs, 
        args.loss_fn_name, 
        args.feature_dim, 
        args.str_encoder, 
        str_encoder_hparams, 
        train_collator.str_node_feat_dim, 
        args.kg_encoder, 
        kg_encoder_hparams, 
        args.cv_encoder, 
        cv_encoder_hparams, 
        args.tx_encoder, 
        tx_encoder_hparams, 
        transformer_fusion_hparams, 
        proj_hparams, 
        hparams, 
        output_dir, 
        args.finetune_mode, 
        device, 
        logger, 
        args.frozen, 
        tab_mod_encoder_hparams_dict,
    )
        
        
if __name__ == "__main__":
    main()
