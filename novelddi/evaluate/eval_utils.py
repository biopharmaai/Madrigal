import torch
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, cdist, squareform

from novelddi.utils import CELL_LINES, NUM_MODALITIES

# NOTE: treat auroc as gold standard for multilabel classification, and f1 as gold standard for multiclass classification
KEY_METRIC_DICT = {'multilabel':'auprc', 'multiclass':'auprc'}
AVERAGE = 'macro'
NUMBER2MODALITY = {
    '0':'str', 
    '1':'kg', 
    '2':'cv', 
    '01':'str+kg', 
    '02':'str+cv', 
    '12':'kg+cv', 
    '012':'str+kg+cv'
}
NUMBER2MODALITY.update({
    str(i+3) : f'tx_{cell_line}'
    for i, cell_line in enumerate(CELL_LINES)
})
MODALITY2NUMBER_LIST = {
    'str':[0],
    'kg':[1],
    'cv':[2],
}
MODALITY2NUMBER_LIST.update({
    f'tx_{cell_line}' : [i+3]
    for i, cell_line in enumerate(CELL_LINES)
})
MODALITY2NUMBER_LIST.update({
    'tx' : [i+3 for i in range(len(CELL_LINES))]
})
FINETUNE_MODE_MODEL_SELECTION_EVAL_TYPE_BETWEEN_MAP = {
    'ablation_str_str':'str_str', 
    'ablation_kg_kg_subset':'kg_kg', 
    'ablation_kg_kg_padded':'kg_kg', 
    'ablation_cv_cv_padded': 'cv_cv',
    'ablation_tx_tx_padded': 'tx_tx',
    'ablation_str_random_str+kg_full_sample':'str_full', 
    'ablation_str_random_str+cv_full_sample':'str_full',
    'ablation_str_random_str+tx_full_sample':'str+tx_full',
    'ablation_str_random_str+kg+cv_full_sample':'str_full', 
    'ablation_str_random_str+kg+tx_full_sample':'str+tx_full',
    'ablation_str_random_str+cv+tx_full_sample':'str+tx_full',
    'str_full':'str_full', 
    'full_full':'str+tx_full', 
    'double_random':'str+tx_full', 
    'str_random_sample':'str+tx_full', 
    'str_str+random_sample':'str+tx_full', 
    'full_str+random_sample':'str+tx_full',
}
FINETUNE_MODE_MODEL_SELECTION_EVAL_TYPE_WITHIN_MAP = {
    'ablation_str_str':'str_str', 
    'ablation_kg_kg_subset':'kg_kg', 
    'ablation_kg_kg_padded':'kg_kg', 
    'ablation_cv_cv_padded': 'cv_cv',
    'ablation_tx_tx_padded': 'tx_tx',
    'ablation_str_random_str+kg_full_sample':'full_full', 
    'ablation_str_random_str+cv_full_sample':'full_full',
    'ablation_str_random_str+tx_full_sample':'full_full',
    'ablation_str_random_str+kg+cv_full_sample':'full_full', 
    'ablation_str_random_str+kg+tx_full_sample':'full_full',
    'ablation_str_random_str+cv+tx_full_sample':'full_full',
    'str_full':'str_str', 
    'full_full':'str_str', 
    'double_random':'str_str', 
    'str_random_sample':'str_str', 
    'str_str+random_sample':'str_str', 
    'full_str+random_sample':'str_str',
}
FINETUNE_MODE_MODEL_SELECTION_EVAL_TYPE_MAP = {
    'ablation_str_str':'str_str', 
    'ablation_kg_kg_subset':'kg_kg',
    'ablation_kg_kg_padded':'kg_kg', 
    'ablaiton_cv_cv_padded': 'cv_cv',
    'ablation_tx_tx_padded': 'tx_tx',
    'ablation_str_random_str+kg_full_sample':'full_full', 
    'ablation_str_random_str+cv_full_sample':'full_full',
    'ablation_str_random_str+tx_full_sample':'full_full',
    'ablation_str_random_str+kg+cv_full_sample':'full_full', 
    'ablation_str_random_str+kg+tx_full_sample':'full_full',
    'ablation_str_random_str+cv+tx_full_sample':'full_full',
    'str_full':'full_full', 
    'full_full':'full_full', 
    'double_random':'full_full', 
    'str_random_sample':'full_full', 
    'str_str+random_sample':'full_full', 
    'full_str+random_sample':'full_full',
}
FINETUNE_MODE_ABLATION_FULL_UNAVAIL_MAP = {
    'ablation_str_str': [1,2] + [i+3 for i in range(len(CELL_LINES))], 
    'ablation_kg_kg_subset': [0,2] + [i+3 for i in range(len(CELL_LINES))], 
    'ablation_kg_kg_padded': [0,2] + [i+3 for i in range(len(CELL_LINES))], 
    'ablation_cv_cv_padded': [0,1] + [i+3 for i in range(len(CELL_LINES))],
    'ablation_tx_tx_padded': [0,1,2],
    'ablation_str_random_str+kg_full_sample': [2] + [i+3 for i in range(len(CELL_LINES))], 
    'ablation_str_random_str+cv_full_sample': [1] + [i+3 for i in range(len(CELL_LINES))],
    'ablation_str_random_str+tx_full_sample': [1,2],
    'ablation_str_random_str+kg+cv_full_sample': [i+3 for i in range(len(CELL_LINES))], 
    'ablation_str_random_str+kg+tx_full_sample': [2],
    'ablation_str_random_str+cv+tx_full_sample': [1],
}
K = 50


def uniform_loss(x, t=2):
    x = x / torch.norm(x, dim=1, keepdim=True)  # NOTE: normalize before uniformity
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def alignment_loss(x1, x2, alpha=2):
    x1 = x1 / torch.norm(x1, dim=1, keepdim=True)
    x2 = x2 / torch.norm(x2, dim=1, keepdim=True)
    return (x1 - x2).norm(p=2, dim=1).pow(alpha).mean()


def stacked_inst_dist_topk_accuracy(output, target, topk=(1, 5, 20)):
    """ Computes the accuracy over the k top predictions for the specified values of k, among representations from both modalities
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(torch.where(target==1)[1].expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k / batch_size)
    
    return res


def knn_classifier(train_features, train_labels, test_features, test_labels, metric='cosine', k=5, T=1, num_classes=2):
    """ From DINO: https://github.com/facebookresearch/dino/blob/main/eval_knn.py
    To account for the multilabel nature of our data (drug ATC classification), we report separately for each of the labels. This also helps us be aware of the different imbalanceness of different labels.
    """
    # top1, top5, total = 0.0, 0.0, 0
    top1, total = 0.0, 0
    num_test_samples, num_chunks = test_labels.shape[0], 1
    imgs_per_chunk = num_test_samples // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    
    for idx in range(0, num_test_samples, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_samples), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_samples)]
        batch_size = targets.shape[0]

        # calculate the similarity/distance and compute top-k neighbors
        if metric=='euclidean':
            similarity = torch.from_numpy(distance_matrix(features, train_features, p=2))  # p is minkowski's p
            distances, indices = similarity.topk(k, largest=False, sorted=True)
        elif metric=='cosine':
            features_norm = features / torch.norm(features, dim=1, keepdim=True)
            train_features_norm = train_features / torch.norm(train_features, dim=1, keepdim=True)  # L2-norm for cosine similarity
            similarity = torch.mm(features_norm, train_features_norm.T)
            distances, indices = similarity.topk(k, largest=True, sorted=True)
        
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)  # get labels of all topk neighbors, shape=[test_batch_size, k]

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()  # construct empty one-hot labels for topk neighbors, flattened, shape=[test_batch_size*k, num_classes]
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1).type(torch.int64), 1)  # fill in the one-hot labels for topk neighbors, flattened, shape=[test_batch_size*k, num_classes]
        distances_transform = distances.clone().div_(T).exp_()  # shape=[test_batch_size, k]
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        # top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    
    top1 = top1 / total
    # top5 = top5 * 100.0 / total
    
    return top1


def foscttm(R, E):
    foscttm_vec = torch.zeros(E.shape[0])
    for i in range(E.shape[0]):
        dist = torch.norm(R - E[i], dim=-1, keepdim=True)
        foscttm = torch.sum(dist < dist[i]) / dist.shape[0]  # get the percentage of elements closer than the true pair
        foscttm_vec[i] = foscttm
    mu, std = torch.mean(foscttm_vec), torch.std(foscttm_vec)       

    # Can also directly calculate using tensor operations, but it takes longer time:
    # dist = (str_embeds.repeat(str_embeds.shape[0], 1, 1) - both_emebds.unsqueeze(1)).norm(dim=-1, keepdim=True).squeeze(-1)
    # torch.mean(torch.sum((dist < dist.diagonal().unsqueeze(-1)), dim=1) / dist.shape[1])
    # torch.std(torch.sum((dist < dist.diagonal().unsqueeze(-1)), dim=1) / dist.shape[1]
    
    print(f"FOSCTTM Metrics, Mean: {mu}, std: {std}")
    
    return mu, std


# def kde_2d_density_plot():
#     pass


# def vmf_kde_angle_density_plot():
#     pass


def get_full_evaluate_mask_for_finetune_mode(finetune_mode, head_or_tail_masks_base):
    """
    Get full modality masks based on finetune_mode.
    For non-ablation modes, all modalities are available.
    For ablation modes, mask out the modalities that have never been seen during training. If the evaluation is between kg-kg  or cv-cv, then need to also ensure that the corresponding modality is always available.
    """
    head_or_tail_masks = head_or_tail_masks_base.clone()
    if 'ablation' in finetune_mode:
        head_or_tail_masks[:, FINETUNE_MODE_ABLATION_FULL_UNAVAIL_MAP[finetune_mode]] = True
        if 'kg_kg' in finetune_mode:
            head_or_tail_masks[:, 1] = False
        elif 'cv_cv' in finetune_mode:
            head_or_tail_masks[:, 2] = False
        elif 'tx_tx' in finetune_mode:
            head_or_tail_masks[:, 3:] = False

    return head_or_tail_masks


def get_modality_evaluate_mask(head_or_tail_masks_base, modality):
    if '+' not in modality:
        modality_col_list = MODALITY2NUMBER_LIST[modality]
        head_or_tail_masks = torch.ones_like(head_or_tail_masks_base)
        head_or_tail_masks[:, modality_col_list] = 0
        head_or_tail_masks = head_or_tail_masks.bool()
    else:  # NOTE: if '+' is in modality, e.g. 'str+cv+tx', we still want to keep the cv/tx modalities masked if they are not available for drugs --> set must-be masked modalities to 1 instead of those desired to 0
        modality_list = modality.split('+')
        head_or_tail_masks = head_or_tail_masks_base.clone()
        modality_col_list = []
        for modality in modality_list:
            modality_col_list.extend(MODALITY2NUMBER_LIST[modality])
        modality_must_mask_col_list = list(set(range(NUM_MODALITIES)) - set(modality_col_list))
        head_or_tail_masks[:, modality_must_mask_col_list] = 1  # Mask out the unavailable ones, leave the others as it is originally in the masks
        head_or_tail_masks = head_or_tail_masks.bool()
    return head_or_tail_masks


# TODO: Make this compatible with ablations (e.g. when missing kg)
def get_evaluate_masks(head_masks_base, tail_masks_base, eval_type, finetune_mode, device):
    """
    Get modality masks for different evaluation settings. 
    Current modalities: [str, kg, cv, tx_*]
    Eval types:
        - general: 'full_full', 'str_str', 'str_full'
        - 'easy'-splits-only: 'kg_full', 'str+kg_full', 'str+cv_full', 'str+tx_full', 'str+kg+cv_full', 'str+kg+tx_full', 'str+cv+tx_full', 'str+kg+cv+tx_full', 'kg_str', 'str+kg_str', 'str+cv_str', 'str+tx_str', 'str+kg+cv_str', 'str+kg+tx_str', 'str+cv+tx_str', 'str+kg+cv+tx_str', 'kg_kg', 'str+kg_kg', 'str+kg_str+kg', 'cv_cv', 'str+cv_str+cv', 'str+tx_str+tx', 'str+kg+cv_str+kg+cv', 'str+kg+tx_str+kg+tx', 'str+cv+tx_str+cv+tx', 'str+kg+cv+tx_str+kg+cv+tx'
    """
    assert len(eval_type.split('_')) == 2
    head_eval_mask_type, tail_eval_mask_type = eval_type.split('_')
    
    if head_eval_mask_type == 'full':
        head_masks = get_full_evaluate_mask_for_finetune_mode(finetune_mode, head_masks_base)
    else:
        head_masks = get_modality_evaluate_mask(head_masks_base, head_eval_mask_type)
        
    if tail_eval_mask_type == 'full':
        tail_masks = get_full_evaluate_mask_for_finetune_mode(finetune_mode, tail_masks_base)
    else:
        tail_masks = get_modality_evaluate_mask(tail_masks_base, tail_eval_mask_type)
        
    return head_masks.to(device), tail_masks.to(device)
