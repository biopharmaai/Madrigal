import pandas as pd
import numpy as np
import os, pickle
from tqdm import tqdm
from time import time
from multiprocessing import Pool

from madrigal.utils import DATA_DIR, BASE_DIR

data_source = 'DrugBank'
split_method = 'split_by_pairs'
repeat = None
# checkpoint = 'drawn-grass-4'  # DrugBank (trained on all data, seed=1)
# checkpoint = 'misty-oath-5'  # DrugBank (trained on all data, seed=0)
# checkpoint = 'whole-fog-7'  # DrugBank (trained on all data, seed=99)
# checkpoint = 'snowy-serenity-8'  # DrugBank (trained on all data, seed=42)
checkpoint = 'revived-aardvark-8'  # DrugBank (trained on all data, seed=2)

# epoch = None
epoch = 700
kg_encoder = 'hgt'
checkpoint_dir = BASE_DIR + f'model_output/{data_source}/{split_method}/{checkpoint}/'
if epoch is None:
    ckpt = checkpoint_dir + "best_model.pt"
else:
    ckpt = checkpoint_dir + f"checkpoint_{epoch}.pt"
finetune_mode = 'str_str+random_sample'
eval_type = "full_full"
raw_scores = np.load(checkpoint_dir + f"{data_source}_drugs_raw_scores_{epoch}.npy", mmap_mode="r+")

# Start normalizing the scores
raw_scores_norm = np.memmap(
    f"{checkpoint_dir}/{data_source}_drugs_normalized_ranks_{epoch}.raw", 
    mode="w+", dtype=np.float32, shape=raw_scores.shape
)
mask_indices = np.vstack(np.triu_indices(raw_scores.shape[1], k=0, m=raw_scores.shape[2]))
interval = 1

def classwise_normalized_rank_3d_numpy(tensor):
    # flatten the tensor while maintaining the class dimension
    # print("Flattening...")
    flat_tensor = tensor.reshape(tensor.shape[0], -1)
    
    # compute the ranks
    # print("Computing ranks (via argsort), will take quite a while...")
    # start = time()
    
    if tensor.shape[0] > 1:
        flat_rank = flat_tensor.argsort(axis=1).argsort(axis=1) + 1
    else:
        temp = flat_tensor.argsort(axis=1)
        flat_rank = np.empty_like(temp)
        flat_rank[0, temp] = np.arange(flat_rank.shape[1]) + 1
        del temp

    # end = time()
    # print(f"Finished computing ranks in {(end - start):.4f} seconds.")
    
    # normalize the ranks
    normalized_rank = flat_rank / (tensor.shape[1] * (tensor.shape[2] - 1) / 2)

    # reshape back to the original shape
    return normalized_rank.reshape(tensor.shape)

def run_slice(tup):
    st = time()
    start, end = tup
    raw_scores_slice = raw_scores[start:end, :, :]
    raw_scores_slice = raw_scores_slice.copy()
    raw_scores_slice[:, mask_indices[0], mask_indices[1]] = 1e7
    raw_scores_slice_norm = classwise_normalized_rank_3d_numpy(raw_scores_slice)
    raw_scores_slice_norm[:, mask_indices[0], mask_indices[1]] = 0
    raw_scores_slice_norm = raw_scores_slice_norm + raw_scores_slice_norm.swapaxes(1, 2)
    # assert raw_scores_slice_norm.max() < 1e7
    raw_scores_norm[start:end, :, :] = raw_scores_slice_norm
    e = time()
    print(f"Finished normalizing class {start} in {((e - st) / 60):.4f} minutes.")

print("Starting to normalize the scores...")
st = time()
with Pool() as pool:
    pool.map(
        run_slice, 
        zip(
            np.arange(0, raw_scores.shape[0], interval), 
            np.arange(0, raw_scores.shape[0], interval)[1:].tolist() + [raw_scores.shape[0]]
        )
    )
e = time()
print(f"No numba takes {(e-st):.4f} seconds to run the whole.")

with open(f"{checkpoint_dir}/{data_source}_drugs_normalized_ranks_{epoch}.npy", "wb") as f:
    np.save(f, raw_scores_norm)
