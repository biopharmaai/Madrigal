
## 👀 Overview of Madrigal

## 🚀 Installation

1⃣️ First, clone the Github repository:

2⃣️ 

3⃣️ Download Datasets


### 🛠️ Training and Testing

### 🌟 Personalize based on your own dataset


### ⚖️ License

The code in this package is licensed under the MIT License.

</details>
# Madrigal
Multimodal drug combination outcome prediction.

NOTE: Not yet tested package installation. Use with caution.

## Installing `madrigal`
Before installing `madrigal`, please set up a new conda environment through `mamba env create -f env_no_build.yaml` (we recommend `mamba` instead of `conda`). Then activate this environment with `mamba activate primekg`.

To install a global reference to `madrigal` package in your interpreter (e.g. `from madrigal.X import Y`), run the following:
```
cd /path/to/Madrigal # Path to main repo directory
python -m pip install -e .
```
Note that the current code and notebooks still use `novelddi` as reference (`from novelddi.X import Y`). To be able to use the scripts and notebooks directly, please make sure to edit `setup.py` and change the line `name='madrigal'` to `name='novelddi'`.
Then, test the install by trying to import `madrigal` (or `novelddi`) in the interpreter:
```
python -c "import madrigal; print('Imported')" 
```
Now you should be able to use `import madrigal` (or `import novelddi`) from anywhere on your system, as long as you use the same python interpreter (recommend keeping to one conda env).  

## Using `madrigal` 
### Setting up data folder
We use `dotenv` to ensure transferrability of code between platforms. Please add a file `.env` to the project directory (root of this project) and specify the following paths:
```
PROJECT_DIR=<project_dir>
BASE_DIR=<root_data_dir>
DATA_DIR=<processed_data_dir>
ENCODER_CKPT_DIR=<encoder_checkpoints_dir>
CL_CKPT_DIR=<contrastive_pretraining_checkpoints_dir>
```

For example, in our system, the paths look like:
```
PROJECT_DIR="/home/<your_hms_id>/workspace/DDI/NovelDDI/"
BASE_DIR="/n/data1/hms/dbmi/zitnik/lab/users/<user>/DDI/"
DATA_DIR="/n/data1/hms/dbmi/zitnik/lab/users/<user>/DDI/processed_data/"
ENCODER_CKPT_DIR="/n/data1/hms/dbmi/zitnik/lab/users/<user>/NovelDDI/pretraining/"
CL_CKPT_DIR="/n/data1/hms/dbmi/zitnik/lab/users/<user>/DDI/model_output/pretrain/DrugBank/"
```

### Running pretraining and fine-tuning
We provide the first-stage pretraining scripts of respective modalities in `./modality_pretraining/`. The second-stage contrastive pretraining script is provided in `./scripts/cl_pretrain/`. The DDI fine-tuning script is provided in `./scripts/ddi_finetune/`. 

Currently, hard-coded paths to embedding checkpoints exist in the `get_str_encoder, get_kg_encoder, get_cv_encoder, and get_tx_encoder` functions in `./novelddi/model/models.py`. Corresponding modality pretrained checkpoints are provided in `./pretrained_checkpoints/first_stage/`. Please edit the paths to load checkpoints from your desired storage locations.

### Example data and second-stage checkpoints.
TODO: Upload example data. 
TODO: Upload second-stage pretraining checkpoints.

## Known issues
1. The `torchdrug` module needs to be imported after importing `torch_geometric` modules.
2. Newer version than `torchdrug=0.2.0.post1` is required, as earlier versions cause an [issue](https://github.com/DeepGraphLearning/torchdrug/issues/148) in LR scheduler.
