<h1 align="center">
  Madrigal: A Unified Multimodal Model for Predicting Drug Combination Effects
</h1>

## 👀 Overview of Madrigal

Madrigal is an open-source model for predicting drug combination outcomes from multimodal preclinical data. This repository provides the implementation of the model as described in our [project page](https://zitniklab.hms.harvard.edu/projects/Madrigal/) and our [paper](https://arxiv.org/abs/2503.02781). 

## 🚀 Preparations

1. Clone this Github repository and install following the section [below](#installing-madrigal).
2. Set up data directories and create a `.env` file (see [below](#setting-up-data-and-checkpoint-directories)).
3. [Optional] Download datasets from our data [repo](https://doi.org/10.7910/DVN/ZFTW3J) in Harvard Dataverse and reorganize according to your `.env` setup.
4. [Optional] Download pretrained checkpoints from our checkpoint [repo](https://huggingface.co/mims-harvard/Madrigal/tree/main) in Huggingface and reorganize according to your `.env` setup.

## 🛠️ Training and Testing

We provide sample model pretraining (second-stage modality alignment) and training scripts in `scripts/`. Specifically, the second-stage pretraining scripts are provided in [`./scripts/cl_pretrain/`](https://github.com/mims-harvard/Madrigal/tree/main/scripts/cl_pretrain), and the fine-tuning scripts are provided in [`./scripts/ddi_finetune/`](https://github.com/mims-harvard/Madrigal/tree/main/scripts/ddi_finetune). The scripts will need to be adapted according to your machine. 

The first-stage modality adaptation training scripts (or notebooks) and checkpoints can be found in `modality_pretraining/`. You can also run inference with model checkpoints using sample Jupyter notebooks (to be uploaded).

## 🌟 Personalize based on your own dataset

Currently, modifications of the codebase are required to enable adaptation of the model to your own dataset. Below is an outline of possible preparations.
- There are certain arguments that require modifications (see `./madrigal/parse_args.py`) if you are incorporating a new dataset.
  - `data_source`: This arg affects path to load data and training and evaluation strategy.
  - `split_method`: This arg affects path to load data and evaluation strategy.
  - `task`: Depending on the nature of your dataset, you might want to change this.
  - `loss_fn_name`: Depending on `task`, you might want to change or reimplement this.
- Preparing data: Please refer to our provided data files for the exact formatting of each file.
  - Drugs
    - Metadata: Key to all other files.
    - Modality data
      - Structure: Use `torchdrug` to generate molecular graphs in the same way as molecules are ordered in metadata.
      - KG: Use `PyG` to generate `HeteroData` objects, making sure drug node indices are ordered in the same way as in metadata.
      - Cell viability: Mainly tables.
      - Transcriptomics: Mainly tables.
        - Note that you will need to regenerate a file (hard-coded as `rdkit2D_embeddings_combined_all_normalized.parquet`) for chemCPA usage/pretraining.
  - Drug combination outcomes
    - Tables of (label_indexed, head (drug 1), tail (drug 2), negs*) (depending on dataset splitting strategy, the negative columns will have different meanings).
    - Mapping between outcome label index and outcome information.

## Detailed instructions

### Installing `madrigal`
Before installing `madrigal`, please set up a new conda environment through `mamba env create -f env_new.yaml` (this process should take less than an hour; see `mamba` installation guidelines [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)). By default, our environment is with CUDA 11.7 (gcc 9.2). Please edit `env_new.yaml` accordingly if you are installing in another CUDA version. We welcome contributions of instructions on setting up the environment with other version control managers such as `uv`.

Then, activate this environment with `mamba activate primekg`. To install a global reference to `madrigal` package in your interpreter (e.g. `from madrigal.X import Y`), run the following:
```
cd /path/to/Madrigal
python -m pip install -e .
```
Then, test the install by trying to import `madrigal` in the interpreter:
```
python -c "import madrigal; print('Imported')" 
```
Now you should be able to use `import madrigal` from anywhere on your system, as long as you use the same python interpreter.  

### Setting up data and checkpoint directories
We organize our data and model output folders in the following way:
```
Madrigal_Data
|-- processed_data
|  |-- polypharmacy_new
|  |  |-- DrugBank
|  |  |  |-- split_by_*
|  |  |  |  |-- data tables
|  |-- views_features_new
|  |  |-- metadata tables
|  |  |-- str
|  |  |  |-- torchdrug-generated molecular graphs
|  |  |-- kg
|  |  |  |-- PyG-generated KGs
|  |  |-- cv
|  |  |  |-- cell viability tables
|  |  |-- tx
|  |  |  |-- transcriptomics tables
|-- model_output
|  |-- pretrain
|  |  |-- DrugBank
|  |  |  |-- split_by_*
|  |-- DrugBank
|  |  |-- split_by_*
```
This structure is reflected in the model code. Please make necessary edits if you are using a different organization.

Then, please add a file `.env` to the project directory (root of this project) and specify the following paths (with `/` at the end):
```
PROJECT_DIR=/path/to/Madrigal/
BASE_DIR=/path/to/Madrigal_Data/
DATA_DIR=/path/to/Madrigal_Data/processed_data/
ENCODER_CKPT_DIR=/path/to/Madrigal/modality_pretraining/
CL_CKPT_DIR=/path/to/Madrigal_Data/model_output/pretrain/
```

### ⚖️ License

The code in this package is licensed under the MIT License. 

## Known issues
1. The `torchdrug` module needs to be imported after importing `torch_geometric` modules.
2. `torchdrug>=0.2.0.post1` is required, as earlier versions cause an [issue](https://github.com/DeepGraphLearning/torchdrug/issues/148) in LR scheduler.
3. We use `pytorch=1.13.1`, which requires `cuda<12.0`.
4. (Updated `env_new.yaml` to resolve this issue.) ~If you encounter `TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'` while installing, please check out [this post](https://github.com/pypa/setuptools/issues/4483). In summary, either `setuptools<71` or `packaging>=22` is required.~

## Citation
Please find our preprint at https://arxiv.org/abs/2503.02781.
```
@article{Huang2025.arXiv:2503.02781,
  author = {Huang, Yepeng and Su, Xiaorui and Ullanat, Varun and Liang, Ivy and Clegg, Lindsay and Olabode, Damilola and Ho, Nicholas and John, Bino and Gibbs, Megan and Zitnik, Marinka},
  title = {Multimodal AI predicts clinical outcomes of drug combinations from preclinical data},
  journal = {arXiv preprint arXiv:2503.02781},
  year = {2025},
  doi = {10.48550/arXiv.2503.02781},
  URL = {https://arxiv.org/abs/2503.02781},
}
```
