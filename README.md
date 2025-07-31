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

The first-stage modality adaptation training scripts (or notebooks) and checkpoints can be found in `modality_pretraining/`. You can also run inference with model checkpoints using sample Jupyter notebooks:
- [generate_embeddings](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/generate_embeddings.ipynb): Generate embeddings and raw scores. Also contains scripts to normalize prediction scores so that they can be used for direct comparisons.
- [quick_predictions](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/quick_predictions.ipynb): Get raw scores and normalized ranks for specific queries of [outcome, drug A, drug B]. 

## Notebooks
| Notebook | Content | Requirement |
|-|-|-|
| [fig1_pretrained_embeds](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/fig1/fig1_pretrained_embeds.ipynb) | Plot UMAP of pretrained modality embeddings (Fig. 1d) | Harvard Dataverse, Huggingface |
| [fig2_model_analyses](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/fig2/fig2_model_analyses.ipynb) | Performance change with drug similarity (Fig. 2c,d) | Harvard Dataverse, Huggingface |
| [fig2_modality_ablations](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/fig2/fig2_modality_ablations.ipynb) | Outcome-specific AUPRC of Madrigal and ablation models (Fig. 2e) | Harvard Dataverse, Huggingface |
| [fig3_self_combo](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/fig3/fig3_self_combo.ipynb) | External validation with FDA safety rankings (Fig. 3a-c) | Harvard Dataverse, Huggingface |
| [fig3_transporter_mediated_ddis](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/fig3/fig3_transporter_mediated_ddis.ipynb) | Transporter-mediated DDIs (Fig. 3d-f) | Harvard Dataverse, Huggingface, Normalized rank |
| [fig4_clinical_trials_combos](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/fig4/fig4_clinical_trials_combos.ipynb) | Evaluation with clinical trials data (Fig. 4b) | Harvard Dataverse, Huggingface, ToolUniverse |
| [fig4_parpi](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/fig4/fig4_clinical_trials_combos.ipynb) | Evaluation with PARPi combinations (Fig. 4c) | Harvard Dataverse, Huggingface, Normalized rank |
| [fig5_t2d_mash](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/fig5/fig5_t2d_mash.ipynb) | Evaluation with combinations in metabolic disorders (Fig. 5) | Harvard Dataverse, Huggingface, Normalized rank |
| [fig6_PDX](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/fig6/fig6_PDX.ipynb) | Individualized predictions with PDXE (Fig. 6c-f) | Harvard Dataverse, Huggingface |
| [fig6_clinical_validation_dfci](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/fig6/fig6_clinical_validation_dfci.ipynb) | Analyses with DFCI cohort (Fig. 6j) | Harvard Dataverse, Huggingface; (access to patient data is restricted) |
| [discussions_proteomics_analysis](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/discussions/discussions_proteomics_analysis.ipynb) | Correlation with proteomics data | Harvard Dataverse, Huggingface |
| [discussions_combomatch](https://github.com/mims-harvard/Madrigal/blob/main/notebooks/discussions/discussions_combomatch.ipynb) | Inference on ComboMATCH drug pairs | Harvard Dataverse, Huggingface, Normalized rank |

Requirements:
- [Harvard Dataverse](https://doi.org/10.7910/DVN/ZFTW3J): Required for running all notebooks.
- [Huggingface](https://huggingface.co/mims-harvard/Madrigal/tree/main): Required for running all notebooks.
- [Normalized rank](https://drive.google.com/file/d/1wvgM5-VoVmnK8C8ixTkcCQwWVZwyf9dS/view?usp=sharing): The full normalized rank tensor (80GB) used in some notebooks. 
- [ToolUniverse](https://github.com/mims-harvard/ToolUniverse): Used to extract clinical trials adverse events data.

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
Before installing `madrigal`, please set up a new conda environment through `mamba env create -f env_new.yaml` (this process should take less than an hour; see `mamba` installation guidelines [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)). By default, our environment is with CUDA 11.7 + gcc 9.2. Please edit `env_new.yaml` accordingly if you are installing in another CUDA version. We welcome contributions of instructions on setting up the environment with other version control managers such as `uv`.

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
Madrigal_Data (BASE_DIR)
|-- processed_data
|  |-- polypharmacy_new (in Harvard Dataverse)
|  |  |-- DrugBank
|  |  |  |-- split_by_*
|  |  |  |  |-- data tables
|  |-- views_features_new (in Harvard Dataverse)
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
|  |-- pretrain (in Huggingface)
|  |  |-- DrugBank
|  |  |  |-- split_by_*
|  |-- DrugBank (in Huggingface)
|  |  |-- split_by_*
|-- raw_data (data used in analyses, in Harvard Dataverse)
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
  - If you have only `cuda>12.0`, our incomplete test indicates that `pytorch=2.1.0` with `pytorch-geometric<2.4.0` might be compatible (CUDA 12.8 + gcc 14.2 + PyTorch 2.1.0). 
4. (Updated `env_new.yaml` to resolve this issue.) ~If you encounter `TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'` while installing, please check out [this post](https://github.com/pypa/setuptools/issues/4483). In summary, either `setuptools<71` or `packaging>=22` is required.

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
