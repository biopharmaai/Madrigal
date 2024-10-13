import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
ROOT = Path(__file__).parent.resolve().parent

PROJECT_DIR = ROOT / "project_folder"
DATA_DIR = PROJECT_DIR / "datasets"
EMBEDDING_DIR = PROJECT_DIR / "embeddings"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
FIGURE_DIR = PROJECT_DIR / "figures"

TX_DATA_DIR = BASE_DIR + "raw_data/transcriptomics_signature/"
KG_DATA_DIR = BASE_DIR + "PrimeKG/"
OUTPUT_DIR = BASE_DIR + "processed_data/"
VIEW_OUTPUT_DIR = OUTPUT_DIR + "views_features_new/"
