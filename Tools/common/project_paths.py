from pathlib import Path
import sys

TOOLS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_DIR.parent

DATASETS_DIR = TOOLS_DIR  / "datasets"
RAW_BIN_DIR = DATASETS_DIR / "raw_bin"
PROCESSED_DIR = DATASETS_DIR / "processed"

PREPROCESSING_DIR = TOOLS_DIR / "preprocessing"
CLASSIFICATION_DIR = TOOLS_DIR / "classification"

FEATURE_DATASET_FILE = PROCESSED_DIR / "feature_dataset.pkl"
PREPROCESSING_PARAMETERS_FILE = PROCESSED_DIR / "preprocessing_parameters.json"

PIPELINE_PARAMETERS_FILE = PREPROCESSING_DIR / "pipeline_parameters.json"

def setup_import_paths():
    paths = [
        TOOLS_DIR,
        CLASSIFICATION_DIR,
    ]

    for path in paths:
        path_str = str(path)

        if path_str not in sys.path:
            sys.path.insert(0, path_str)