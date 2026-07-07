from pathlib import Path
import sys

TOOLS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOLS_DIR.parent

DATASETS_DIR = TOOLS_DIR / "datasets"
RAW_BIN_DIR = DATASETS_DIR / "raw_bin"
PROCESSED_DIR = DATASETS_DIR / "processed"

RESULTS_DIR = DATASETS_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
REPORTS_DIR = RESULTS_DIR / "reports"
LOGS_DIR = RESULTS_DIR / "logs"

PREPROCESSING_DIR = TOOLS_DIR / "preprocessing"
TRAINING_DIR = TOOLS_DIR / "training"
CLASSIFICATION_DIR = TOOLS_DIR / "classification"
REALTIME_DIR = TOOLS_DIR / "realtime"

FEATURE_DATASET_FILE = PROCESSED_DIR / "feature_dataset.pkl"
PREPROCESSING_PARAMETERS_FILE = PROCESSED_DIR / "preprocessing_parameters.json"

SELECTED_FEATURE_DATASET_FILE = PROCESSED_DIR / "selected_feature_dataset.pkl"
FEATURE_RANKING_FILE = PROCESSED_DIR / "feature_ranking.json"
FEATURE_SELECTION_PARAMETERS_FILE = PROCESSED_DIR / "feature_selection_parameters.json"

CLASSIFIER_FILE = PROCESSED_DIR / "classifier.pkl"
CLASSIFIER_PARAMETERS_FILE = PROCESSED_DIR / "classifier_parameters.json"

FINAL_PIPELINE_PARAMETERS_FILE = PROCESSED_DIR / "pipeline_parameters.json"

def setup_import_paths():
    paths = [
        TOOLS_DIR,
        CLASSIFICATION_DIR,
    ]

    for path in paths:
        path_str = str(path)

        if path_str not in sys.path:
            sys.path.insert(0, path_str)