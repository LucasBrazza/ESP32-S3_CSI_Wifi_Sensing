# ================= PREPROCESSING =================

WINDOW_SIZE = 5
STEP_SIZE = 2

CORRELATION_THRESHOLD = 0.95
MIN_INFORMATIVE_STD = 1e-6

HAMPEL_WINDOW_SIZE = 5
HAMPEL_N_SIGMAS = 3.0

MOVING_AVERAGE_WINDOW_SIZE = 3


# ================= FEATURE EXTRACTION =================

FEATURES_PER_SUBCARRIER = 6

FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "peak_to_peak",
    "energy",
]


# ================= FEATURE SELECTION =================

TOP_K_FEATURES = 126
FISHER_MINIMUM_SCORE = 0.0


# ================= CLASSIFIER =================

MODEL = "decision_tree"

MAX_TREE_DEPTH = 6
MIN_SAMPLES_SPLIT = 5