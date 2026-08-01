"""
Central configuration for the IDS three-axis evaluation study.

All paths, feature lists, seeds, and hyperparameters live here so they can be
imported from any module. To reproduce on a different machine, only the paths
in this file need to change.
"""

import os

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# Base directory containing the raw NetFlow v2 parquet files and outputs.
# On Colab: '/content/drive/MyDrive/CybersecClassification'
# Locally:  './data'
BASE_DIR = os.environ.get("IDS_BASE_DIR", "./data")
OUTPUT_DIR = os.environ.get("IDS_OUTPUT_DIR", os.path.join(BASE_DIR, "ids_output"))

# Input datasets (download from UQ NetFlow v2 collection).
UNSW_PATH = os.path.join(BASE_DIR, "NF-UNSW-NB15-V2.parquet")
CIC_PATH = os.path.join(BASE_DIR, "NF-CSE-CIC-IDS2018-V2.parquet")

# Cached balanced samples (created by data_loading.load_balanced).
UNSW_BALANCED = os.path.join(OUTPUT_DIR, "unsw_balanced_50k.parquet")
CIC_BALANCED = os.path.join(OUTPUT_DIR, "cic_balanced_50k.parquet")

# Trained models and results.
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

RF_MODEL = os.path.join(MODELS_DIR, "rf_baseline.joblib")
XGB_MODEL = os.path.join(MODELS_DIR, "xgb_29feat.joblib")
LLM_ADAPTER_UNSW = os.path.join(MODELS_DIR, "roberta_lora_unsw_full")
LLM_ADAPTER_CIC = os.path.join(MODELS_DIR, "roberta_lora_cic_full")


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in (OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# LOCKED FEATURE SET (from Block 3.8 cross-dataset ablation)
# ---------------------------------------------------------------------------
# The full NetFlow v2 schema has 43 columns. We drop:
#   - Bias-prone identifiers: L4_SRC_PORT, L4_DST_PORT, DNS_QUERY_ID
#   - Label columns: Label (binary target), Attack (multiclass, taxonomy differs)
#   - Environmental artifacts detected via cross-dataset ablation:
#         MIN_TTL, MAX_TTL          (OS + hop-count fingerprint)
#         MIN_IP_PKT_LEN, MAX_IP_PKT_LEN, SHORTEST_FLOW_PKT, LONGEST_FLOW_PKT
#                                    (tool-specific packet size fingerprint)
#         SERVER_TCP_FLAGS, TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT
#                                    (OS network-stack fingerprint)
#
# Set C (29 features) locked because it delivered the best cross-dataset
# transfer (0.822 UNSW->CIC) with negligible same-dataset degradation.

FEATURES = [
    "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
]

LABEL = "Label"

DROPPED_BIAS = ["L4_SRC_PORT", "L4_DST_PORT", "DNS_QUERY_ID"]
DROPPED_ARTIFACTS = [
    "MIN_TTL", "MAX_TTL",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SHORTEST_FLOW_PKT", "LONGEST_FLOW_PKT",
    "SERVER_TCP_FLAGS", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
]

# ---------------------------------------------------------------------------
# EXPERIMENTAL CONFIG
# ---------------------------------------------------------------------------
N_PER_CLASS = 50_000       # balanced sample size per class per dataset
SEEDS = [42, 43, 44, 45]   # for multi-seed classical baselines (Bui-style)
DEFAULT_SEED = 42

# LLM training config
LLM_MODEL_NAME = "roberta-base"
LLM_MAX_LEN = 256
LLM_BATCH_SIZE = 32        # increase on larger GPUs (48+ on L4/A100)
LLM_EPOCHS = 3
LLM_LR = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query", "key", "value"]

# Adversarial evasion config
# Features that are categorical/bounded and must NOT be perturbed.
ADV_FIXED_FEATURES = [
    "PROTOCOL", "L7_PROTO", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
]
ADV_EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
