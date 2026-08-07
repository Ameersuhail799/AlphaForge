"""
AlphaForge Configuration

Central configuration for the entire project.
Nothing should be hardcoded outside this file.
"""

from pathlib import Path

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# DATA DIRECTORIES
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# ==========================================================
# MODEL DIRECTORIES
# ==========================================================

MODEL_DIR = PROJECT_ROOT / "models"

TRAINED_MODEL_DIR = MODEL_DIR / "trained"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# ==========================================================
# REPORTS
# ==========================================================

REPORT_DIR = PROJECT_ROOT / "reports"

FIGURE_DIR = REPORT_DIR / "figures"
BACKTEST_DIR = REPORT_DIR / "backtests"
EXPERIMENT_REPORT_DIR = REPORT_DIR / "experiments"
FEATURE_IMPORTANCE_REPORT_DIR = REPORT_DIR / "feature_importance"
ANALYSIS_REPORT_DIR = REPORT_DIR / "analysis"

# ==========================================================
# LOGGING
# ==========================================================

LOG_LEVEL = "INFO"

# ==========================================================
# MARKET SETTINGS
# ==========================================================

DEFAULT_SYMBOL = "RELIANCE.NS"

START_DATE = "2000-01-01"

INTERVAL = "1d"

# ==========================================================
# STORAGE
# ==========================================================

FILE_FORMAT = "parquet"

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_STATE = 42
