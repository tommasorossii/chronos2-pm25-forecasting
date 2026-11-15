import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

RAW_CSV = DATA_DIR / "preprocessed_data.csv"
RAW_NAN_CSV = DATA_DIR / "data_without_preprocessing.csv"
PREPARED_CSV = DATA_DIR / os.environ.get("PREPARED_FILE", "osc_prepared.csv")

SELECTED_SENSORS = [
    "ari-2433",
    "fec01",
    "sensor bangalore",
    "sensor calgary",
    "sensor delhi",
    "sensor hamirpur",
    "5e8",
    "ari-2172",
    "ari-2263",
    "pms5003-2018062702491",
]

TIME_COL = "valid_at"
SENSOR_COL = "sensor_id"
CITY_COL = "city"
LCS_COL = "pm2p5_x"
RS_COL = "pm2p5_y"
MISSING_COL = "is_missing"
METEO_COLS = ["temperature", "relative_humidity", "pressure", "wind_speed"]

TRAIN_FRAC = 0.75

INPUT_WINDOWS = [12, 24, 48, 72, 512]
HORIZONS = [1, 12, 24, 48, 72]


def valid_configs():
    return [(i, h) for i in INPUT_WINDOWS for h in HORIZONS if h <= 2 * i]


MODEL_ID = "amazon/chronos-2"
DEVICE = "cuda"
SEED = 42

FT_MODE = "full"
FT_LR = 1e-5
FT_NUM_STEPS = 1000
FT_BATCH_SIZE = 32

PRED_BATCH_SIZE = 256
