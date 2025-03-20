from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_LAKE_RAW = BASE_DIR / "data_lake" / "raw"
DATA_LAKE_REFINED = BASE_DIR / "data_lake" / "refined"
MODEL_DIR = BASE_DIR / "trained_model"
REPORT_DIR = BASE_DIR / "generated_reports"
DASH_DIR = BASE_DIR / "generated_dashboards"

for directory in [DATA_LAKE_RAW, DATA_LAKE_REFINED, MODEL_DIR, REPORT_DIR, DASH_DIR]:
    directory.mkdir(parents=True, exist_ok=True)