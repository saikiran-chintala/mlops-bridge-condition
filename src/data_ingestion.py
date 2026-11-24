from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/bridge_data.csv")
OUTPUT_PATH = Path("data/ingested/bridge_data.csv")

REQUIRED_COLUMNS = [
    "age_of_bridge",
    "traffic_volume",
    "material_type",
    "maintenance_level",
    "bridge_condition",
]


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}. Put the Bridge CSV at this location."
        )

    data = pd.read_csv(RAW_PATH)

    # Normalize column names slightly before checking to prevent whitespace/case mismatches
    data.columns = data.columns.str.strip().str.lower()

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Keep only model-relevant columns (drops any extra ID or irrelevant columns)
    data = data[REQUIRED_COLUMNS].copy()

    if data.empty:
        raise ValueError("The input dataset is empty.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Data ingestion completed: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
