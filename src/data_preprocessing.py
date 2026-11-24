from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("data/ingested/bridge_data.csv")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

TARGET = "bridge_condition"
CAT_COLS = ["material_type", "maintenance_level"]


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Ingested dataset not found: {INPUT_PATH}. Run data_ingestion.py first."
        )

    data = pd.read_csv(INPUT_PATH)

    # Standardize column names and strip whitespace from categorical values
    data.columns = data.columns.str.strip().str.lower()
    for c in CAT_COLS:
        if c in data.columns:
            data[c] = data[c].astype(str).str.strip()

    # Remove exact duplicate records before splitting
    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(data)} duplicate rows.")

    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(params["test_size"]),
        random_state=int(params["random_state"]),
        stratify=y,
    )

    train = X_train.copy()
    train[TARGET] = y_train
    test = X_test.copy()
    test[TARGET] = y_test

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"Training data: {train.shape}")
    print(f"Testing data: {test.shape}")
    print(f"Saved: {TRAIN_PATH} and {TEST_PATH}")


if __name__ == "__main__":
    main()
