from pathlib import Path
import pandas as pd
import pytest

RAW_DATA_PATH = Path("data/raw/bridge_data.csv")
INGESTED_DATA_PATH = Path("data/ingested/bridge_data.csv")
TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")

NUM_COLS = ["age_of_bridge", "traffic_volume"]
CAT_COLS = ["material_type", "maintenance_level"]
EXPECTED_FEATURES = NUM_COLS + CAT_COLS
EXPECTED_TARGET = "bridge_condition"
VALID_CLASSES = {0, 1}  # 0 = Poor, 1 = Good


def test_raw_dataset_exists_and_valid():
    """Verify raw Bridge dataset exists, has expected columns, and valid target values."""
    assert RAW_DATA_PATH.exists(), f"Raw dataset not found at {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    assert len(df) > 0, "Raw Bridge dataset is empty"
    for col in EXPECTED_FEATURES + [EXPECTED_TARGET]:
        assert col in df.columns, f"Missing column {col} in raw data"

    assert df[EXPECTED_FEATURES].isnull().sum().sum() == 0, "Null values found in features"
    assert set(df[EXPECTED_TARGET].astype(int).unique()).issubset(VALID_CLASSES)


def test_preprocessed_splits_exist_and_consistent():
    """Verify train and test sets exist, are deduplicated, and preserve stratified classes."""
    assert TRAIN_DATA_PATH.exists(), f"Train set not found at {TRAIN_DATA_PATH}"
    assert TEST_DATA_PATH.exists(), f"Test set not found at {TEST_DATA_PATH}"

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    assert len(train_df) > 0 and len(test_df) > 0, "Train or test split is empty"

    if INGESTED_DATA_PATH.exists():
        ingested_df = pd.read_csv(INGESTED_DATA_PATH)
        ingested_df.columns = ingested_df.columns.str.strip().str.lower()
        for c in CAT_COLS:
            ingested_df[c] = ingested_df[c].astype(str).str.strip()
        expected_dedup_rows = len(ingested_df.drop_duplicates())
        total_rows = len(train_df) + len(test_df)
        assert (
            total_rows == expected_dedup_rows
        ), f"Expected {expected_dedup_rows} deduplicated rows across splits, got {total_rows}"

    for col in EXPECTED_FEATURES + [EXPECTED_TARGET]:
        assert col in train_df.columns, f"Missing {col} in train.csv"
        assert col in test_df.columns, f"Missing {col} in test.csv"

    # Verify both splits contain both binary classes (stratified)
    assert set(train_df[EXPECTED_TARGET].astype(int).unique()) == VALID_CLASSES
    assert set(test_df[EXPECTED_TARGET].astype(int).unique()) == VALID_CLASSES