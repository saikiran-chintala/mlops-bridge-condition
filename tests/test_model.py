from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

MODEL_PATH = Path("models/model.pkl")
X_TRAIN_PATH = Path("data/processed/X_train.csv")

NUM_COLS = ["age_of_bridge", "traffic_volume"]
CAT_COLS = ["material_type", "maintenance_level"]
FEATURES = NUM_COLS + CAT_COLS
VALID_CLASSES = {0, 1}  # 0 = Poor, 1 = Good


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), f"Champion model not found at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def sample_categories():
    """Fetch valid categories from X_train.csv if available, else fallback defaults."""
    if X_TRAIN_PATH.exists():
        df = pd.read_csv(X_TRAIN_PATH)
        materials = sorted(df["material_type"].dropna().astype(str).unique().tolist())
        maint = sorted(df["maintenance_level"].dropna().astype(str).unique().tolist())
        if materials and maint:
            return materials, maint
    return ["Concrete", "Steel"], ["Low", "High"]


def test_model_pipeline_structure(model):
    """Ensure the champion is a Pipeline with Preprocessor and Model stages."""
    assert hasattr(model, "steps"), "Model should be a scikit-learn Pipeline"
    step_names = [name for name, _ in model.steps]
    assert "Preprocessor" in step_names, "Pipeline missing 'Preprocessor' stage"
    assert "Model" in step_names, "Pipeline missing 'Model' stage"


def test_single_row_inference(model, sample_categories):
    """Test inference on a single typical bridge record."""
    materials, maint = sample_categories
    sample = pd.DataFrame(
        [[25.0, 12000.0, materials[0], maint[0]]],
        columns=FEATURES,
    )
    pred = model.predict(sample)
    assert len(pred) == 1
    assert int(pred[0]) in VALID_CLASSES


def test_batch_inference_and_probabilities(model, sample_categories):
    """Test batch inference and verify binary probability distributions sum to ~1.0."""
    materials, maint = sample_categories
    mat_a = materials[0]
    mat_b = materials[-1]
    maint_a = maint[0]
    maint_b = maint[-1]

    batch = pd.DataFrame(
        [
            [5.0, 4000.0, mat_a, maint_b],
            [45.0, 22000.0, mat_b, maint_a],
            [85.0, 55000.0, mat_a, maint_a],
        ],
        columns=FEATURES,
    )

    preds = model.predict(batch)
    assert len(preds) == 3
    for p in preds:
        assert int(p) in VALID_CLASSES

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(batch)
        assert probas.shape == (3, 2)
        for row in probas:
            assert np.isclose(np.sum(row), 1.0, atol=1e-5)


def test_prediction_determinism(model, sample_categories):
    """Ensure identical inputs yield identical outputs (deterministic)."""
    materials, maint = sample_categories
    sample = pd.DataFrame(
        [[40.0, 18000.0, materials[0], maint[0]]],
        columns=FEATURES,
    )
    pred_1 = model.predict(sample)[0]
    pred_2 = model.predict(sample)[0]
    assert pred_1 == pred_2