import numpy as np
import pandas as pd

from monitor_drift import (
    CAT_COLS,
    FEATURE_COLUMNS,
    NUM_COLS,
    analyze_drift,
    generate_alert_markdown,
    log_inference,
    send_drift_alert,
)


def _make_bridge_df(
    age_mean: float,
    traffic_mean: float,
    materials: list,
    maint_levels: list,
    n: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "age_of_bridge": rng.normal(age_mean, 5.0, n),
            "traffic_volume": rng.normal(traffic_mean, 1000.0, n),
            "material_type": rng.choice(materials, size=n),
            "maintenance_level": rng.choice(maint_levels, size=n),
        },
        columns=FEATURE_COLUMNS,
    )


def test_log_inference_writes_record(tmp_path):
    """Verify that logging Bridge inference appends a valid structured record."""
    test_log = tmp_path / "test_logs.csv"
    input_sample = pd.DataFrame(
        [[30.0, 15000.0, "Concrete", "Medium"]],
        columns=FEATURE_COLUMNS,
    )

    log_inference(
        input_df=input_sample,
        prediction_code=1,
        prediction_label="Good",
        probabilities={"Poor": 0.12, "Good": 0.88},
        log_path=test_log,
    )

    assert test_log.exists()
    logged_df = pd.read_csv(test_log)
    assert len(logged_df) == 1
    assert int(logged_df.iloc[0]["predicted_class"]) == 1
    assert logged_df.iloc[0]["predicted_label"] == "Good"
    assert "timestamp" in logged_df.columns
    assert logged_df.iloc[0]["prob_Good"] == 0.88
    assert logged_df.iloc[0]["prob_Poor"] == 0.12


def test_analyze_drift_no_drift_when_identical():
    """Verify that identical numerical and categorical distributions report no drift."""
    df = _make_bridge_df(
        age_mean=35.0,
        traffic_mean=15000.0,
        materials=["Concrete", "Steel"],
        maint_levels=["Low", "Medium", "High"],
        n=150,
    )

    report = analyze_drift(df, df.copy())
    assert report["dataset_drift_detected"] is False
    assert report["drifted_features"] == 0
    assert report["drift_share_pct"] == 0.0


def test_analyze_drift_detects_distribution_shift():
    """Verify that significant numerical and categorical shifts trigger drift detection."""
    ref_df = _make_bridge_df(
        age_mean=30.0,
        traffic_mean=12000.0,
        materials=["Concrete"],
        maint_levels=["High"],
        n=150,
        seed=42,
    )
    curr_df = _make_bridge_df(
        age_mean=85.0,
        traffic_mean=45000.0,
        materials=["Wood"],
        maint_levels=["Low"],
        n=150,
        seed=99,
    )

    report = analyze_drift(ref_df, curr_df)
    assert report["dataset_drift_detected"] is True
    assert report["drifted_features"] == len(NUM_COLS) + len(CAT_COLS)


def test_generate_alert_markdown_creates_valid_report(tmp_path):
    """Verify that alert markdown is written with all tables and headers."""
    alert_path = tmp_path / "test_alert.md"
    ref_df = _make_bridge_df(
        age_mean=30.0,
        traffic_mean=12000.0,
        materials=["Concrete", "Steel"],
        maint_levels=["High"],
        n=120,
        seed=42,
    )
    curr_df = _make_bridge_df(
        age_mean=80.0,
        traffic_mean=40000.0,
        materials=["Wood"],
        maint_levels=["Low"],
        n=120,
        seed=7,
    )

    report = analyze_drift(ref_df, curr_df)
    md_text = generate_alert_markdown(report, output_path=alert_path)

    assert alert_path.exists()
    assert "DATA DRIFT DETECTED" in md_text
    assert "Continuous Training Required" in md_text
    assert "Statistical Drift Tests" in md_text


def test_send_drift_alert_graceful_without_webhook():
    """Verify send_drift_alert handles missing webhook without crashing."""
    report = {
        "dataset_drift_detected": True,
        "drifted_features": 4,
        "total_features": 4,
        "drift_share_pct": 100.0,
    }
    result = send_drift_alert(report, webhook_url=None)
    assert result is False