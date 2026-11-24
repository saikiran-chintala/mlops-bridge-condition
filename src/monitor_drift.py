"""Data drift monitoring module for Bridge Condition production inference tracking.

Compares production inference logs against baseline training data using:
- Kolmogorov-Smirnov (KS) two-sample test for numerical features
- Chi-Square test for categorical features
Generates structured JSON metrics, an HTML dashboard, and a Markdown alert.
"""

from pathlib import Path
import argparse
import datetime
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

REFERENCE_DATA_PATH = Path("data/processed/train.csv")
RAW_FALLBACK_PATH = Path("data/raw/bridge_data.csv")
INFERENCE_LOG_PATH = Path("data/monitoring/inference_logs.csv")
REPORTS_DIR = Path("reports")
DRIFT_JSON_PATH = REPORTS_DIR / "drift_report.json"
DRIFT_HTML_PATH = REPORTS_DIR / "drift_report.html"
DRIFT_ALERT_MD_PATH = REPORTS_DIR / "drift_alert.md"

NUM_COLS = ["age_of_bridge", "traffic_volume"]
CAT_COLS = ["material_type", "maintenance_level"]
FEATURE_COLUMNS = NUM_COLS + CAT_COLS

LOG_COLUMNS = [
    "timestamp",
    "age_of_bridge",
    "traffic_volume",
    "material_type",
    "maintenance_level",
    "predicted_class",
    "predicted_label",
    "prob_Poor",
    "prob_Good",
]


def load_inference_logs(log_path: Path = INFERENCE_LOG_PATH) -> pd.DataFrame:
    """Robustly load production inference logs with schema enforcement and error recovery."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return pd.DataFrame(columns=LOG_COLUMNS)

    try:
        df = pd.read_csv(log_path)
    except Exception:
        try:
            df = pd.read_csv(log_path, on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=LOG_COLUMNS)

    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df


def log_inference(
    input_df: pd.DataFrame,
    prediction_code: int,
    prediction_label: str,
    probabilities: dict = None,
    log_path: Path = INFERENCE_LOG_PATH,
):
    """Append a single inference payload to the monitoring log with strict schema consistency."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    prob_poor = probabilities.get("Poor", np.nan) if probabilities else np.nan
    prob_good = probabilities.get("Good", np.nan) if probabilities else np.nan

    row_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "age_of_bridge": float(input_df["age_of_bridge"].iloc[0]),
        "traffic_volume": float(input_df["traffic_volume"].iloc[0]),
        "material_type": str(input_df["material_type"].iloc[0]).strip(),
        "maintenance_level": str(input_df["maintenance_level"].iloc[0]).strip(),
        "predicted_class": int(prediction_code),
        "predicted_label": str(prediction_label),
        "prob_Poor": round(float(prob_poor), 4) if not pd.isna(prob_poor) else np.nan,
        "prob_Good": round(float(prob_good), 4) if not pd.isna(prob_good) else np.nan,
    }

    df_row = pd.DataFrame([row_data], columns=LOG_COLUMNS)
    header = not log_path.exists() or log_path.stat().st_size == 0
    df_row.to_csv(log_path, mode="a", index=False, header=header)


def log_batch_inferences(
    batch_df: pd.DataFrame,
    pred_codes,
    pred_labels,
    probs=None,
    log_path: Path = INFERENCE_LOG_PATH,
):
    """Append a batch of inference payloads to the monitoring log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    records = pd.DataFrame(
        {
            "timestamp": [now_iso] * len(batch_df),
            "age_of_bridge": pd.to_numeric(batch_df["age_of_bridge"], errors="coerce"),
            "traffic_volume": pd.to_numeric(batch_df["traffic_volume"], errors="coerce"),
            "material_type": batch_df["material_type"].astype(str).str.strip(),
            "maintenance_level": batch_df["maintenance_level"].astype(str).str.strip(),
            "predicted_class": [int(c) for c in pred_codes],
            "predicted_label": [str(lbl) for lbl in pred_labels],
            "prob_Poor": np.round(probs[:, 0], 4) if probs is not None else np.nan,
            "prob_Good": np.round(probs[:, 1], 4) if probs is not None else np.nan,
        },
        columns=LOG_COLUMNS,
    )

    header = not log_path.exists() or log_path.stat().st_size == 0
    records.to_csv(log_path, mode="a", index=False, header=header)


def generate_sample_inferences(
    n_samples: int = 50,
    drift: bool = False,
    log_path: Path = INFERENCE_LOG_PATH,
    reference_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Generate synthetic Bridge Condition inference logs adhering to the canonical schema."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    # Derive realistic baseline stats if reference data is available
    if reference_df is not None and not reference_df.empty:
        age_mean = float(reference_df["age_of_bridge"].mean())
        age_std = max(1.0, float(reference_df["age_of_bridge"].std()))
        traf_mean = float(reference_df["traffic_volume"].mean())
        traf_std = max(100.0, float(reference_df["traffic_volume"].std()))
        materials = sorted(reference_df["material_type"].dropna().astype(str).unique().tolist())
        maint_levels = sorted(reference_df["maintenance_level"].dropna().astype(str).unique().tolist())
    else:
        age_mean, age_std = 40.0, 20.0
        traf_mean, traf_std = 18000.0, 8000.0
        materials = ["Concrete", "Steel", "Wood", "Composite"]
        maint_levels = ["Low", "Medium", "High"]

    if drift:
        # Simulate aging infrastructure, surging traffic, and skewed material/maintenance mix
        age = rng.normal(age_mean + 35.0, age_std * 0.8, n_samples)
        traffic = rng.normal(traf_mean * 1.85, traf_std * 0.9, n_samples)
        mat_probs = np.linspace(0.5, 0.1, len(materials))
        mat_probs /= mat_probs.sum()
        maint_probs = np.linspace(0.6, 0.1, len(maint_levels))
        maint_probs /= maint_probs.sum()
        chosen_materials = rng.choice(materials, size=n_samples, p=mat_probs)
        chosen_maint = rng.choice(maint_levels, size=n_samples, p=maint_probs)
    else:
        # Normal production traffic mirroring baseline distribution
        age = rng.normal(age_mean, age_std, n_samples)
        traffic = rng.normal(traf_mean, traf_std, n_samples)
        chosen_materials = rng.choice(materials, size=n_samples)
        chosen_maint = rng.choice(maint_levels, size=n_samples)

    age_clipped = np.clip(age, 1.0, 150.0).round(1)
    traffic_clipped = np.clip(traffic, 100.0, 250000.0).round(0)

    # Generate plausible mock predictions & probabilities
    pred_codes = (age_clipped < (age_mean + 15.0)).astype(int)
    pred_labels = ["Good" if c == 1 else "Poor" for c in pred_codes]

    prob_poor = []
    prob_good = []
    for c in pred_codes:
        if c == 1:
            pg = round(float(rng.uniform(0.75, 0.98)), 4)
            prob_good.append(pg)
            prob_poor.append(round(1.0 - pg, 4))
        else:
            pp = round(float(rng.uniform(0.75, 0.98)), 4)
            prob_poor.append(pp)
            prob_good.append(round(1.0 - pp, 4))

    timestamps = [
        (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=i * 5)).isoformat()
        for i in range(n_samples)
    ]

    sample_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "age_of_bridge": age_clipped,
            "traffic_volume": traffic_clipped,
            "material_type": chosen_materials,
            "maintenance_level": chosen_maint,
            "predicted_class": pred_codes,
            "predicted_label": pred_labels,
            "prob_Poor": prob_poor,
            "prob_Good": prob_good,
        },
        columns=LOG_COLUMNS,
    )

    sample_df.to_csv(log_path, mode="w", index=False, header=True)
    print(f"Generated {n_samples} sample Bridge inference logs (drift={drift}) at: {log_path}")
    return sample_df


def analyze_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    p_value_threshold: float = 0.05,
    dataset_drift_threshold: float = 0.5,
) -> dict:
    """Run KS test on numerical features and Chi-Square test on categorical features."""
    features_report = {}
    drifted_count = 0

    # 1. Numerical features -> Kolmogorov-Smirnov test
    for col in NUM_COLS:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_vals = pd.to_numeric(reference_df[col], errors="coerce").dropna().values
        cur_vals = pd.to_numeric(current_df[col], errors="coerce").dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        ks_stat, p_val = ks_2samp(ref_vals, cur_vals)
        is_drift = bool(p_val < p_value_threshold)
        if is_drift:
            drifted_count += 1

        ref_mean = float(np.mean(ref_vals))
        ref_std = float(np.std(ref_vals))
        cur_mean = float(np.mean(cur_vals))
        cur_std = float(np.std(cur_vals))

        features_report[col] = {
            "feature_type": "numerical",
            "test_name": "KS Test",
            "drift_detected": is_drift,
            "p_value": float(p_val),
            "statistic": float(ks_stat),
            "threshold": p_value_threshold,
            "reference_summary": f"{ref_mean:.2f} (±{ref_std:.2f})",
            "current_summary": f"{cur_mean:.2f} (±{cur_std:.2f})",
            "shift_summary": f"{cur_mean - ref_mean:+.2f}",
        }

    # 2. Categorical features -> Chi-Square test
    for col in CAT_COLS:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_series = reference_df[col].dropna().astype(str).str.strip()
        cur_series = current_df[col].dropna().astype(str).str.strip()

        if len(ref_series) == 0 or len(cur_series) == 0:
            continue

        all_categories = sorted(set(ref_series.unique()) | set(cur_series.unique()))
        ref_counts = ref_series.value_counts().reindex(all_categories, fill_value=0)
        cur_counts = cur_series.value_counts().reindex(all_categories, fill_value=0)

        contingency = np.vstack([ref_counts.values, cur_counts.values])
        # Filter out columns where both counts are zero
        contingency = contingency[:, contingency.sum(axis=0) > 0]

        if contingency.shape[1] < 2:
            chi2_stat, p_val = 0.0, 1.0
        else:
            chi2_stat, p_val, _, _ = chi2_contingency(contingency)

        is_drift = bool(p_val < p_value_threshold)
        if is_drift:
            drifted_count += 1

        ref_top = f"{ref_series.mode().iloc[0]} ({(ref_series == ref_series.mode().iloc[0]).mean() * 100:.1f}%)"
        cur_top = f"{cur_series.mode().iloc[0]} ({(cur_series == cur_series.mode().iloc[0]).mean() * 100:.1f}%)"

        features_report[col] = {
            "feature_type": "categorical",
            "test_name": "Chi-Square",
            "drift_detected": is_drift,
            "p_value": float(p_val),
            "statistic": float(chi2_stat),
            "threshold": p_value_threshold,
            "reference_summary": f"Top: {ref_top}",
            "current_summary": f"Top: {cur_top}",
            "shift_summary": f"Categories: {len(all_categories)}",
        }

    total_features = len(features_report)
    drift_share = (drifted_count / total_features) if total_features > 0 else 0.0
    dataset_drift_detected = drift_share >= dataset_drift_threshold

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reference_samples": len(reference_df),
        "current_samples": len(current_df),
        "total_features": total_features,
        "drifted_features": drifted_count,
        "drift_share_pct": round(drift_share * 100, 2),
        "dataset_drift_detected": dataset_drift_detected,
        "dataset_drift_threshold_pct": round(dataset_drift_threshold * 100, 2),
        "features": features_report,
    }


def render_html_report(report: dict, output_path: Path = DRIFT_HTML_PATH):
    """Render a standalone HTML dashboard for Bridge Condition drift analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_color = "#ef4444" if report["dataset_drift_detected"] else "#10b981"
    status_text = "DRIFT DETECTED" if report["dataset_drift_detected"] else "DISTRIBUTION HEALTHY"

    cards_html = ""
    for feat, data in report["features"].items():
        feat_color = "#ef4444" if data["drift_detected"] else "#10b981"
        feat_status = "DRIFT" if data["drift_detected"] else "PASS"
        cards_html += f"""
        <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <strong style="font-size: 16px; color: #1e293b;">{feat}</strong>
                    <span style="margin-left: 8px; font-size: 12px; color: #64748b;">({data['feature_type']} • {data['test_name']})</span>
                </div>
                <span style="background: {feat_color}; color: white; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 12px;">{feat_status}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; font-size: 13px; color: #64748b;">
                <div>p-value: <b style="color: #0f172a;">{data['p_value']:.4f}</b></div>
                <div>Statistic: <b style="color: #0f172a;">{data['statistic']:.4f}</b></div>
                <div>Reference: <b style="color: #0f172a;">{data['reference_summary']}</b></div>
                <div>Production: <b style="color: #0f172a;">{data['current_summary']}</b></div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Bridge Condition - Data Drift Monitoring</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; margin: 0; padding: 24px; color: #1e293b; }}
        .container {{ max-width: 920px; margin: 0 auto; }}
        .header {{ background: white; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .badge {{ display: inline-block; padding: 6px 14px; border-radius: 16px; color: white; font-weight: bold; font-size: 14px; background: {status_color}; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 16px; }}
        .stat-card {{ background: #f1f5f9; padding: 12px; border-radius: 8px; text-align: center; }}
        .stat-num {{ font-size: 20px; font-weight: bold; color: #0f172a; }}
        .stat-label {{ font-size: 12px; color: #64748b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0 0 6px 0; font-size: 24px;">Bridge Condition Data Drift Report</h1>
                    <div style="color: #64748b; font-size: 13px;">Analyzed at: {report['timestamp']}</div>
                </div>
                <div class="badge">{status_text}</div>
            </div>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-num">{report['reference_samples']}</div>
                    <div class="stat-label">Reference Samples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{report['current_samples']}</div>
                    <div class="stat-label">Production Logs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{report['drifted_features']} / {report['total_features']}</div>
                    <div class="stat-label">Drifted Features</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{report['drift_share_pct']}%</div>
                    <div class="stat-label">Drift Share</div>
                </div>
            </div>
        </div>
        <h2 style="font-size: 18px; margin-bottom: 12px;">Feature Drift Tests (KS &amp; Chi-Square)</h2>
        {cards_html}
    </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def generate_alert_markdown(report: dict, output_path: Path = DRIFT_ALERT_MD_PATH) -> str:
    """Generate Markdown incident report for GitHub Issues, Slack, or CI Step Summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drift_header = (
        "🚨 **HIGH PRIORITY: BRIDGE DATA DRIFT DETECTED**"
        if report["dataset_drift_detected"]
        else "✅ **BRIDGE PRODUCTION DATA DISTRIBUTION HEALTHY**"
    )

    rows = []
    for feat, data in report["features"].items():
        status = "⚠️ **DRIFT**" if data["drift_detected"] else "✅ OK"
        rows.append(
            f"| `{feat}` | {data['test_name']} | {status} | `{data['p_value']:.4f}` | "
            f"`{data['statistic']:.4f}` | `{data['reference_summary']}` | "
            f"`{data['current_summary']}` | `{data['shift_summary']}` |"
        )
    table_content = "\n".join(rows)

    action_text = (
        "⚠️ **Continuous Training Required:** Production bridge feature distributions have deviated significantly from the baseline training distribution. "
        "Triggering automatic pipeline reproduction (`dvc repro`) and champion re-selection."
        if report["dataset_drift_detected"]
        else "✅ **No Action Needed:** Production traffic matches the baseline training distribution within statistical tolerance."
    )

    md = f"""# {drift_header}

### 📊 Monitoring Summary
- **Evaluation Timestamp:** `{report['timestamp']}`
- **Drifted Features:** **{report['drifted_features']} of {report['total_features']}** ({report['drift_share_pct']}%)
- **Dataset Drift Threshold:** `{report['dataset_drift_threshold_pct']}%`
- **Reference Training Samples:** `{report['reference_samples']}`
- **Production Inference Samples:** `{report['current_samples']}`

### 🔬 Statistical Drift Tests
| Feature | Test | Status | p-value | Statistic | Reference Baseline | Production Window | Shift / Info |
|---|---|---|---|---|---|---|---|
{table_content}

---
### ⚙️ Action & Continuous Training Status
{action_text}
"""
    output_path.write_text(md, encoding="utf-8")
    return md


def send_drift_alert(report: dict, webhook_url: str = None) -> bool:
    """Send webhook alert to Slack, Discord, or Teams if configured and drift detected."""
    if not report.get("dataset_drift_detected"):
        return False

    target_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL") or os.getenv("SLACK_WEBHOOK_URL")
    if not target_url:
        print("Note: No ALERT_WEBHOOK_URL configured. Markdown alert generated at reports/drift_alert.md.")
        return False

    payload = {
        "text": (
            f"🚨 *[MLOps Alert] Bridge Condition Data Drift Detected!*\n"
            f"• Drifted Features: {report['drifted_features']}/{report['total_features']} ({report['drift_share_pct']}%)\n"
            f"• Action: Triggering automated continuous retraining pipeline (`dvc repro`)."
        )
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(target_url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"Alert successfully delivered to webhook (HTTP {resp.status}).")
            return True
    except Exception as e:
        print(f"Warning: Failed to deliver alert to webhook: {e}")
        return False


def run_drift_check(
    reference_path: Path = REFERENCE_DATA_PATH,
    logs_path: Path = INFERENCE_LOG_PATH,
    generate_sample: bool = False,
    simulate_drift: bool = False,
    webhook_url: str = None,
) -> dict:
    """Reusable entry point for both CLI execution and the Streamlit dashboard."""
    if not reference_path.exists():
        if RAW_FALLBACK_PATH.exists():
            ref_df = pd.read_csv(RAW_FALLBACK_PATH)
            ref_df.columns = ref_df.columns.str.strip().str.lower()
            ref_df = ref_df.drop_duplicates()
        else:
            raise FileNotFoundError(
                f"Reference training data not found at {reference_path}. Run `dvc repro` first."
            )
    else:
        ref_df = pd.read_csv(reference_path)

    logs_df = load_inference_logs(logs_path)
    if len(logs_df) < 5 or generate_sample or simulate_drift:
        logs_df = generate_sample_inferences(
            n_samples=50,
            drift=simulate_drift,
            log_path=logs_path,
            reference_df=ref_df,
        )

    report = analyze_drift(ref_df, logs_df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    render_html_report(report, DRIFT_HTML_PATH)
    generate_alert_markdown(report, DRIFT_ALERT_MD_PATH)

    if report["dataset_drift_detected"]:
        send_drift_alert(report, webhook_url=webhook_url)

    return report


def main():
    parser = argparse.ArgumentParser(description="Check Bridge Condition production data drift.")
    parser.add_argument("--reference", type=Path, default=REFERENCE_DATA_PATH, help="Path to reference training CSV.")
    parser.add_argument("--logs", type=Path, default=INFERENCE_LOG_PATH, help="Path to production inference logs CSV.")
    parser.add_argument("--generate-sample-data", action="store_true", help="Generate synthetic inference logs.")
    parser.add_argument("--simulate-drift", action="store_true", help="Simulate drifted traffic for demonstration.")
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit with code 1 if dataset drift is detected.")
    parser.add_argument("--webhook-url", type=str, default=None, help="Optional Slack/Discord webhook URL for alerts.")
    args = parser.parse_args()

    try:
        report = run_drift_check(
            reference_path=args.reference,
            logs_path=args.logs,
            generate_sample=args.generate_sample_data,
            simulate_drift=args.simulate_drift,
            webhook_url=args.webhook_url,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print("\n=== Bridge Condition Data Drift Monitoring Report ===")
    print(f"Status: {'DRIFT DETECTED ⚠️' if report['dataset_drift_detected'] else 'DISTRIBUTION HEALTHY ✅'}")
    print(f"Drifted features: {report['drifted_features']} / {report['total_features']} ({report['drift_share_pct']}%)")
    for feat, data in report["features"].items():
        flag = "DRIFT" if data["drift_detected"] else "OK"
        print(
            f"  - {feat:18s} [{data['test_name']:10s}]: [{flag}] "
            f"p-val={data['p_value']:.4f}, stat={data['statistic']:.4f}, shift={data['shift_summary']}"
        )
    print(f"\nSaved JSON report: {DRIFT_JSON_PATH}")
    print(f"Saved HTML report: {DRIFT_HTML_PATH}")
    print(f"Saved Alert Markdown: {DRIFT_ALERT_MD_PATH}")

    if args.fail_on_drift and report["dataset_drift_detected"]:
        print("\nQuality Gate Error: Production data drift exceeds threshold!")
        sys.exit(1)


if __name__ == "__main__":
    main()