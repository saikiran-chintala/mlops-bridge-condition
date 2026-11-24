from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from monitor_drift import (
        DRIFT_ALERT_MD_PATH,
        DRIFT_HTML_PATH,
        DRIFT_JSON_PATH,
        INFERENCE_LOG_PATH,
        REFERENCE_DATA_PATH,
        load_inference_logs,
        log_batch_inferences,
        log_inference,
        run_drift_check,
    )
except ImportError:
    from src.monitor_drift import (
        DRIFT_ALERT_MD_PATH,
        DRIFT_HTML_PATH,
        DRIFT_JSON_PATH,
        INFERENCE_LOG_PATH,
        REFERENCE_DATA_PATH,
        load_inference_logs,
        log_batch_inferences,
        log_inference,
        run_drift_check,
    )

MODEL_PATH = Path("models/model.pkl")
SELECTION_PATH = Path("reports/model_selection.json")
METRICS_PATH = Path("reports/metrics.json")
DETAILS_PATH = Path("reports/evaluation_details.json")
COMPARISON_CSV_PATH = Path("reports/model_comparison.csv")
X_TRAIN_PATH = Path("data/processed/X_train.csv")

NUM_COLS = ["age_of_bridge", "traffic_volume"]
CAT_COLS = ["material_type", "maintenance_level"]
LABEL_MAP = {0: "Poor", 1: "Good"}


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


@st.cache_data
def load_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_feature_metadata(path: Path):
    """Derives min/max/default values and valid categories from X_train.csv."""
    if not path.exists():
        return {
            "age_of_bridge": {"min": 0.0, "max": 150.0, "default": 35.0},
            "traffic_volume": {"min": 0.0, "max": 100000.0, "default": 15000.0},
            "material_type": ["Concrete", "Steel", "Wood", "Composite"],
            "maintenance_level": ["Low", "Medium", "High"],
        }

    df = pd.read_csv(path)
    meta = {}
    for col in NUM_COLS:
        meta[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "default": float(df[col].median()),
        }
    for col in CAT_COLS:
        meta[col] = sorted(df[col].dropna().astype(str).unique().tolist())
    return meta


def main():
    st.set_page_config(
        page_title="Bridge Condition Predictor",
        page_icon="🌉",
        layout="wide",
    )

    st.title("🌉 Bridge Condition Prediction & MLOps Dashboard")
    st.markdown(
        "Predict structural condition (**Good** vs. **Poor**) using the automatically "
        "selected champion pipeline and monitor live production feature drift."
    )

    if not MODEL_PATH.exists():
        st.error(
            f"Champion model not found at `{MODEL_PATH}`. "
            "Run `dvc repro` or `python model_building.py` first to train and select a model."
        )
        st.stop()

    model = load_model(MODEL_PATH)
    selection = load_json(SELECTION_PATH)
    metrics = load_json(METRICS_PATH)
    details = load_json(DETAILS_PATH)
    feature_meta = load_feature_metadata(X_TRAIN_PATH)

    tab_predict, tab_mlops, tab_drift = st.tabs(
        [
            "🔍 Single & Batch Prediction",
            "📊 Champion Model & MLOps Metrics",
            "📡 Data Drift Monitoring",
        ]
    )

    # =========================================================================
    # TAB 1: PREDICTION INTERFACE
    # =========================================================================
    with tab_predict:
        col_form, col_result = st.columns([1.2, 1], gap="large")

        with col_form:
            st.subheader("Bridge Parameters")
            with st.form("bridge_input_form"):
                age = st.number_input(
                    "Age of Bridge (years)",
                    min_value=0.0,
                    max_value=max(200.0, feature_meta["age_of_bridge"]["max"] * 1.5),
                    value=feature_meta["age_of_bridge"]["default"],
                    step=1.0,
                )

                traffic = st.number_input(
                    "Daily Traffic Volume (vehicles/day)",
                    min_value=0.0,
                    max_value=max(500000.0, feature_meta["traffic_volume"]["max"] * 1.5),
                    value=feature_meta["traffic_volume"]["default"],
                    step=100.0,
                )

                material = st.selectbox(
                    "Material Type",
                    options=feature_meta["material_type"],
                )

                maintenance = st.selectbox(
                    "Maintenance Level",
                    options=feature_meta["maintenance_level"],
                )

                log_single = st.checkbox(
                    f"Log prediction to monitoring store (`{INFERENCE_LOG_PATH.as_posix()}`)",
                    value=True,
                )

                submitted = st.form_submit_button(
                    "Predict Bridge Condition", use_container_width=True
                )

        with col_result:
            st.subheader("Prediction Result")
            if submitted:
                input_df = pd.DataFrame(
                    [
                        {
                            "age_of_bridge": age,
                            "traffic_volume": traffic,
                            "material_type": material,
                            "maintenance_level": maintenance,
                        }
                    ]
                )

                pred_class = int(model.predict(input_df)[0])
                pred_label = LABEL_MAP.get(pred_class, str(pred_class))

                if pred_class == 1:
                    st.success(f"### Predicted Condition: **{pred_label}**")
                else:
                    st.error(f"### Predicted Condition: **{pred_label}**")

                prob_dict = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_df)[0]
                    prob_poor = float(probs[0])
                    prob_good = float(probs[1])
                    prob_dict = {"Poor": prob_poor, "Good": prob_good}

                    st.markdown("#### Prediction Confidence")
                    c1, c2 = st.columns(2)
                    c1.metric("Good Condition (1)", f"{prob_good * 100:.1f}%")
                    c2.metric("Poor Condition (0)", f"{prob_poor * 100:.1f}%")
                    st.progress(int(round(prob_good * 100)))

                if log_single:
                    log_inference(
                        input_df=input_df,
                        prediction_code=pred_class,
                        prediction_label=pred_label,
                        probabilities=prob_dict,
                    )
                    st.caption(f"✅ Logged to `{INFERENCE_LOG_PATH.as_posix()}` for drift tracking.")

                with st.expander("View Input Payload"):
                    st.dataframe(input_df, use_container_width=True)
            else:
                st.info("Adjust the bridge features on the left and click **Predict Bridge Condition**.")

        # Batch CSV Upload section
        st.divider()
        st.subheader("Batch CSV Prediction")
        log_batch = st.checkbox(
            f"Log batch predictions to `{INFERENCE_LOG_PATH.as_posix()}`",
            value=True,
        )
        uploaded_file = st.file_uploader(
            "Upload a CSV containing `age_of_bridge`, `traffic_volume`, `material_type`, and `maintenance_level`",
            type=["csv"],
        )
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            batch_df.columns = batch_df.columns.str.strip().str.lower()
            missing = sorted(set(NUM_COLS + CAT_COLS) - set(batch_df.columns))

            if missing:
                st.error(f"Uploaded CSV is missing required columns: {missing}")
            else:
                X_batch = batch_df[NUM_COLS + CAT_COLS].copy()
                for c in CAT_COLS:
                    X_batch[c] = X_batch[c].astype(str).str.strip()

                preds = model.predict(X_batch)
                pred_labels = [LABEL_MAP.get(int(p), str(p)) for p in preds]
                batch_df["predicted_condition_code"] = preds
                batch_df["predicted_condition"] = pred_labels

                probs = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_batch)
                    batch_df["poor_probability"] = probs[:, 0].round(4)
                    batch_df["good_probability"] = probs[:, 1].round(4)

                # Log batch predictions once per uploaded file when logging is enabled
                if log_batch:
                    file_sig = f"{uploaded_file.name}_{uploaded_file.size}"
                    if st.session_state.get("last_logged_batch") != file_sig:
                        log_batch_inferences(
                            batch_df=X_batch,
                            pred_codes=preds,
                            pred_labels=pred_labels,
                            probs=probs,
                        )
                        st.session_state["last_logged_batch"] = file_sig
                        st.toast(f"Logged {len(X_batch)} batch predictions to drift monitor.")

                st.dataframe(batch_df, use_container_width=True)
                st.download_button(
                    "Download Predictions CSV",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="bridge_predictions.csv",
                    mime="text/csv",
                )

    # =========================================================================
    # TAB 2: MLOPS & MODEL COMPARISON DASHBOARD
    # =========================================================================
    with tab_mlops:
        if selection and metrics:
            champ = selection["champion"]
            st.subheader(f"🏆 Active Champion: {champ['display_name']}")
            st.caption(champ["explanation"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            m2.metric("CV Validation Acc", f"{metrics['champion_validation_accuracy_pct']:.2f}%")
            m3.metric("Test F1 Score", f"{metrics['f1']:.4f}")
            m4.metric("Predict Latency", f"{metrics['champion_prediction_time_ms']:.3f} ms")

        if COMPARISON_CSV_PATH.exists():
            st.subheader("Candidate Model Comparison (Training & CV Selection)")
            comp_df = pd.read_csv(COMPARISON_CSV_PATH)
            st.dataframe(comp_df, use_container_width=True)

        if details and "champion_confusion_matrix" in details:
            st.subheader("Champion Confusion Matrix (Untouched Test Set)")
            cm = details["champion_confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: Poor (0)", "Actual: Good (1)"],
                columns=["Predicted: Poor (0)", "Predicted: Good (1)"],
            )
            st.table(cm_df)

    # =========================================================================
    # TAB 3: PRODUCTION DATA DRIFT MONITORING
    # =========================================================================
    with tab_drift:
        st.subheader("Production Inference Drift Monitor")
        st.markdown(
            f"Compares logged inference requests (`{INFERENCE_LOG_PATH.as_posix()}`) "
            f"against baseline training distributions (`{REFERENCE_DATA_PATH.as_posix()}`) using "
            "**Kolmogorov-Smirnov** (numerical) and **Chi-Square** (categorical) tests."
        )

        with st.expander("Optional Alert Webhook Configuration"):
            webhook_url = st.text_input(
                "Slack / Discord / Teams Webhook URL (leave blank to use environment variable)",
                value="",
                type="password",
            ).strip() or None

        btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
        with btn_c1:
            run_live = st.button("🔄 Run Drift Check on Live Logs", use_container_width=True)
        with btn_c2:
            run_healthy_sim = st.button("✅ Simulate Healthy Traffic (50)", use_container_width=True)
        with btn_c3:
            run_drift_sim = st.button("🚨 Simulate Drifted Traffic (50)", use_container_width=True)
        with btn_c4:
            clear_logs = st.button("🗑️ Clear Inference Logs", use_container_width=True)

        if clear_logs:
            if INFERENCE_LOG_PATH.exists():
                INFERENCE_LOG_PATH.unlink()
            st.session_state.pop("last_logged_batch", None)
            st.success(f"Cleared `{INFERENCE_LOG_PATH.as_posix()}`.")

        report = None
        try:
            if run_live:
                report = run_drift_check(
                    reference_path=REFERENCE_DATA_PATH,
                    logs_path=INFERENCE_LOG_PATH,
                    webhook_url=webhook_url,
                )
            elif run_healthy_sim:
                report = run_drift_check(
                    reference_path=REFERENCE_DATA_PATH,
                    logs_path=INFERENCE_LOG_PATH,
                    generate_sample=True,
                    simulate_drift=False,
                    webhook_url=webhook_url,
                )
            elif run_drift_sim:
                report = run_drift_check(
                    reference_path=REFERENCE_DATA_PATH,
                    logs_path=INFERENCE_LOG_PATH,
                    generate_sample=True,
                    simulate_drift=True,
                    webhook_url=webhook_url,
                )
            elif DRIFT_JSON_PATH.exists():
                with open(DRIFT_JSON_PATH, "r", encoding="utf-8") as f:
                    report = json.load(f)
        except FileNotFoundError as exc:
            st.error(str(exc))

        if report:
            if report["dataset_drift_detected"]:
                st.error(
                    f"🚨 **DATA DRIFT DETECTED** — {report['drifted_features']} of "
                    f"{report['total_features']} features drifted ({report['drift_share_pct']}% >= "
                    f"{report['dataset_drift_threshold_pct']}% threshold). Retraining (`dvc repro`) recommended."
                )
            else:
                st.success(
                    f"✅ **DISTRIBUTION HEALTHY** — {report['drifted_features']} of "
                    f"{report['total_features']} features drifted ({report['drift_share_pct']}% < "
                    f"{report['dataset_drift_threshold_pct']}% threshold)."
                )

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Reference Samples", report["reference_samples"])
            d2.metric("Production Logs", report["current_samples"])
            d3.metric("Drifted Features", f"{report['drifted_features']} / {report['total_features']}")
            d4.metric("Drift Share", f"{report['drift_share_pct']}%")

            st.markdown("#### Feature-Level Statistical Test Results")
            feat_rows = []
            for feat, info in report["features"].items():
                feat_rows.append(
                    {
                        "Feature": feat,
                        "Type": info["feature_type"].capitalize(),
                        "Test": info["test_name"],
                        "Status": "DRIFT" if info["drift_detected"] else "PASS",
                        "p-value": round(info["p_value"], 4),
                        "Statistic": round(info["statistic"], 4),
                        "Reference Baseline": info["reference_summary"],
                        "Production Window": info["current_summary"],
                        "Shift / Info": info["shift_summary"],
                    }
                )
            st.dataframe(pd.DataFrame(feat_rows), use_container_width=True)
            st.caption(f"Last evaluated at: `{report['timestamp']}`")

            # Download buttons for generated drift reports
            dl_c1, dl_c2, dl_c3 = st.columns(3)
            with dl_c1:
                st.download_button(
                    "⬇️ Download Drift JSON",
                    data=json.dumps(report, indent=4).encode("utf-8"),
                    file_name="drift_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with dl_c2:
                if DRIFT_HTML_PATH.exists():
                    st.download_button(
                        "⬇️ Download Drift HTML Dashboard",
                        data=DRIFT_HTML_PATH.read_bytes(),
                        file_name="drift_report.html",
                        mime="text/html",
                        use_container_width=True,
                    )
            with dl_c3:
                if DRIFT_ALERT_MD_PATH.exists():
                    st.download_button(
                        "⬇️ Download Drift Alert Markdown",
                        data=DRIFT_ALERT_MD_PATH.read_bytes(),
                        file_name="drift_alert.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

            with st.expander("Preview Standalone HTML Drift Report"):
                if DRIFT_HTML_PATH.exists():
                    components.html(
                        DRIFT_HTML_PATH.read_text(encoding="utf-8"),
                        height=620,
                        scrolling=True,
                    )

            with st.expander("Preview Incident Alert Markdown (`reports/drift_alert.md`)"):
                if DRIFT_ALERT_MD_PATH.exists():
                    st.markdown(DRIFT_ALERT_MD_PATH.read_text(encoding="utf-8"))
        else:
            st.info(
                "No drift report generated yet. Submit predictions in Tab 1 or click "
                "**Run Drift Check on Live Logs** above."
            )

        with st.expander("Recent Logged Production Predictions"):
            logs_preview = load_inference_logs(INFERENCE_LOG_PATH)
            if logs_preview.empty:
                st.write("No inference logs recorded yet.")
            else:
                st.write(f"Total logged predictions: **{len(logs_preview)}**")
                st.dataframe(logs_preview.tail(25), use_container_width=True)


if __name__ == "__main__":
    main()