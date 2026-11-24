from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

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
        "selected champion pipeline."
    )

    if not MODEL_PATH.exists():
        st.error(
            f"Champion model not found at `{MODEL_PATH}`. "
            "Run `dvc repro` or `python main.py` first to train and select a model."
        )
        st.stop()

    model = load_model(MODEL_PATH)
    selection = load_json(SELECTION_PATH)
    metrics = load_json(METRICS_PATH)
    details = load_json(DETAILS_PATH)
    feature_meta = load_feature_metadata(X_TRAIN_PATH)

    tab_predict, tab_mlops = st.tabs(
        ["🔍 Single & Batch Prediction", "📊 Champion Model & MLOps Metrics"]
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

                # Display confidence probabilities if supported by the champion model
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_df)[0]
                    prob_poor = float(probs[0]) * 100
                    prob_good = float(probs[1]) * 100

                    st.markdown("#### Prediction Confidence")
                    c1, c2 = st.columns(2)
                    c1.metric("Good Condition (1)", f"{prob_good:.1f}%")
                    c2.metric("Poor Condition (0)", f"{prob_poor:.1f}%")
                    st.progress(int(round(prob_good)))

                with st.expander("View Input Payload"):
                    st.dataframe(input_df, use_container_width=True)
            else:
                st.info("Adjust the bridge features on the left and click **Predict Bridge Condition**.")

        # Batch CSV Upload section
        st.divider()
        st.subheader("Batch CSV Prediction")
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
                batch_df["predicted_condition_code"] = preds
                batch_df["predicted_condition"] = [
                    LABEL_MAP.get(int(p), str(p)) for p in preds
                ]
                if hasattr(model, "predict_proba"):
                    batch_df["good_probability"] = model.predict_proba(X_batch)[:, 1].round(4)

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


if __name__ == "__main__":
    main()
