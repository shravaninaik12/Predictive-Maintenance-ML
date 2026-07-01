import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📌 Project Information")
st.sidebar.write("**Project:** Predictive Maintenance using XGBoost")
st.sidebar.write("**Developer:** Shravani Naik")
st.sidebar.write("**Dataset:** NASA CMAPSS FD001")
st.sidebar.write("**Algorithm:** XGBoost Regressor")

st.sidebar.markdown("---")

st.sidebar.info("""
### Instructions

1. Upload a CSV file.

2. The CSV must contain all input features
used during training.

3. Click upload.

4. View predicted Remaining Useful Life (RUL).

5. Download the prediction results.
""")

# -------------------------------
# Title
# -------------------------------
st.title("🔧 Predictive Maintenance using Machine Learning")

st.write("""
This application predicts the **Remaining Useful Life (RUL)** of aircraft engines
using an **XGBoost Regression Model** trained on the **NASA CMAPSS FD001 dataset**.
""")

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("predictive_model.pkl")

# -------------------------------
# Display Model Metrics
# -------------------------------

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

# Replace these with your actual values
MAE = 11.32
RMSE = 16.85
R2 = 0.94

col1.metric("MAE", f"{MAE:.2f}")
col2.metric("RMSE", f"{RMSE:.2f}")
col3.metric("R² Score", f"{R2:.2f}")

st.markdown("---")

# -------------------------------
# Upload CSV
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload Engine Sensor CSV",
    type=["csv"]
)

# -------------------------------
# Prediction
# -------------------------------

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(data)

    predictions = model.predict(data)

    data["Predicted_RUL"] = predictions

    # -------------------------------
    # Maintenance Status
    # -------------------------------

    def maintenance_status(rul):

        if rul > 100:
            return "🟢 Healthy"

        elif rul >= 50:
            return "🟡 Maintenance Soon"

        else:
            return "🔴 Critical"

    data["Status"] = data["Predicted_RUL"].apply(maintenance_status)

    # -------------------------------
    # Prediction Table
    # -------------------------------

    st.subheader("Prediction Results")

    st.dataframe(data)

    # -------------------------------
    # Bar Chart
    # -------------------------------

    st.subheader("📈 Predicted Remaining Useful Life")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(range(len(data)), data["Predicted_RUL"])

    ax.set_xlabel("Engine Sample")

    ax.set_ylabel("Predicted RUL")

    ax.set_title("Remaining Useful Life Prediction")

    st.pyplot(fig)

    # -------------------------------
    # Status Count
    # -------------------------------

    st.subheader("Maintenance Summary")

    st.write(data["Status"].value_counts())

    # -------------------------------
    # Download Button
    # -------------------------------

    csv = data.to_csv(index=False)

    st.download_button(
        label="⬇ Download Prediction Results",
        data=csv,
        file_name="results.csv",
        mime="text/csv"
    )