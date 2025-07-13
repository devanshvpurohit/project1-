import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Dynamic Linear Regression Dashboard", layout="centered")
st.title("📊 Linear Regression Dashboard (NumPy Only)")

st.markdown("""
Upload a CSV with at least **6 feature columns** and **1 target column (last)**.
This app will:
- Train a linear regression model from scratch using NumPy
- Display the learned equation
- Let you input new values and get predictions based on it
""")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Convert all to numeric and drop bad rows
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    if df.shape[1] < 7:
        st.error("❌ Please upload a CSV with at least 6 feature columns and 1 target column (min 7 columns total).")
    else:
        st.success("✅ Data loaded successfully!")

        st.subheader("🧾 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        feature_names = df.columns[:-1]
        target_name = df.columns[-1]

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Add intercept
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Fit model: Normal equation
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        intercept = theta[0]
        coefficients = theta[1:]

        # Predictions and R²
        y_pred = X_b.dot(theta)
        r2_score = compute_r2(y, y_pred)

        # Display Equation
        st.subheader("📐 Regression Equation")
        equation = f"{target_name} = " + " + ".join([
            f"{coef:.2f} × {name}" for coef, name in zip(coefficients, feature_names)
        ]) + f" + ({intercept:.2f})"
        st.code(equation)

        st.subheader("📈 Model Accuracy")
        st.metric("R² Score", f"{r2_score * 100:.2f}%")

        # Prediction interface
        st.subheader("🎯 Predict New Value")
        st.markdown("Enter feature values below (if unknown, leave as 0):")

        input_values = []
        for feature in feature_names:
            val = st.number_input(f"{feature}", value=0.0, format="%.2f")
            input_values.append(val)

        input_array = np.array([1] + input_values)
        prediction = input_array.dot(theta)

        st.success(f"🔮 Predicted {target_name}: **{prediction:.2f}**")
