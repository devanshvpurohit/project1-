import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Linear Regression from Scratch", layout="centered")

st.title("📈 Linear Regression from Scratch (NumPy Only)")
st.markdown("""
Upload a CSV file with independent features and one target column (at the end).
This app calculates the regression coefficients, builds the equation, and reports accuracy using the R² score.
""")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🧾 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    if df.shape[1] < 2:
        st.error("Dataset must have at least one feature column and one target column.")
    else:
        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        feature_names = df.columns[:-1]
        target_name = df.columns[-1]

        # Add bias (intercept) term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Calculate weights using normal equation
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        intercept = theta[0]
        coefficients = theta[1:]

        # Predictions and R² score
        y_pred = X_b.dot(theta)
        r2_score = compute_r2(y, y_pred)

        # Display coefficients
        st.subheader("📊 Regression Coefficients")
        coef_df = pd.DataFrame({
            "Feature": ["Intercept"] + list(feature_names),
            "Coefficient": [intercept] + list(coefficients)
        })
        st.dataframe(coef_df, use_container_width=True)

        # Display equation
        st.subheader("🧮 Regression Equation")
        equation = f"{target_name} = "
        equation += " + ".join([f"{coef:.2f} × {name}" for coef, name in zip(coefficients, feature_names)])
        equation += f" + ({intercept:.2f})"
        st.code(equation)

        # Display accuracy
        st.subheader("✅ Model Accuracy")
        st.metric(label="R² Score", value=f"{r2_score * 100:.2f}%")

        if r2_score >= 0.9823:
            st.success("🎉 Excellent! Your model has reached or exceeded 98.23% accuracy.")
        else:
            st.warning("📉 Model accuracy is below 98.23%. You might want to explore feature selection or data quality.")
