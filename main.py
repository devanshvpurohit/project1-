import streamlit as st
import numpy as np
import pandas as pd

# Set layout
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Predictor")

# Load local CSV from GitHub repo (same folder)
@st.cache_data
def load_data():
    return pd.read_csv("Student_Performance.csv")

df = load_data()

# Convert to numeric, drop NaNs
df = df.apply(pd.to_numeric, errors='coerce').dropna()

if df.shape[1] < 7:
    st.error("CSV must have at least 6 feature columns and 1 target column (7 columns total).")
    st.stop()

st.success("✅ Model trained on hardcoded dataset!")

# Separate features and target
feature_names = df.columns[:-1]
target_name = df.columns[-1]

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Add bias term
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Train linear regression using normal equation
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

intercept = theta[0]
coefficients = theta[1:]

# Calculate R²
def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

y_pred = X_b.dot(theta)
r2_score = compute_r2(y, y_pred)

# Show regression equation
st.subheader("📐 Regression Equation")
equation = f"{target_name} = " + " + ".join([
    f"{coef:.2f} × {name}" for coef, name in zip(coefficients, feature_names)
]) + f" + ({intercept:.2f})"
st.code(equation)

# Show accuracy
st.subheader("📈 Model Accuracy")
st.metric("R² Score", f"{r2_score * 100:.2f}%")

# Prediction interface
st.subheader("🧮 Predict New Performance")
input_values = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0, format="%.2f")
    input_values.append(val)

input_array = np.array([1] + input_values)
prediction = input_array.dot(theta)

st.success(f"🎯 Predicted {target_name}: **{prediction:.2f}**")
