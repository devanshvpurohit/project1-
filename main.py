import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai

# Configure Gemini API (hardcoded key)
genai.configure(api_key="AIzaSyAcfTRSVuhJTPsw4uxChpNWRUfTnxniU_k")
model = genai.GenerativeModel("gemini-pro")

# Streamlit page settings
st.set_page_config(page_title="🎓 AI Student Performance Predictor", layout="centered")
st.title("🎓 AI-Powered Student Performance Predictor")
st.markdown("This app trains a model from a fixed CSV on GitHub, predicts performance, visualizes data, and provides AI-powered recommendations.")

# Load data from GitHub
CSV_URL = "https://raw.githubusercontent.com/devanshvpurohit/project1-/main/Student_Performance.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df

df = load_data()
st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# Feature and target extraction
feature_names = df.columns[:-1].tolist()
target_name = df.columns[-1]

# Train linear regression model using normal equation
def train_model(df):
    X = df[feature_names].values
    y = df[target_name].values
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta

theta = train_model(df)
intercept = theta[0]
coefficients = theta[1:]

# Display the equation and coefficients
st.subheader("🔬 Trained Model Coefficients")
eq = f"{target_name} = " + " + ".join([f"{c:.2f}×{n}" for c, n in zip(coefficients, feature_names)]) + f" + ({intercept:.2f})"
st.code(eq)

coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
st.dataframe(coef_df)

# User input form for prediction
inputs = []
with st.form("input_form"):
    st.subheader("📥 Input Student Data")
    for feature in feature_names:
        vmin, vmax = float(df[feature].min()), float(df[feature].max())
        val = st.slider(f"{feature}", min_value=vmin, max_value=vmax, value=(vmin+vmax)/2)
        inputs.append(val)
    submitted = st.form_submit_button("🔮 Predict")

# Prediction function
def predict(inputs):
    return float(np.dot(coefficients, inputs) + intercept)

# Gemini prompt construction
def prompt_gen(inputs, pred):
    txt = "Student indicators:\n" + "\n".join([f"- {n}: {v}" for n, v in zip(feature_names, inputs)])
    txt += f"\n\nPredicted score: {pred:.2f}.\nGive bullet‑point recommendations to improve academic performance."
    return txt

def get_ai_tips(inputs, pred):
    try:
        return model.generate_content(prompt_gen(inputs, pred)).text
    except Exception as e:
        return f"⚠️ Error: {e}"

# When user submits input
if submitted:
    score = predict(inputs)
    st.subheader("📊 Prediction Result")
    st.metric("Predicted Score", f"{score:.2f}")
    grade = ("A" if score>=90 else "B" if score>=80 else "C" if score>=70 else "D" if score>=60 else "F")
    st.write(f"**Estimated Grade:** `{grade}`")
    if score < 60:
        st.error("🚨 This student is at risk—early intervention recommended.")
    else:
        st.success("✅ Performance prediction is satisfactory.")

    # Bar chart of inputs
    fig1, ax1 = plt.subplots(figsize=(8,4))
    bars = ax1.bar(feature_names, inputs, color='skyblue')
    ax1.set_title("Input Feature Values")
    for b in bars:
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{b.get_height():.1f}", ha='center')
    plt.xticks(rotation=15)
    st.pyplot(fig1)

    # Horizontal performance gauge
    fig2, ax2 = plt.subplots(figsize=(6,1.5))
    ax2.barh([0], [score], color="green" if score>=60 else "red")
    ax2.set_xlim(0,100)
    ax2.set_yticks([])
    ax2.set_title("Performance Score")
    ax2.text(score+2, 0, f"{score:.1f}", va='center')
    st.pyplot(fig2)

    # AI tips
    with st.spinner("🤖 Generating advice..."):
        advice = get_ai_tips(inputs, score)
    st.subheader("💡 AI Study Tips")
    st.markdown(advice)

# Footer
st.markdown("---")
st.caption("🔐 Built with NumPy + Matplotlib + Google Gemini AI")
