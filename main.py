import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.linear_model import LinearRegression

# Set up Gemini API key
genai.configure(api_key="AIzaSyAcfTRSVuhJTPsw4uxChpNWRUfTnxniU_k")
model = genai.GenerativeModel("gemini-pro")

# Page settings
st.set_page_config(page_title="🎓 AI Student Performance Predictor", layout="centered")
st.title("🎓 AI-Powered Student Performance Predictor")
st.markdown("Predict academic performance and get actionable study tips based on your habits and history.")

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/devanshvpurohit/project1-/main/Student_Performance.csv")
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    return df.dropna()

df = load_data()
st.success(f"✅ Loaded dataset with {df.shape[0]} records")

# Define features
feature_names = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
target_name = 'Performance Index'

# Train model
X = df[feature_names]
y = df[target_name]
model_lr = LinearRegression().fit(X, y)
coefficients = model_lr.coef_
intercept = model_lr.intercept_

# Show equation
st.subheader("📈 Model Equation")
eq = f"{target_name} = " + " + ".join([f"{c:.2f}×{n}" for c, n in zip(coefficients, feature_names)]) + f" + ({intercept:.2f})"
st.code(eq)

# Show coefficients
st.dataframe(pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients}))

# Input form
inputs = []
with st.form("input_form"):
    st.subheader("📝 Enter Student Data")
    for feature in feature_names:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        default_val = float(df[feature].mean())
        val = st.slider(feature, min_value=min_val, max_value=max_val, value=default_val)
        inputs.append(val)
    submitted = st.form_submit_button("🔮 Predict")

# Predict and display
if submitted:
    score = model_lr.predict([inputs])[0]
    st.metric("Predicted Score", f"{score:.2f}")
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"
    st.write(f"**Predicted Grade:** `{grade}`")
    st.warning("⚠️ At-risk student. Recommend early intervention.") if score < 60 else st.success("👍 Performance prediction is positive.")

    # Plot input values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(feature_names, inputs, color='skyblue')
    ax.set_title("Student Input Values")
    plt.xticks(rotation=15)
    st.pyplot(fig)

    # Fallback + AI Recommendations
    def get_recommendations(inputs, pred_score):
        tips = []
        for feature, value in zip(feature_names, inputs):
            avg = df[feature].mean()
            if value < avg * 0.8:
                if "Sample" in feature:
                    tips.append("📘 Solve more sample question papers.")
                elif "Sleep" in feature:
                    tips.append("🛌 Aim for at least 7 hours of sleep.")
                elif "Hours Studied" in feature:
                    tips.append("📖 Increase study time.")
                elif "Previous Scores" in feature:
                    tips.append("🔁 Revise fundamental concepts.")
                elif "Extracurricular" in feature:
                    tips.append("⚖️ Engage in some extracurricular activities for balance.")

        rule_based = "### 📌 Recommendations Based on Data Analysis:\n" + "\n".join(f"- {t}" for t in tips)

        # AI attempt
        try:
            prompt = "Student data:\n" + "\n".join([f"- {n}: {v}" for n, v in zip(feature_names, inputs)])
            prompt += f"\n\nPredicted score: {pred_score:.2f}\nGive personalized academic tips to improve this student's outcome."
            ai_reply = model.generate_content(prompt).text
            return f"{ai_reply}\n\n---\n{rule_based}"
        except Exception as e:
            return f"⚠️ Gemini API unavailable. Default advice:\n\n{rule_based}"

    st.subheader("💡 Study Recommendations")
    with st.spinner("🤖 Analyzing student profile..."):
        tips = get_recommendations(inputs, score)
    st.markdown(tips)

# Footer
st.markdown("---")
