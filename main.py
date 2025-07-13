import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.linear_model import LinearRegression

genai.configure(api_key="AIzaSyAcfTRSVuhJTPsw4uxChpNWRUfTnxniU_k")
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")

# Streamlit page settings
st.set_page_config(page_title="🎓 AI Student Performance Predictor", layout="centered")
st.title("🎓 AI-Powered Student Performance Predictor")
st.markdown("Predict academic performance based on study habits, past scores, and behavior. Get AI recommendations to improve outcomes.")

# Load data from GitHub
CSV_URL = "https://raw.githubusercontent.com/devanshvpurohit/project1-/main/Student_Performance.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    return df.dropna()

df = load_data()
st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# Features and Target
feature_names = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
target_name = 'Performance Index'
X = df[feature_names]
y = df[target_name]

# Train model
model_lr = LinearRegression()
model_lr.fit(X, y)
coefficients = model_lr.coef_
intercept = model_lr.intercept_

# Show model equation
st.subheader("🔬 Trained Model Equation")
eq = f"{target_name} = " + " + ".join([f"{c:.2f}×{n}" for c, n in zip(coefficients, feature_names)]) + f" + ({intercept:.2f})"
st.code(eq)

# Show coefficient table
coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
st.dataframe(coef_df)

# User inputs
inputs = []
with st.form("input_form"):
    st.subheader("📥 Enter Student Data")
    for feature in feature_names:
        vmin, vmax = float(df[feature].min()), float(df[feature].max())
        val = st.slider(feature, min_value=vmin, max_value=vmax, value=(vmin + vmax) / 2)
        inputs.append(val)
    submitted = st.form_submit_button("🔮 Predict")

# Predict
if submitted:
    score = model_lr.predict([inputs])[0]
    st.subheader("📊 Prediction Result")
    st.metric("Predicted Score", f"{score:.2f}")
    grade = ("A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F")
    st.write(f"**Estimated Grade:** `{grade}`")
    if score < 60:
        st.error("🚨 This student is at risk—early intervention recommended.")
    else:
        st.success("✅ Performance prediction is satisfactory.")

    # Plot input values
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    bars = ax1.bar(feature_names, inputs, color='skyblue')
    ax1.set_title("Input Feature Values")
    for b in bars:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}", ha='center')
    plt.xticks(rotation=15)
    st.pyplot(fig1)

    # Performance gauge
    fig2, ax2 = plt.subplots(figsize=(6, 1.5))
    ax2.barh([0], [score], color="green" if score >= 60 else "red")
    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_title("Performance Score")
    ax2.text(score + 2, 0, f"{score:.1f}", va='center')
    st.pyplot(fig2)

    # AI recommendation
    def get_ai_tips(inputs, pred):
        prompt = "Student indicators:\n" + "\n".join([f"- {n}: {v}" for n, v in zip(feature_names, inputs)])
        prompt += f"\n\nPredicted score: {pred:.2f}.\nGive bullet‑point recommendations to improve academic performance."
        try:
            return model.generate_content(prompt).text
        except Exception as e:
            return f"⚠️ Error: {e}"

    with st.spinner("🤖 Generating study advice..."):
        advice = get_ai_tips(inputs, score)
    st.subheader("💡 AI Recommendations")
    st.markdown(advice)

# Footer
st.markdown("---")
st.caption("🔐 Built using Scikit-learn + Gemini AI + Streamlit + Matplotlib")
