import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Streamlit settings
st.set_page_config(page_title="🎓 AI Student Performance Predictor", layout="centered")
st.title("🎓 AI-Powered Student Performance Predictor")
st.markdown("Predict academic performance based on study habits, past scores, and get study advice—powered entirely offline.")

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/devanshvpurohit/project1-/main/Student_Performance.csv")
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    return df.dropna()

df = load_data()
st.success(f"✅ Loaded dataset with {df.shape[0]} records")

# Feature columns
feature_names = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
target_name = 'Performance Index'

# Train model
X = df[feature_names]
y = df[target_name]
model_lr = LinearRegression().fit(X, y)
coefficients = model_lr.coef_
intercept = model_lr.intercept_

# Show model equation
st.subheader("📈 Model Equation")
equation = f"{target_name} = " + " + ".join([f"{c:.2f}×{n}" for c, n in zip(coefficients, feature_names)]) + f" + ({intercept:.2f})"
st.code(equation)

# Coefficient display
st.dataframe(pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients}))

# User Input
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

# On prediction
if submitted:
    predicted_score = model_lr.predict([inputs])[0]
    st.metric("Predicted Score", f"{predicted_score:.2f}")
    grade = "A" if predicted_score >= 90 else "B" if predicted_score >= 80 else "C" if predicted_score >= 70 else "D" if predicted_score >= 60 else "F"
    st.write(f"**Predicted Grade:** `{grade}`")
    st.warning("⚠️ At-risk student. Early support recommended.") if predicted_score < 60 else st.success("👍 Student performance is on track.")

    # Plot inputs
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(feature_names, inputs, color='skyblue')
    ax.set_title("Student Input Values")
    plt.xticks(rotation=15)
    st.pyplot(fig)

    # Rule-based recommendation
    def generate_recommendations(inputs, df):
        tips = []
        for feature, value in zip(feature_names, inputs):
            avg = df[feature].mean()
            if value < avg * 0.8:
                if "Sample" in feature:
                    tips.append("📘 Solve more sample question papers.")
                elif "Sleep" in feature:
                    tips.append("🛌 Aim for 7–8 hours of consistent sleep.")
                elif "Hours Studied" in feature:
                    tips.append("📖 Increase study time gradually.")
                elif "Previous Scores" in feature:
                    tips.append("📚 Review and strengthen previous concepts.")
                elif "Extracurricular" in feature:
                    tips.append("⚖️ Consider some extracurriculars for mental balance.")
        return tips

    st.subheader("💡 Study Recommendations")
    tips = generate_recommendations(inputs, df)
    if tips:
        st.markdown("### 📌 Suggestions:")
        for t in tips:
            st.markdown(f"- {t}")
    else:
        st.success("🎉 You're performing well across all metrics!")

# Footer
st.markdown("---")
st.caption("🧠 Built with Scikit-learn, Streamlit & Matplotlib – 100% offline, no API used.")
