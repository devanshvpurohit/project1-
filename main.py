import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Streamlit page config
st.set_page_config(page_title="🎓 AI Student Performance Predictor", layout="centered")
st.title("🎓 AI-Powered Student Performance Predictor")
st.markdown("Predict academic performance based on academic history, behavior, and engagement metrics. Receive early alerts and personalized tips to improve outcomes.")

# Load dataset
CSV_URL = "https://raw.githubusercontent.com/devanshvpurohit/project1-/main/Student_Performance.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    df.dropna(inplace=True)
    return df

df = load_data()
st.success(f"✅ Loaded dataset with {df.shape[0]} records")

# Feature Engineering
df['Performance_Label'] = df['Performance Index'].apply(lambda x: 1 if x >= 60 else 0)  # 1 = Pass, 0 = Fail

# Features and labels
feature_names = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
target_name = 'Performance_Label'
X = df[feature_names]
y = df[target_name]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Evaluation")
st.write(f"**Accuracy:** {acc:.2f}")
st.text("Classification Report:")
st.code(classification_report(y_test, y_pred))

# User Input
inputs = []
with st.form("input_form"):
    st.subheader("📥 Enter Student Data")
    for feature in feature_names:
        vmin, vmax = float(df[feature].min()), float(df[feature].max())
        val = st.slider(feature, min_value=vmin, max_value=vmax, value=(vmin + vmax) / 2)
        inputs.append(val)
    submitted = st.form_submit_button("🔮 Predict")

# Prediction
if submitted:
    predicted_label = model.predict([inputs])[0]
    predicted_proba = model.predict_proba([inputs])[0][1]

    st.subheader("📊 Prediction Result")
    status = "Pass" if predicted_label == 1 else "Fail"
    st.metric("Predicted Outcome", status)
    st.write(f"**Confidence:** {predicted_proba:.2%}")

    # Risk Warning
    if predicted_label == 0:
        st.warning("⚠️ At-risk student. Early support recommended.")
    else:
        st.success("👍 Student performance is on track.")

    # Plot input values
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    bars = ax1.bar(feature_names, inputs, color='teal')
    ax1.set_title("Input Feature Values")
    for b in bars:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}", ha='center')
    plt.xticks(rotation=15)
    st.pyplot(fig1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix")
    st.pyplot(fig2)

    # Recommendations (rule-based)
    recommendations = []
    if inputs[feature_names.index("Hours Studied")] < 2:
        recommendations.append("📚 Increase study time to at least 2 hours daily.")
    if inputs[feature_names.index("Sleep Hours")] < 6:
        recommendations.append("😴 Ensure at least 6 hours of sleep for better concentration.")
    if inputs[feature_names.index("Sample Question Papers Practiced")] < 3:
        recommendations.append("📝 Practice more sample papers for exam readiness.")
    if inputs[feature_names.index("Extracurricular Activities")] == 0:
        recommendations.append("🎯 Engage in extracurriculars for better cognitive balance.")

    if recommendations:
        st.subheader("💡 Recommendations")
        for tip in recommendations:
            st.write(tip)

# Footer
st.markdown("---")
st.caption("🔐 Built using Random Forest, Streamlit, Matplotlib, and Seaborn")
