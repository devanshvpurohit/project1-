import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# App config
st.set_page_config(page_title="🎓 Student Performance Predictor", layout="centered")
st.title("🎓 AI Student Performance Predictor")
st.markdown("Predict student academic performance and get personalized improvement recommendations.")

# Load and preprocess data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/devanshvpurohit/project1-/main/Student_Performance.csv"
    df = pd.read_csv(url)
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    df.dropna(inplace=True)
    return df

df = load_data()
st.success(f"✅ Dataset loaded with {df.shape[0]} records.")

# Create binary performance label
df['Performance_Label'] = df['Performance Index'].apply(lambda x: 1 if x >= 60 else 0)

# Define features and target
features = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
X = df[features]
y = df['Performance_Label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Evaluation")
st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
st.code(classification_report(y_test, y_pred), language="text")

# User Input Form
with st.form("user_input"):
    st.subheader("🧮 Enter Student Metrics")
    inputs = []
    for feature in features:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        default_val = (min_val + max_val) / 2
        user_val = st.slider(feature, min_value=min_val, max_value=max_val, value=default_val)
        inputs.append(user_val)
    submitted = st.form_submit_button("Predict Performance")

# Predict & Display Results
if submitted:
    prediction = model.predict([inputs])[0]
    confidence = model.predict_proba([inputs])[0][1]

    result = "Pass" if prediction == 1 else "Fail"
    st.subheader("📈 Prediction Result")
    st.metric("Prediction", result)
    st.write(f"Confidence: **{confidence:.2%}**")

    if prediction == 0:
        st.warning("⚠️ At-risk student. Early intervention recommended.")
    else:
        st.success("✅ Student is likely on track.")

    # Visualize input values
    fig1, ax1 = plt.subplots()
    bars = ax1.bar(features, inputs, color='skyblue')
    ax1.set_title("Input Feature Values")
    ax1.set_ylabel("Value")
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.3, f"{height:.1f}", ha='center')
    plt.xticks(rotation=15)
    st.pyplot(fig1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    # Recommendations (rule-based with if-else)
    st.subheader("💡 Recommendations")
    gave_tips = False

    if inputs[features.index("Hours Studied")] < 2:
        st.write("📚 Recommendation: Increase study time to at least 2 hours per day.")
        gave_tips = True
    else:
        st.write("✅ Good amount of study time!")

    if inputs[features.index("Sleep Hours")] < 6:
        st.write("😴 Recommendation: Sleep at least 6 hours for better focus.")
        gave_tips = True
    else:
        st.write("✅ You're sleeping enough!")

    if inputs[features.index("Sample Question Papers Practiced")] < 3:
        st.write("📝 Recommendation: Practice more sample papers.")
        gave_tips = True
    else:
        st.write("✅ Well done on practicing sample papers!")

    if inputs[features.index("Extracurricular Activities")] == 0:
        st.write("🎯 Recommendation: Consider joining extracurricular activities.")
        gave_tips = True
    else:
        st.write("✅ Great to see extracurricular participation!")

    if not gave_tips:
        st.success("🎉 You're doing great! No major recommendations at this time.")

# Footer
st.markdown("---")
st.caption("🔧 Built with Scikit-learn, Streamlit, Matplotlib, and Seaborn.")
