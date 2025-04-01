import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "creditcard.csv"  # Ensure this file is present in the same folder

@st.cache_data
def load_data():
    data = pd.read_csv(file_path)
    return data

data = load_data()

# Data preprocessing
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
legit_sample = legit.sample(n=len(fraud), random_state=2)
balanced_data = pd.concat([legit_sample, fraud], axis=0)

# Splitting data
X = balanced_data.drop(columns=["Class"], axis=1)
y = balanced_data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train Logistic Regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_train_acc = accuracy_score(log_model.predict(X_train), y_train)
log_test_acc = accuracy_score(log_model.predict(X_test), y_test)

# Train RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=2)
rf_model.fit(X_train, y_train)
rf_train_acc = accuracy_score(rf_model.predict(X_train), y_train)
rf_test_acc = accuracy_score(rf_model.predict(X_test), y_test)

# Streamlit UI
st.title("💳 Credit Card Fraud Detection")
st.markdown("Detect fraudulent transactions using machine learning.")

# Sidebar stats
st.sidebar.header("📊 Dataset Overview")
st.sidebar.write(f"Total Transactions: {len(data)}")
st.sidebar.write(f"Fraud Cases: {len(fraud)}")
st.sidebar.write(f"Legitimate Cases: {len(legit)}")

# User Input Section
st.subheader("🔍 Predict a Transaction")
st.write("Enter the transaction features below:")
input_data = st.text_area("Enter feature values separated by commas:")

if st.button("Predict Fraudulence"):
    try:
        features = np.array(input_data.split(","), dtype=np.float64)
        prediction = log_model.predict(features.reshape(1, -1))
        
        if prediction[0] == 0:
            st.success("✅ Legitimate Transaction")
        else:
            st.error("🚨 Fraudulent Transaction Detected!")
    except Exception as e:
        st.error("Invalid input. Please enter valid feature values.")

# Model Accuracy Display
st.sidebar.subheader("📈 Model Performance")
st.sidebar.write(f"Logistic Regression - Training Accuracy: {log_train_acc}")
st.sidebar.write(f"Logistic Regression - Test Accuracy: {log_test_acc}")
st.sidebar.write(f"Random Forest - Training Accuracy: {rf_train_acc}")
st.sidebar.write(f"Random Forest - Test Accuracy: {rf_test_acc}")

# Accuracy Line Chart in Sidebar
st.sidebar.subheader("📈 Training vs Test Accuracy")
fig, ax = plt.subplots()
models = ["Logistic Regression", "Random Forest"]
train_acc = [log_train_acc, rf_train_acc]
test_acc = [log_test_acc, rf_test_acc]
ax.plot(models, train_acc, marker='o', linestyle='-', label="Training Accuracy", color='blue')
ax.plot(models, test_acc, marker='o', linestyle='--', label="Test Accuracy", color='orange')
ax.set_ylabel("Accuracy")
ax.legend()
st.sidebar.pyplot(fig)