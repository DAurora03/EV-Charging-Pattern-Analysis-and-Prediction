import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="EV User Type Prediction", layout="wide")
st.title("🚗 EV User Type Prediction (Commuter vs Non-Commuter)")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ev_charging_patterns.csv", encoding='utf-8', low_memory=False)
    return df

df = load_data()
st.write(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------
# Preprocessing
# ----------------------------
data = df.dropna(subset=["User Type"]).copy()
data["User Type Binary"] = data["User Type"].apply(lambda x: "Commuter" if x=="Commuter" else "Non-Commuter")
data["Charging Start Time"] = pd.to_datetime(data["Charging Start Time"], errors='coerce')
data["Hour"] = data["Charging Start Time"].dt.hour.fillna(12)
data["SOC_Diff"] = data["State of Charge (End %)"] - data["State of Charge (Start %)"]

features = ["Battery Capacity (kWh)", "SOC_Diff", "Charging Duration (hours)", "Hour"]
X = data[features]
y = data["User Type Binary"]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Handle missing values and scale
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split and model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ----------------------------
# Model Evaluation
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)
st.subheader("🔍 Model Performance")
st.write(f"**Accuracy:** {accuracy:.2f}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

#
