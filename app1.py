# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Metastasis Predictor", layout="centered")

st.title("Metastasis Prediction using Random Forest")
st.write("This app predicts **metastasis (yes/no)** from Snail1 and EMT values.")
st.write("Input: snail1 (1=low,2=moderate,3=high), emt (1=low,2=moderate,3=high)")

# --- Upload dataset ---
uploaded = st.file_uploader("Upload Excel or CSV file", type=["xlsx","xls","csv"])
if uploaded is not None:
    if uploaded.name.endswith(('.xlsx','.xls')):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)
else:
    st.info("Using default dataset (datahibahdres.xlsx).")
    df = pd.read_excel("datahibahdres.xlsx")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --- Prepare Data ---
if set(df['metastasis'].unique()).issubset({1,2}):
    df['met_bin'] = (df['metastasis'] == 2).astype(int)
else:
    from sklearn.preprocessing import LabelEncoder
    df['met_bin'] = LabelEncoder().fit_transform(df['metastasis'])

X = df[['snail1','emt']]
y = df['met_bin']

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# --- Train Random Forest ---
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = rf_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

st.subheader("Model Evaluation (Random Forest)")
st.write(f"Accuracy on test set: **{test_acc:.4f}**")
st.text("Classification report:")
st.text(classification_report(y_test, y_pred, target_names=["No metastasis","Metastasis"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.matshow(cm, cmap=plt.cm.Blues)
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, str(val), va='center', ha='center')
ax.set_xticklabels([''] + ["Pred No", "Pred Yes"])
ax.set_yticklabels([''] + ["True No", "True Yes"])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# --- Save model ---
if st.button("Save Random Forest Model"):
    joblib.dump(rf_model, "rf_met_model.joblib")
    st.success("Random Forest model saved as rf_met_model.joblib")

# --- Manual prediction ---
st.subheader("Manual Prediction")
snail_val = st.selectbox("Snail1 (1=low,2=moderate,3=high)", [1,2,3], index=0)
emt_val = st.selectbox("EMT (1=low,2=moderate,3=high)", [1,2,3], index=0)

if st.button("Predict Metastasis"):
    pred = rf_model.predict([[snail_val, emt_val]])[0]
    if pred == 1:
        st.warning("Prediction: Metastasis (Yes)")
    else:
        st.success("Prediction: No Metastasis")