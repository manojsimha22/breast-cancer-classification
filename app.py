import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Breast Cancer Classification Dashboard", layout="wide")

st.title("🩺 Breast Cancer Classification Dashboard")
st.markdown("""
This application demonstrates various machine learning models trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset**.
Upload some test data to see the model performance and evaluation metrics.
""")

st.sidebar.header("User Input Features")

model_options = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Choose a Model", model_options)

@st.cache_resource
def load_model(name):
    filename = f"model/{name.lower().replace(' ', '_')}.joblib"
    if os.path.exists(filename):
        return joblib.load(filename)
    return None

model = load_model(selected_model_name)

uploaded_file = st.sidebar.file_content = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success("File Uploaded Successfully!")
    
    if st.checkbox("Show Raw Data"):
        st.write(test_df.head())

    if 'target' not in test_df.columns:
        st.error("The uploaded CSV must contain a 'target' column for evaluation.")
    else:
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        if model:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            with col2:
                st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
            with col3:
                st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
            with col4:
                st.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
            with col5:
                st.metric("AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
            with col6:
                st.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")


            st.write("---")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                st.pyplot(fig)
            
            with col_b:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.write(report_df)
                                
        else:
            st.error("Model file not found. Please ensure the models are trained and saved in the 'model/' directory.")
else:
    st.info("Please upload a test dataset (CSV) to begin.")
    st.markdown("""
    **Download Template/Test Data:**
    If you don't have test data, you can use the `test_data.csv` generated during training.
    """)
    if os.path.exists("test_data.csv"):
        with open("test_data.csv", "rb") as file:
            st.download_button(
                label="Download test_data.csv",
                data=file,
                file_name="test_data.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("Created for ML Assignment 2 - BITS Pilani")
st.markdown("M P Manoj Simha")
st.markdown("2025AA05213")