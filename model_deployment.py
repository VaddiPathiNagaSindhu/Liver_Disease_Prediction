#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[2]:


st.set_page_config(page_title="Random Forest Model Deployment", layout="centered")

st.title("🚀 Random Forest Model Deployment (No Pickle Used)")

st.write("""
Upload your dataset, select the target column, and the Random Forest model  
will be trained instantly **without using any pickle file**.
""")



# In[3]:


# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset Uploaded Successfully!")
    st.dataframe(df.head())

    # -----------------------------
    # Target Column Selection
    # -----------------------------
    target_column = st.selectbox("Select Target Column", df.columns)

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    if st.button("Train Random Forest Model"):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Use only numeric features
        X = X.select_dtypes(include=[np.number])

        if X.empty:
            st.error("Dataset has no numeric columns to train on.")
            st.stop()

        # Scaling (Optional)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # -----------------------------
        # Model Training
        # -----------------------------
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        st.success("Random Forest Model Trained Successfully!")

        # -----------------------------
        # Evaluation
        # -----------------------------
        y_pred = model.predict(X_test)

        st.subheader("📊 Evaluation Metrics")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred, zero_division=0))
        st.write("Recall:", recall_score(y_test, y_pred, zero_division=0))
        st.write("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

        # -----------------------------
        # Prediction Section
        # -----------------------------
        st.header("🔮 Make Predictions")

        input_data = {}

        for col in X.columns:
            input_data[col] = st.number_input(f"Enter value for {col}", value=float(X[col].mean()))

        if st.button("Predict Output"):
            input_array = np.array(list(input_data.values())).reshape(1, -1)
            input_array = scaler.transform(input_array)

            prediction = model.predict(input_array)[0]
            st.success(f"Prediction: **{prediction}**")


# In[ ]:




