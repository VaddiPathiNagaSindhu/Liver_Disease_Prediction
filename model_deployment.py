#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


# In[2]:


model=pickle.load(open('log.pkl','rb'))


# In[3]:


model


# In[4]:


st.set_page_config(page_title="Liver Disease Prediction", page_icon="🧬")
st.title("🧬 Liver Disease Prediction App")
st.write("Enter your values below to predict whether the patient has liver disease.")


# In[5]:


# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ("Male", "Female"))
total_bilirubin = st.number_input("Total Bilirubin", 0.0, 50.0, 1.0)
direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 20.0, 0.3)
alkaline_phosphatase = st.number_input("Alkaline Phosphatase", 0, 2000, 200)
alamine_aminotransferase = st.number_input("Alamine Aminotransferase (ALT)", 0, 2000, 30)
aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (AST)", 0, 2000, 35)
total_proteins = st.number_input("Total Proteins", 0.0, 10.0, 6.5)
albumin = st.number_input("Albumin", 0.0, 6.0, 3.5)
albumin_globulin_ratio = st.number_input("Albumin/Globulin Ratio", 0.0, 3.0, 1.1)



# In[6]:


# Convert gender to numeric
gender_val = 1 if gender == "Male" else 0


# In[7]:


# Prediction Code
if st.button("Predict"):
    # Arrange data
    input_data = np.array([[age, gender_val, total_bilirubin, direct_bilirubin,
                            alkaline_phosphatase, alamine_aminotransferase,
                            aspartate_aminotransferase, total_proteins,
                            albumin, albumin_globulin_ratio]])

    # Scale data
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)[0]

    # Output
    if prediction == 1:
        st.error(" The patient is likely to have Liver Disease.")
    else:
        st.success("The patient does NOT have Liver Disease.")


# In[ ]:




