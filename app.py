import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
@st.cache_resource
def load_model():
    with open('model/best_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_info():
    
    with open('model/model_info.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
info = load_info()

# Title
st.title("Salary Predictor")
st.image('salary.jpg')

# Inputs
education = st.selectbox("Education Level", info['edu_levels'])
job_field = st.selectbox("Job Field", sorted(info['job_fields']))

# Predict
if st.button("Predict"):
    edu_num = info['edu_num_map'][education]

    input_df = pd.DataFrame([{
        'edu_numeric': edu_num,
        'category': job_field
    }])

    predicted_salary = np.expm1(model.predict(input_df)[0])

    st.write("Predicted Salary:")
    st.write(f"${predicted_salary:,.0f} per month")