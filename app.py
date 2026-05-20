import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and polynomial transformer
model = joblib.load("model.pkl")
poly = joblib.load("poly.pkl")

st.title("GSO Processing Time Prediction System")

st.write("Enter the details below to predict processing time.")

# User Inputs
num_requests_received = st.number_input("Number of Requests Received", min_value=0)

num_staff_on_duty = st.number_input("Number of Staff on Duty", min_value=0)

budget_allocated = st.number_input("Budget Allocated", min_value=0.0)

system_downtime_hours = st.number_input("System Downtime Hours", min_value=0.0)

avg_request_complexity = st.number_input(
    "Average Request Complexity",
    min_value=0.0,
    max_value=10.0
)

citizen_satisfaction_score = st.number_input(
    "Citizen Satisfaction Score",
    min_value=0.0,
    max_value=10.0
)

# Predict Button
if st.button("Predict Processing Time"):

    # Create dataframe
    input_data = pd.DataFrame([[
        num_requests_received,
        num_staff_on_duty,
        budget_allocated,
        system_downtime_hours,
        avg_request_complexity,
        citizen_satisfaction_score
    ]], columns=[
        "num_requests_received",
        "num_staff_on_duty",
        "budget_allocated",
        "system_downtime_hours",
        "avg_request_complexity",
        "citizen_satisfaction_score"
    ])

    # Transform using polynomial features
    input_poly = poly.transform(input_data)

    # Predict
    prediction = model.predict(input_poly)[0]

    st.subheader(f"Predicted Processing Time: {prediction:.2f} days")

    # Classification
    if prediction <= 3:
        st.success("Category: Fast")
    elif prediction <= 4.5:
        st.warning("Category: Moderate")
    else:
        st.error("Category: Slow")
