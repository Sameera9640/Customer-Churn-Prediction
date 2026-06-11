import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("churn_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave the company.")

# User inputs
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

if st.button("Predict"):

    # Create input with all model features
    input_data = {feature: 0 for feature in feature_names}

    # Fill available features
    if "tenure" in input_data:
        input_data["tenure"] = tenure

    if "MonthlyCharges" in input_data:
        input_data["MonthlyCharges"] = monthly_charges

    if "TotalCharges" in input_data:
        input_data["TotalCharges"] = total_charges

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    st.write(f"Churn Probability: {probability:.2%}")