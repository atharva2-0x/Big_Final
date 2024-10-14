import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and the column transformer
model = joblib.load('logistic_regression_model.joblib')


# Function to make predictions
def predict_campaign_success(data):
    prediction = model.predict(data)
    return prediction


# Streamlit app layout
st.title("Campaign Success Predictor")
st.write("Enter the details of the promotional campaign:")

# User inputs
previous_sales = st.number_input("Previous Sales",
                                 min_value=0.0,
                                 format="%.2f")
during_campaign_sales = st.number_input("Sales During Campaign",
                                        min_value=0.0,
                                        format="%.2f")
foot_traffic_before = st.number_input("Foot Traffic Before Campaign",
                                      min_value=0)
foot_traffic_during = st.number_input("Foot Traffic During Campaign",
                                      min_value=0)
campaign_duration = st.number_input("Campaign Duration (Days)", min_value=1)

# Categorical inputs
promotion_type = st.selectbox(
    "Promotion Type",
    ["Discount", "Buy One Get One", "Loyalty Points", "Freebie"])
day_of_week = st.selectbox("Day of the Week", [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday"
])

# Prepare the input data for the model
input_data = pd.DataFrame({
    'Previous_Sales': [previous_sales],
    'During_Campaign_Sales': [during_campaign_sales],
    'Foot_Traffic_Before_Campaign': [foot_traffic_before],
    'Foot_Traffic_During_Campaign': [foot_traffic_during],
    'Campaign_Duration_Days': [campaign_duration],
    'Promotion_Type': [promotion_type],
    'Day_of_Week': [day_of_week]
})

# Create dummy variables for categorical features
input_data = pd.get_dummies(input_data,
                            columns=['Promotion_Type', 'Day_of_Week'],
                            drop_first=True)

# Ensure all necessary columns are present in the input data
# Load the model's expected feature names
model_feature_names = model.feature_names_in_

# Create missing columns with zeros
for col in model_feature_names:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder the columns to match the model's training data
input_data = input_data[model_feature_names]

# Make prediction
if st.button("Predict"):
    prediction = predict_campaign_success(input_data)
    result = "Successful" if prediction[0] == 1 else "Not Successful"
    st.write(f"The predicted campaign success is: **{result}**")
