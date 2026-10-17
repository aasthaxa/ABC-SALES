import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")

st.title("🚚 Delivery Delay Prediction App")
st.write("This app uses a Decision Tree model to predict whether a delivery will be delayed.")

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('decision_tree_model.joblib')

model = load_model()

# Input fields for all features in X
st.header("Enter Delivery Parameters:")

delivery_distance = st.number_input("Delivery Distance (miles)", min_value=0.0, value=20.0, step=0.1)
traffic_congestion = st.slider("Traffic Congestion Level (1-5)", 1, 5, 3)
weather_condition = st.slider("Weather Condition Score (1-4)", 1, 4, 2)
delivery_slot = st.slider("Delivery Slot (1-3)", 1, 3, 2)
driver_experience = st.number_input("Driver Experience (years)", min_value=0, value=5, step=1)
num_stops = st.slider("Number of Stops", 1, 15, 4)
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, value=5, step=1)
road_condition_score = st.slider("Road Condition Score (1-5)", 1, 5, 3)
package_weight = st.number_input("Package Weight (lbs)", min_value=0.0, value=15.0, step=0.1)
fuel_efficiency = st.number_input("Fuel Efficiency (mpg)", min_value=0.0, value=12.0, step=0.1)
warehouse_processing_time = st.number_input("Warehouse Processing Time (minutes)", min_value=0, value=45, step=1)

# Prepare input array
input_data = pd.DataFrame([[
    delivery_distance,
    traffic_congestion,
    weather_condition,
    delivery_slot,
    driver_experience,
    num_stops,
    vehicle_age,
    road_condition_score,
    package_weight,
    fuel_efficiency,
    warehouse_processing_time
]], columns=[
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
    'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
    'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
    'Warehouse_Processing_Time'
])

# Prediction trigger
if st.button("Predict Delay"):
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ Prediction: The delivery is likely to be **DELAYED**.")
    else:
        st.success("✅ Prediction: The delivery will likely be **ON TIME**.")
