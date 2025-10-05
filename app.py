# app.py
import streamlit as st
import pandas as pd
import joblib

# =============================
# 1. Load Model
# =============================
model = joblib.load("xgboost.pkl")

# Extract training feature names
if hasattr(model, "get_booster"):
    feature_names = model.get_booster().feature_names
else:
    feature_names = model.feature_names_in_

# =============================
# 2. Streamlit App UI
# =============================
st.set_page_config(page_title="Amazon Delivery Time Prediction", layout="centered")
st.title("Amazon Delivery Time Prediction App")

st.write("Enter order details below to predict the delivery time (in hours).")

# --- Numerical Inputs ---
agent_age = st.number_input("Agent Age", min_value=18, max_value=70, value=30)
agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.0)
distance_km = st.number_input("Distance (km)", min_value=0.1, max_value=5000.0, value=10.0)
pickup_delay = st.number_input("Pickup Delay (minutes)", min_value=0, max_value=600, value=10)
order_day = st.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)))
order_hour = st.slider("Order Hour", 0, 23, 12)

# --- Categorical Inputs ---
weather = st.selectbox("Weather", ["Fog", "Sandstorms", "Stormy", "Sunny", "Windy"])
traffic = st.selectbox("Traffic", ["Jam ", "Low ", "Medium "])  # note spaces
vehicle = st.selectbox("Vehicle", ["scooter ", "van"])  # note space in "scooter "
area = st.selectbox("Area", ["Other", "Semi-Urban ", "Urban "])  # note spaces
category = st.selectbox(
    "Product Category",
    ["Books","Clothing","Cosmetics","Electronics","Grocery","Home","Jewelry",
     "Kitchen","Outdoors","Pet Supplies","Shoes","Skincare","Snacks","Sports","Toys"]
)

# =============================
# 3. Build Input Dictionary
# =============================
input_dict = {
    "Agent_Age": agent_age,
    "Agent_Rating": agent_rating,
    "Store_Latitude": 0,   # not provided in UI
    "Store_Longitude": 0,  # not provided in UI
    "Drop_Latitude": 0,    # not provided in UI
    "Drop_Longitude": 0,   # not provided in UI
    "Distance_km": distance_km,
    "Order_DayOfWeek": order_day,
    "Order_Hour": order_hour,
    "Pickup_Delay_min": pickup_delay,
}

# Add one-hot encoded features for chosen options
input_dict[f"Weather_{weather}"] = 1
input_dict[f"Traffic_{traffic}"] = 1
input_dict[f"Vehicle_{vehicle}"] = 1
input_dict[f"Area_{area}"] = 1
input_dict[f"Category_{category}"] = 1

# Convert to DataFrame
input_data = pd.DataFrame([input_dict])

# =============================
# 4. Align with Training Features
# =============================
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# =============================
# 5. Prediction
# =============================
if st.button("Predict Delivery Time"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Delivery Time: **{prediction:.2f} hours**")
