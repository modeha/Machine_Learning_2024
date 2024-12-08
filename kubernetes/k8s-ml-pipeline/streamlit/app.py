import streamlit as st
import requests

# Title
st.title("House Price Prediction")

# User inputs
st.header("Enter the features:")
square_footage = st.number_input("Square Footage", min_value=100, max_value=10000, value=1200, step=50)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)

# Submit button
if st.button("Predict"):
    # Send request to the Flask API
    url = "http://localhost:5000/predict"  # Ensure your Flask API is running
    payload = {
        "square_footage": square_footage,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            prediction = response.json().get("predicted_price", "N/A")
            st.success(f"Predicted Price: ${prediction}")
        else:
            st.error(f"Error: Unable to get prediction. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error connecting to the API: {e}")
