# ==========================
# app.py
# ==========================

import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("final_csat_model.pkl")
le_channel = joblib.load("le_channel.pkl")
le_shift = joblib.load("le_shift.pkl")

st.set_page_config(page_title="CSAT Predictor", layout="centered")
st.title("📊 Customer Satisfaction Predictor")

# Input UI
channel = st.selectbox("Channel", ["Phone", "Email", "Chat"])
shift = st.selectbox("Agent Shift", ["Day", "Night"])
item_price = st.number_input("Item Price", min_value=0.0, step=1.0)

if st.button("🔮 Predict"):
    try:
        # Encode categorical values
        channel_encoded = le_channel.transform([channel])[0]
        shift_encoded = le_shift.transform([shift])[0]

        # Build input
        input_data = pd.DataFrame([[channel_encoded, shift_encoded, item_price]],
                                  columns=['channel_name', 'Agent Shift', 'Item_price'])

        prediction = model.predict(input_data)[0]

        # Output result
        if prediction == 1:
            st.success("✅ Customer will be Satisfied.")
        else:
            st.warning("⚠️ Customer may be Unsatisfied.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
