# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("_final_csat_model.pkl")

st.title("Customer Satisfaction Predictor")

channel = st.selectbox("Channel", ["Phone", "Email", "Chat"])
shift = st.selectbox("Agent Shift", ["Day", "Night"])
item_price = st.number_input("Item Price")
# Add more inputs...

if st.button("Predict"):
    # Convert inputs into dataframe (include encoding logic here)
    data = pd.DataFrame([[channel, shift, item_price]], columns=['channel_name', 'Agent Shift', 'Item_price'])
    # Preprocess (label encode if needed)
    prediction = model.predict(data)[0]
    st.success("Customer will be Satisfied!" if prediction == 1 else "Customer may be Unsatisfied.")
