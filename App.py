import streamlit as st
import numpy as np
import pickle

# ---------------- LOAD ----------------
model = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🛍️ Customer Segmentation App")

st.write("Enter customer details")

# 👉 IMPORTANT: MATCH NUMBER OF FEATURES (6)
age = st.number_input("Age", 18, 100)
annual_income = st.number_input("Annual Income")
spending_score = st.number_input("Spending Score")
purchase_freq = st.number_input("Purchase Frequency")
loyalty_score = st.number_input("Loyalty Score")
product_rating = st.number_input("Product Rating")

if st.button("Predict Segment"):

    input_data = np.array([[
        age,
        annual_income,
        spending_score,
        purchase_freq,
        loyalty_score,
        product_rating
    ]])

    scaled = scaler.transform(input_data)
    cluster = model.predict(scaled)

    st.success(f"Customer belongs to Segment: {cluster[0]}")