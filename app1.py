import streamlit as st
import pickle
import numpy as np

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title="NutriCheck",
    page_icon="🥗",
    layout="centered"
)

# ---------------------------------
# Custom CSS (Fix label color + blue background)
# ---------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #B3D1FF 0%, #EAF2FF 50%, #FFFFFF 100%);
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #0D47A1;
    margin-top: 30px;
    margin-bottom: 8px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #444;
    margin-bottom: 30px;
}

/* 🔑 INPUT LABEL FIX */
label {
    color: #0D47A1 !important;
    font-weight: 600;
}

/* Result box */
.result-box {
    background-color: #E3F2FD;
    color: #0D47A1;
    padding: 18px;
    border-radius: 12px;
    font-size: 18px;
    margin-top: 25px;
    font-weight: 600;
    text-align: center;
}

/* Info box */
.info-box {
    background-color: #F1F8E9;
    color: #1B5E20;
    padding: 12px;
    border-radius: 10px;
    font-size: 14px;
    margin-top: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load model & encoders
# ---------------------------------
with open("nutricheck_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("gender_encoder.pkl", "rb") as file:
    gender = pickle.load(file)

with open("deficiency_encoder.pkl", "rb") as file:
    defi = pickle.load(file)

with open("food_encoder.pkl", "rb") as file:
    food = pickle.load(file)

# ---------------------------------
# Title Section
# ---------------------------------
st.markdown('<div class="title">🥗 NutriCheck</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Smart food recommendations based on nutritional deficiency</div>',
    unsafe_allow_html=True
)

# ---------------------------------
# Inputs (labels now visible)
# ---------------------------------
age = st.number_input("Age", min_value=1, max_value=100, value=25)

gender_input = st.selectbox(
    "Gender",
    gender.classes_
)

deficiency_input = st.selectbox(
    "Nutritional Deficiency",
    defi.classes_
)

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("🔍 Get Food Recommendation"):
    gender_enc = gender.transform([gender_input])[0]
    def_enc = defi.transform([deficiency_input])[0]

    prediction = model.predict([[age, gender_enc, def_enc]])
    recommended_food = food.inverse_transform(prediction)[0]

    st.markdown(
        f"""
        <div class="result-box">
            🍽️ Recommended Food: {recommended_food}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
             This recommendation is for dietary guidance only and not medical advice.
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.caption("NutriCheck | ML-based Food Recommendation System")
