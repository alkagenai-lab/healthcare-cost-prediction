import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/api/v1/predict",
)
# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Healthcare Cost Prediction",
    page_icon="🏥",
    layout="centered",
)


# ---------------------------------------------------------
# API Configuration
# ---------------------------------------------------------



# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/api/v1/predict",
)
st.title("🏥 Healthcare Treatment Cost Prediction")

st.write(
    "Enter patient information to predict the estimated "
    "healthcare treatment cost."
)


# ---------------------------------------------------------
# Input Fields
# ---------------------------------------------------------

st.subheader("Patient Information")

age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=30,
    step=1,
)

sex = st.selectbox(
    "Sex",
    options=["female", "male"],
)

bmi = st.number_input(
    "BMI",
    min_value=1.0,
    max_value=100.0,
    value=25.0,
    step=0.1,
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=20,
    value=0,
    step=1,
)

smoker = st.selectbox(
    "Smoker",
    options=["no", "yes"],
)

region = st.selectbox(
    "Region",
    options=[
        "southwest",
        "southeast",
        "northwest",
        "northeast",
    ],
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button(
    "💰 Predict Healthcare Cost",
    type="primary",
    use_container_width=True,
):

    payload = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }

    try:

        with st.spinner("Calculating healthcare cost..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=10,
            )

        if response.status_code == 200:

            result = response.json()

            predicted_cost = result["predicted_cost"]
            currency = result["currency"]

            st.success(
                "Prediction generated successfully!"
            )

            st.metric(
                label="Estimated Healthcare Cost",
                value=f"{currency} {predicted_cost:,.2f}",
            )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the FastAPI server. "
            "Please make sure the backend is running."
        )

    except requests.exceptions.Timeout:

        st.error(
            "The API request timed out. "
            "Please try again."
        )

    except requests.exceptions.RequestException as exc:

        st.error(
            f"Request failed: {exc}"
        )