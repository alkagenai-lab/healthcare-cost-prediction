<<<<<<< HEAD
# Healthcare Treatment Cost Prediction

An end-to-end machine learning application that predicts healthcare treatment costs based on patient information.

The project uses Linear Regression for prediction, Scikit-learn for machine learning, FastAPI for the backend API, and Streamlit for the frontend.

---

## Project Overview

The application predicts estimated healthcare treatment costs using the following patient information:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The trained machine learning model is exposed through a FastAPI REST API and consumed by a Streamlit web application.

---

## Architecture

```text
User
 |
 v
Streamlit Frontend
 |
 | HTTP POST Request
 v
FastAPI Backend
 |
 v
Pydantic Validation
 |
 v
Prediction Service
 |
 v
Preprocessing Pipeline
 |
 v
Linear Regression Model
 |
 v
Predicted Healthcare Cost
=======
# healthcare-cost-prediction
End-to-end Healthcare Treatment Cost Prediction using Machine Learning, FastAPI, Streamlit and Docker.
>>>>>>> 5e12ed681c38d20436d6a36066092d2fb88c10bc
