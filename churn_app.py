import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("columns.pkl")

st.title("📊 Smart Customer Churn Prediction System")
st.write("Predict whether a customer is likely to churn")

# ------------------------
# INPUTS
# ------------------------

tenure = st.number_input("Tenure", min_value=0, max_value=100, value=1)

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)

TotalCharges = st.number_input("Total Charges", min_value=0.0, value=100.0)

SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])

partner = st.selectbox("Partner", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# ------------------------
# INPUT DATAFRAME
# ------------------------

input_data = pd.DataFrame([[0] * len(model_columns)], columns=model_columns)

# Numeric features
if "tenure" in input_data.columns:
    input_data["tenure"] = tenure

if "MonthlyCharges" in input_data.columns:
    input_data["MonthlyCharges"] = MonthlyCharges

if "TotalCharges" in input_data.columns:
    input_data["TotalCharges"] = TotalCharges

if "SeniorCitizen" in input_data.columns:
    input_data["SeniorCitizen"] = SeniorCitizen

# Binary features
if partner == "Yes" and "Partner_Yes" in input_data.columns:
    input_data["Partner_Yes"] = 1

if dependents == "Yes" and "Dependents_Yes" in input_data.columns:
    input_data["Dependents_Yes"] = 1

# Categorical features
contract_col = f"Contract_{contract}"
if contract_col in input_data.columns:
    input_data[contract_col] = 1

internet_col = f"InternetService_{internet}"
if internet_col in input_data.columns:
    input_data[internet_col] = 1

payment_col = f"PaymentMethod_{payment}"
if payment_col in input_data.columns:
    input_data[payment_col] = 1

# ------------------------
# PREDICTION
# ------------------------

if st.button("Predict Churn"):

    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)

    st.write("Raw Prediction:", prediction)
    st.write("Churn Probability:", round(proba[0][1] * 100, 2), "%")

    # Risk labeling
    churn_risk = proba[0][1]

    if churn_risk < 0.3:
        st.success("🟢 Low Risk Customer")
    elif churn_risk < 0.6:
        st.warning("🟡 Medium Risk Customer")
    else:
        st.error("🔴 High Risk Customer")

    # Final output
    if prediction[0] == 1:
        st.error("⚠ Customer is Likely to Churn")

        st.write("### Risk Analysis")
        st.write("""
        This customer shows churn behavior patterns.

        Possible reasons:
        - High service dissatisfaction
        - Better competitor offers
        - Pricing concerns
        """)

    else:
        st.success("✅ Customer is Likely to Stay")

        st.write("### Stability Analysis")
        st.write("""
        This customer appears stable and engaged.

        Positive indicators:
        - Healthy service usage
        - Stable subscription pattern
        - Lower churn probability
        """)