import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# Load Data & Model
df = pd.read_csv("Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)
df_encoded = pd.get_dummies(df, drop_first=True)

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

# Page Selection
st.set_page_config(page_title="AI Customer Churn Dashboard", layout="wide")
st.title("🚀 AI Customer Churn System")

page = st.sidebar.selectbox("Select Page", ["Prediction", "Analysis"])

# Prediction Page
if page == "Prediction":
    st.subheader("Fill customer details to predict churn:")

    # Numeric Inputs
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges", 0, 150, 70)
    total_charges = st.sidebar.slider("Total Charges", 0, 9000, 1500)

    # Categorical Inputs
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])
    phone_service = st.sidebar.selectbox("Phone Service?", ["No", "Yes"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines?", ["No phone service", "No", "Yes"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    # Create input dataframe with dummy encoding
    input_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender_Male': 1 if gender=="Male" else 0,
        'SeniorCitizen_Yes': 1 if senior_citizen=="Yes" else 0,
        'Partner_Yes': 1 if partner=="Yes" else 0,
        'Dependents_Yes': 1 if dependents=="Yes" else 0,
        'PhoneService_Yes': 1 if phone_service=="Yes" else 0,
        'MultipleLines_No phone service': 1 if multiple_lines=="No phone service" else 0,
        'MultipleLines_No': 1 if multiple_lines=="No" else 0,
        'MultipleLines_Yes': 1 if multiple_lines=="Yes" else 0,
        'InternetService_DSL': 1 if internet_service=="DSL" else 0,
        'InternetService_Fiber optic': 1 if internet_service=="Fiber optic" else 0,
        'InternetService_No': 1 if internet_service=="No" else 0,
        'Contract_Month-to-month': 1 if contract=="Month-to-month" else 0,
        'Contract_One year': 1 if contract=="One year" else 0,
        'Contract_Two year': 1 if contract=="Two year" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method=="Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method=="Mailed check" else 0,
        'PaymentMethod_Bank transfer (automatic)': 1 if payment_method=="Bank transfer (automatic)" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method=="Credit card (automatic)" else 0
    }

    input_df = pd.DataFrame([input_dict])

    # Ensure all model columns exist
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columns]

    # Predict
    if st.sidebar.button("Predict Churn"):
        proba = model.predict_proba(input_df)[0][1]
        risk_percent = round(proba * 100, 2)

        # Risk display
        if risk_percent >= 70:
            st.markdown(f"<h2 style='color:red;'>⚠️ High Risk of Churn ({risk_percent}%)</h2>", unsafe_allow_html=True)
        elif risk_percent >= 40:
            st.markdown(f"<h2 style='color:orange;'>⚠️ Medium Risk of Churn ({risk_percent}%)</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:green;'>✅ Low Risk of Churn ({risk_percent}%)</h2>", unsafe_allow_html=True)

        # SHAP Feature Contribution
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        st.subheader("📊 Feature Contribution")
        fig, ax = plt.subplots(figsize=(6,4))
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        st.pyplot(fig)

        # Suggested Action
        st.subheader("💡 Suggested Action")
        if risk_percent >= 70:
            st.write("Offer retention plan or discount. High priority customer.")
        elif risk_percent >= 40:
            st.write("Monitor closely and send engagement offers.")
        else:
            st.write("Customer is low risk. Continue normal engagement.")

# Analysis Page
elif page == "Analysis":
    st.subheader("📈 Dataset Analysis & Insights")

    # Two-column layout for charts
    col1, col2 = st.columns(2)

    # Churn Distribution
    with col1:
        st.write("**Churn Distribution**")
        fig, ax = plt.subplots()
        churn_counts = df_encoded['Churn_Yes'].value_counts()
        churn_counts.plot(kind='bar', color=['green','red'], ax=ax)
        ax.set_xlabel("Churn (0=No, 1=Yes)")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)

    # Average Monthly Charges by Churn
    with col2:
        st.write("**Average Monthly Charges by Churn**")
        fig, ax = plt.subplots()
        avg_monthly = df_encoded.groupby('Churn_Yes')['MonthlyCharges'].mean()
        avg_monthly.plot(kind='bar', color=['green','red'], ax=ax)
        ax.set_xlabel("Churn (0=No, 1=Yes)")
        ax.set_ylabel("Avg Monthly Charges")
        st.pyplot(fig)

    # Tenure vs Churn (below first row)
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Average Tenure by Churn**")
        fig, ax = plt.subplots()
        avg_tenure = df_encoded.groupby('Churn_Yes')['tenure'].mean()
        avg_tenure.plot(kind='bar', color=['green','red'], ax=ax)
        ax.set_xlabel("Churn (0=No, 1=Yes)")
        ax.set_ylabel("Avg Tenure (months)")
        st.pyplot(fig)

    # Top Features by SHAP
    with col4:
        st.write("**Top Features Affecting Churn (SHAP)**")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 = Churn

        shap_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values(by='Importance', ascending=True).tail(10)  # for horizontal bars

        fig, ax = plt.subplots()
        ax.barh(shap_importance['Feature'], shap_importance['Importance'], color='dodgerblue')
        ax.set_xlabel("Mean |SHAP Value|")
        st.pyplot(fig)

    # Insights
    st.write("💡 Insights:")
    st.write("- Customers with **higher MonthlyCharges** tend to churn more.")
    st.write("- **Shorter tenure** correlates with higher churn risk.")
    st.write("- Contract type and InternetService are important features in prediction.")
