# churn.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Page config & style
st.set_page_config(page_title="📊 Telecom Churn Dashboard", layout="wide")
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('churn_dataset.csv')

@st.cache_resource
def load_advanced_model():
    with open('advanced_churn_model.pkl', 'rb') as f:
        model, scaler, columns = pickle.load(f)
    return model, scaler, columns

data = load_data()
model, scaler, model_columns = load_advanced_model()

# Sidebar branding
st.sidebar.title("📊 Churn Dashboard")
st.sidebar.caption("Analyze, predict & act on customer churn.")

# Main title & intro
st.title("✨ Telecom Customer Churn Dashboard")
st.markdown("Use the dashboard below to explore churn trends and predict churn risk for new customers.")

# Key metric
churn_rate = (data['Churn'].value_counts(normalize=True) * 100).get('Yes', 0)
st.metric("📉 Overall Churn Rate", f"{churn_rate:.2f} %")

# Tabs
tab1, tab2 = st.tabs(["📈 EDA & Insights", "🔮 Predict Churn"])

with tab1:
    st.subheader("✅ Churn Distribution")
    churn_counts = data['Churn'].value_counts()
    fig, ax = plt.subplots()
    bars = ax.bar(churn_counts.index, churn_counts.values, color=['#FF6B6B','#4ECDC4'])
    ax.bar_label(bars)
    st.pyplot(fig)

    st.subheader("📑 Churn by Contract Type")
    churn_rate_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().get('Yes',0)*100
    fig, ax = plt.subplots()
    bars = ax.bar(churn_rate_contract.index, churn_rate_contract.values, color='#ffa600')
    ax.bar_label(bars, fmt='%.1f%%')
    ax.set_ylabel('Churn Rate (%)')
    st.pyplot(fig)

    st.subheader("💳 Churn by Payment Method")
    churn_rate_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack().get('Yes',0)*100
    churn_rate_payment = churn_rate_payment.sort_values(ascending=False)
    fig, ax = plt.subplots()
    bars = ax.barh(churn_rate_payment.index, churn_rate_payment.values, color='#00b4d8')
    ax.bar_label(bars, fmt='%.1f%%')
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### ✏️ **Key Insights:**")
    st.markdown("""
    - Month-to-month contracts and electronic checks see the highest churn.
    - Short tenure customers churn significantly more.
    - Higher charges can be an indicator, but contract type matters more.
    """)

with tab2:
    st.subheader("🔮 Predict Customer Churn")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider('Tenure (months)', 0, 100, 12)
            monthly = st.number_input('Monthly Charges', 0.0, 200.0, 70.0)
            total = st.number_input('Total Charges', 0.0, 10000.0, 2500.0)
        with col2:
            contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
            payment = st.selectbox('Payment Method', [
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ])
            internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

        submitted = st.form_submit_button('Predict')

    if submitted:
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly],
            'TotalCharges': [total],
            f'Contract_{contract}': [1],
            f'PaymentMethod_{payment}': [1],
            f'InternetService_{internet}': [1]
        })

        # Add missing cols
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model_columns]

        # Predict
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]*100

        if pred == 1:
            st.error(f"⚠️ Likely to churn! (Probability: {prob:.1f}%)")
        else:
            st.success(f"✅ Not likely to churn (Probability: {100 - prob:.1f}%)")

        # Feature importance
        st.subheader("📊 Feature Importance (Top 5)")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'feature': model_columns, 'importance': importances})
        feat_df = feat_df.sort_values('importance', ascending=False).head(5)
        fig, ax = plt.subplots()
        bars = ax.barh(feat_df['feature'], feat_df['importance'], color='#4e79a7')
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        st.pyplot(fig)

st.markdown("---")
st.caption("Built by Sudhakardamarasingi")



st.markdown("---")



