# churn.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Page config
st.set_page_config(page_title="📊 Churn Prediction Dashboard", page_icon="📊", layout="wide")

# Custom CSS for pro look
st.markdown("""
    <style>
    body { background-color: #fafafa; }
    .stApp { background-color: #ffffff; }
    .big-font { font-size:22px !important; }
    .metric { font-size: 26px; font-weight: bold; color: #4e79a7; }
    .footer { color: gray; text-align: center; font-size: 14px; margin-top: 50px; }
    </style>
    """, unsafe_allow_html=True)

# Load data & model
@st.cache_data
def load_data():
    return pd.read_csv('churn_dataset.csv')

@st.cache_resource
def load_model():
    with open('advanced_churn_model.pkl', 'rb') as f:
        model, scaler, columns = pickle.load(f)
    return model, scaler, columns

data = load_data()
model, scaler, model_columns = load_model()

# Sidebar branding
st.sidebar.title("💡 Churn Insights")
st.sidebar.caption("Built with ❤️ by Sudhakardamarasingi")

# Header
st.title("✨ Telecom Churn Prediction Dashboard")
st.caption("Analyze churn trends & predict customer churn with a single click.")

# Top KPIs
churn_rate = (data['Churn'].value_counts(normalize=True) * 100).get('Yes', 0)
col1, col2, col3 = st.columns(3)
col1.metric("📉 Overall Churn Rate", f"{churn_rate:.1f}%")
col2.metric("📦 Total Customers", f"{len(data):,}")
col3.metric("💲 Avg Monthly Charges", f"${data['MonthlyCharges'].mean():.2f}")

# Tabs
tab1, tab2 = st.tabs(["📊 EDA & Insights", "🔮 Predict Churn"])

with tab1:
    st.subheader("✅ Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=data, palette=['#FF6B6B','#4ECDC4'], ax=ax)
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.subheader("💳 Churn by Payment Method")
    churn_by_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    churn_by_payment = churn_by_payment.sort_values()
    fig, ax = plt.subplots()
    sns.barplot(x=churn_by_payment, y=churn_by_payment.index, palette='coolwarm', ax=ax)
    ax.set_xlabel('Churn Rate (%)')
    st.pyplot(fig)

    st.subheader("📑 Churn by Contract Type")
    churn_by_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    fig, ax = plt.subplots()
    sns.barplot(x=churn_by_contract.index, y=churn_by_contract.values, palette='viridis', ax=ax)
    ax.set_ylabel('Churn Rate (%)')
    st.pyplot(fig)

    st.markdown("### ✏️ **Insights:**")
    st.markdown("- Highest churn with month-to-month contracts.\n- Electronic check payments have higher churn.\n- Long-term contracts reduce churn risk.")

with tab2:
    st.subheader("🔮 Predict if a customer will churn")
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.slider('Tenure (months)', 0, 100, 12)
            monthly = st.number_input('Monthly Charges', 0.0, 200.0, 70.0)
            total = st.number_input('Total Charges', 0.0, 10000.0, 2500.0)
        with c2:
            contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
            payment = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        predict_btn = st.form_submit_button('✅ Predict Now')

    if predict_btn:
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly],
            'TotalCharges': [total],
            f'Contract_{contract}': [1],
            f'PaymentMethod_{payment}': [1],
            f'InternetService_{internet}': [1]
        })

        # Add missing columns
        for col in model_columns:
            if col not in input_data:
                input_data[col] = 0
        input_data = input_data[model_columns]

        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]*100

        if pred == 1:
            st.error(f"⚠️ Customer likely to churn! (Prob: {prob:.1f}%)")
        else:
            st.success(f"✅ Customer unlikely to churn. (Prob: {100 - prob:.1f}%)")

    st.markdown("---")
    st.subheader("📊 Feature Importance (Top 5)")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'feature': model_columns, 'importance': importances})
    feat_df = feat_df.sort_values('importance', ascending=False).head(5)
    fig, ax = plt.subplots()
    sns.barplot(x='importance', y='feature', data=feat_df, palette='coolwarm', ax=ax)
    st.pyplot(fig)

# Footer
st.markdown("<div class='footer'>Built with Streamlit & ❤️ by Sudhakardamarasingi</div>", unsafe_allow_html=True)



