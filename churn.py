# Original churn.py restored based on user feedback
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# Page config
st.set_page_config(page_title="Telecom Churn Dashboard", page_icon="", layout="wide")

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
with st.sidebar:
    st.markdown("Sudhakardamarasingi")
    st.markdown("Customer Churn Prediction App")
    st.markdown("[View on GitHub](https://github.com/Sudhakardamarasingi/churn-prediction)")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Insights"])

# Header
st.markdown("<h1 style='color:#00e1ff;'>Telecom Customer Churn Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div style='color:#9ca3af;'>Understand why customers churn & predict risk instantly.</div>", unsafe_allow_html=True)

# Metrics
churn_rate = (data['Churn'].value_counts(normalize=True) * 100).get('Yes', 0)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Churn Rate", value=f"{churn_rate:.1f}%")
with col2:
    st.metric(label="Total Customers", value=f"{len(data):,}")
with col3:
    st.metric(label="Avg Monthly Charge", value=f"${data['MonthlyCharges'].mean():.2f}")

# Pages
if page == "🏠 Home":
    st.subheader("🔮 Predict if customer will churn")
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 100, 12)
            monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
            total = st.number_input("Total Charges", 0.0, 10000.0, 2500.0)
        with c2:
            contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
            payment = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

        predict_btn = st.form_submit_button("🚀 Predict")

    if predict_btn:
        input_df = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly],
            'TotalCharges': [total],
            f'Contract_{contract}': [1],
            f'PaymentMethod_{payment}': [1],
            f'InternetService_{internet}': [1]
        })
        for col in model_columns:
            if col not in input_df:
                input_df[col] = 0
        input_df = input_df[model_columns]

        pred = model.predict(scaler.transform(input_df))[0]
        prob = model.predict_proba(scaler.transform(input_df))[0][1] * 100

        if prob > 70:
            st.error(f"⚠ High churn risk: {prob:.1f}%")
        elif prob > 40:
            st.warning(f"⚠ Medium churn risk: {prob:.1f}%")
        else:
            st.success(f"✅ Low churn risk: {prob:.1f}%")

elif page == "📊 Insights":
    st.subheader("📊 Data Insights")

    st.subheader("Churn Distribution")
    fig1 = px.histogram(data, x='Churn', color='Churn', color_discrete_sequence=['#FF6B6B','#4ECDC4'])
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Churn by Payment Method")
    churn_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    fig2 = px.bar(churn_payment.sort_values(), orientation='h', color=churn_payment, color_continuous_scale='blues')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Churn by Contract Type")
    churn_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    fig3 = px.bar(x=churn_contract.index, y=churn_contract.values, color=churn_contract.values, color_continuous_scale='teal')
    st.plotly_chart(fig3, use_container_width=True)

# Footer
st.markdown("<div style='text-align:center; color:gray; font-size:13px;'>Developed by Sudhakardamarasingi | <a href='https://github.com/Sudhakardamarasingi' style='color:#9ca3af;'>GitHub</a></div>", unsafe_allow_html=True)
