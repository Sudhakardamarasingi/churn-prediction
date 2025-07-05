# churn.py
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# Page config
st.set_page_config(page_title="📊 Churn Dashboard", page_icon="📊", layout="wide")

# Sidebar theme toggle
theme = st.sidebar.selectbox("🎨 Choose Theme", ["Light", "Dark"])

if theme == "Dark":
    primaryColor = "#0d1117"
    textColor = "#f0f6fc"
    bgColor = "#161b22"
else:
    primaryColor = "#ffffff"
    textColor = "#000000"
    bgColor = "#f9f9f9"

# Custom CSS
st.markdown(f"""
<style>
body {{
    background-color: {bgColor};
    color: {textColor};
}}
.stApp {{
    background-color: {bgColor};
}}
.big-title {{
    font-size:30px !important;
    font-weight:bold;
}}
.metric-card {{
    background-color: #ffffff;
    padding:20px;
    border-radius:12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    text-align:center;
}}
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

# Header
st.markdown("<div class='big-title'>📊 Telecom Churn Prediction Dashboard</div>", unsafe_allow_html=True)
st.caption("Professional dashboard to explore churn trends and predict churn risk.")

# Metrics
churn_rate = (data['Churn'].value_counts(normalize=True) * 100).get('Yes', 0)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><h3>📉 Churn Rate</h3><h2>{churn_rate:.1f}%</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h3>👥 Total Customers</h3><h2>{len(data):,}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h3>💲 Avg Monthly</h3><h2>${data['MonthlyCharges'].mean():.2f}</h2></div>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📈 EDA & Insights", "🔮 Predict Churn"])

with tab1:
    st.subheader("✅ Churn Distribution")
    fig1 = px.histogram(data, x='Churn', color='Churn', color_discrete_sequence=['#FF6B6B','#4ECDC4'])
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("💳 Churn by Payment Method")
    churn_rate_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    fig2 = px.bar(churn_rate_payment.sort_values(), orientation='h', color=churn_rate_payment,
                  color_continuous_scale='blues', labels={'value':'Churn Rate (%)'}, title="Churn by Payment Method")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📑 Churn by Contract Type")
    churn_rate_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    fig3 = px.bar(churn_rate_contract, x=churn_rate_contract.index, y=churn_rate_contract.values,
                  color=churn_rate_contract.values, color_continuous_scale='teal',
                  labels={'y':'Churn Rate (%)'}, title="Churn by Contract Type")
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("🔮 Predict if customer will churn")
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
        btn = st.form_submit_button("✅ Predict")

    if btn:
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
        prob = model.predict_proba(scaler.transform(input_df))[0][1]*100
        if pred == 1:
            st.error(f"⚠️ Likely to churn! (Prob: {prob:.1f}%)")
        else:
            st.success(f"✅ Not likely to churn (Prob: {100 - prob:.1f}%)")

# Footer
st.markdown(f"<div style='text-align:center; color:gray; margin-top:40px;'>Built with ❤️ by Sudhakardamarasingi</div>", unsafe_allow_html=True)

