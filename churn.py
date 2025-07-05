# churn.py
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# Page config
st.set_page_config(page_title="📊 Telecom Churn Dashboard", page_icon="📊", layout="wide")

# Sidebar theme toggle
theme = st.sidebar.radio("🎨 Choose Theme", ["Light", "Dark"])

# Theme colors
if theme == "Dark":
    primary_bg = "#0d1117"
    card_bg = "#161b22"
    text_color = "#f0f6fc"
    accent_color = "#3b82f6"
else:
    primary_bg = "#ffffff"
    card_bg = "#f9f9f9"
    text_color = "#000000"
    accent_color = "#2563eb"

# Custom CSS
st.markdown(f"""
<style>
body, .stApp {{
    background-color: {primary_bg};
    color: {text_color};
}}
.big-title {{
    font-size:32px !important;
    font-weight:bold;
    color: {accent_color};
}}
.metric-card {{
    background-color: {card_bg};
    padding:20px;
    border-radius:12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
    text-align:center;
}}
.footer {{
    color: gray;
    text-align: center;
    font-size: 13px;
    margin-top: 40px;
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

# Hero header
st.markdown(f"<div class='big-title'>📊 Telecom Customer Churn Dashboard</div>", unsafe_allow_html=True)
st.caption("Explore churn trends & predict risk — modern light/dark dashboard.")

# Metrics
churn_rate = (data['Churn'].value_counts(normalize=True) * 100).get('Yes', 0)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><h4>📉 Churn Rate</h4><h2>{churn_rate:.1f}%</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h4>👥 Total Customers</h4><h2>{len(data):,}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h4>💲 Avg Monthly</h4><h2>${data['MonthlyCharges'].mean():.2f}</h2></div>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📊 EDA & Insights", "🔮 Predict Churn"])

with tab1:
    st.subheader("✅ Churn Distribution")
    fig1 = px.histogram(data, x='Churn', color='Churn',
                        color_discrete_sequence=['#FF6B6B','#4ECDC4'],
                        title="Churn Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("💳 Churn by Payment Method")
    churn_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    fig2 = px.bar(churn_payment.sort_values(), orientation='h', color=churn_payment,
                  color_continuous_scale='blues', title="Churn by Payment Method")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📑 Churn by Contract Type")
    churn_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    fig3 = px.bar(x=churn_contract.index, y=churn_contract.values, color=churn_contract.values,
                  color_continuous_scale='teal', labels={'y':'Churn Rate (%)'}, title="Churn by Contract")
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
        btn = st.form_submit_button("✅ Predict Now")

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
st.markdown("<div class='footer'>Built with ❤️ by Sudhakardamarasingi</div>", unsafe_allow_html=True)

