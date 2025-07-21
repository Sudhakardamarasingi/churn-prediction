# Updated churn.py with improved UI
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# Page config
st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📊", layout="wide")

# Load data & model
@st.cache_data
def load_data():
    return pd.read_csv('churn_dataset.csv')

@st.cache_resource
def load_model():
    with open('advanced_churn_model.pkl', 'rb') as f:
        model, scaler, columns = pickle.load(f)
    return model, scaler, columns

# Data & model
data = load_data()
model, scaler, model_columns = load_model()

# Tabs for layout
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Insights", "ℹ️ About"])

# ----- PREDICTION TAB -----
with tab1:
    st.markdown("## 🔮 Customer Churn Prediction")
    st.write("""
    Fill in the customer's information below and click **Predict** to see their churn risk level.
    """)

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            tenure = st.slider("Tenure (months)", 0, 100, 12, help="Number of months the customer has stayed")
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, help="Monthly subscription fee")
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, 2500.0, help="Total billing to date")

        with c2:
            contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
            payment = st.selectbox("Payment Method", [
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

        predict_btn = st.form_submit_button("🚀 Predict Churn Risk")

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
        prob = model.predict_proba(scaler.transform(input_df))[0][1]*100

        st.markdown("### 📊 Prediction Result")
        st.progress(int(prob))
        if prob > 70:
            st.error(f"⚠ High Risk of Churn — {prob:.1f}%")
        elif prob > 40:
            st.warning(f"⚠ Medium Risk of Churn — {prob:.1f}%")
        else:
            st.success(f"✅ Low Risk of Churn — {prob:.1f}%")

        csv = input_df.to_csv(index=False)
        st.download_button("📥 Download Prediction Data", data=csv, file_name="prediction.csv")


# ----- INSIGHTS TAB -----
with tab2:
    st.markdown("## 📊 Data Insights & EDA")

    st.subheader("✅ Churn Distribution")
    fig1 = px.histogram(data, x='Churn', color='Churn', color_discrete_sequence=['#FF6B6B','#4ECDC4'])
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("💳 Churn by Payment Method")
    churn_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    fig2 = px.bar(churn_payment.sort_values(), orientation='h', color=churn_payment, color_continuous_scale='blues')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📑 Churn by Contract Type")
    churn_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    fig3 = px.bar(x=churn_contract.index, y=churn_contract.values, color=churn_contract.values, color_continuous_scale='teal')
    st.plotly_chart(fig3, use_container_width=True)


# ----- ABOUT TAB -----
with tab3:
    st.markdown("## ℹ️ About This App")
    st.write("""
    This churn prediction app was developed by **Sudhakardamarasingi** using:
    - **Streamlit** for frontend UI
    - **scikit-learn** for machine learning
    - **Plotly** for data visualizations
    
    The model predicts churn based on contract type, payment method, internet service, and billing details.
    """)
    st.markdown("---")
    st.markdown("[📂 GitHub Repo](https://github.com/Sudhakardamarasingi/churn-prediction)")
    st.markdown("[🚀 Streamlit App](https://sudhakardamarasingi.streamlit.app/)")
