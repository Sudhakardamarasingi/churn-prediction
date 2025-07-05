# Sidebar navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Insights"])

if page == "🏠 Home":
    st.subheader("🔮 Predict if customer will churn")
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='pred-card'>📅 Tenure (months)</div>", unsafe_allow_html=True)
            tenure = st.slider('', 0, 100, 12)

            st.markdown("<div class='pred-card'>💰 Monthly Charges</div>", unsafe_allow_html=True)
            monthly = st.number_input('', 0.0, 200.0, 70.0)

            st.markdown("<div class='pred-card'>💵 Total Charges</div>", unsafe_allow_html=True)
            total = st.number_input('', 0.0, 10000.0, 2500.0)
        with c2:
            st.markdown("<div class='pred-card'>📄 Contract Type</div>", unsafe_allow_html=True)
            contract = st.selectbox('', ['Month-to-month', 'One year', 'Two year'])

            st.markdown("<div class='pred-card'>💳 Payment Method</div>", unsafe_allow_html=True)
            payment = st.selectbox('', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

            st.markdown("<div class='pred-card'>🌐 Internet Service</div>", unsafe_allow_html=True)
            internet = st.selectbox('', ['DSL', 'Fiber optic', 'No'])

        predict_btn = st.form_submit_button("🚀 Predict Customer Churn Risk")  # bigger, bolder text

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

        st.markdown("<div class='result-card'><h4>📊 Prediction Result</h4>", unsafe_allow_html=True)
        if prob > 70:
            st.markdown(f"<div class='result-card'>⚠ **High churn risk!** Estimated risk: **{prob:.1f}%**.<br>"
                        f"👉 Customer likely to churn. Consider loyalty discount or proactive contact.</div>", unsafe_allow_html=True)
        elif prob > 40:
            st.markdown(f"<div class='result-card'>⚠ **Medium churn risk**: **{prob:.1f}%**.<br>"
                        f"👉 Consider engagement strategies.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-card'>✅ **Low churn risk**: **{prob:.1f}%**.<br>"
                        f"Customer likely to stay. Continue current retention approach.</div>", unsafe_allow_html=True)

elif page == "📊 Insights":
    st.subheader("📊 Data Insights & EDA")
    st.subheader("✅ Churn Distribution")
    fig1 = px.histogram(data, x='Churn', color='Churn', color_discrete_sequence=['#FF6B6B','#4ECDC4'])
    fig1.update_layout(paper_bgcolor=primary_bg, plot_bgcolor=primary_bg, font_color=text_color)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("💳 Churn by Payment Method")
    churn_payment = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    fig2 = px.bar(churn_payment.sort_values(), orientation='h', color=churn_payment, color_continuous_scale='blues')
    fig2.update_layout(paper_bgcolor=primary_bg, plot_bgcolor=primary_bg, font_color=text_color)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📑 Churn by Contract Type")
    churn_contract = data.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()['Yes']*100
    fig3 = px.bar(x=churn_contract.index, y=churn_contract.values, color=churn_contract.values, color_continuous_scale='teal')
    fig3.update_layout(paper_bgcolor=primary_bg, plot_bgcolor=primary_bg, font_color=text_color)
    st.plotly_chart(fig3, use_container_width=True)
