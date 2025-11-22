<p align="center">
  <img src="banner.svg" alt="Customer Churn Prediction Banner" />
</p>


<p align="center">
  <img src="https://img.shields.io/badge/Project-Customer%20Churn%20Prediction-0f172a?style=for-the-badge" alt="Customer Churn Prediction" />
</p>

<p align="center">
  <a href="https://sudhakardamarasingi.streamlit.app/">
    <img src="https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit App" />
  </a>
  <a href="https://huggingface.co/spaces/Madmax003/churn_prediction2">
    <img src="https://img.shields.io/badge/Live_Space-HuggingFace-ffcc00?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face Space" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-ML%20Model-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Made_by-Sudhakar_Damarasingi-6b21a8?style=flat-square" alt="Author" />
</p>

---

# 📊 Customer Churn Prediction

This project is an **end-to-end Machine Learning application** that predicts whether a customer is likely to **churn (leave a service)** based on historical behavior and account information.

It is designed to help businesses:

- Identify **high-risk customers**
- Take **proactive retention actions**
- Understand **drivers of churn** using model insights

You can try the live app here:

- 🚀 **Streamlit App** → [Customer Churn Prediction](https://sudhakardamarasingi.streamlit.app/)  
- 🤗 **Hugging Face Space** → [Churn Prediction Space](https://huggingface.co/spaces/Madmax003/churn_prediction2)

---

## 🧠 Problem Statement

Customer churn is a critical issue for subscription-based and service-based businesses.  
Acquiring a new customer is often **more expensive** than retaining an existing one.

This project builds a **classification model** that predicts whether a customer will churn based on:

- Demographics  
- Services subscribed  
- Tenure  
- Billing and payment patterns  
- Usage and engagement signals  

---

## 🧱 Project Structure

```bash
.
├── app.py                  # Streamlit / Flask app for serving predictions
├── churn_dataset.csv       # Dataset used for training and evaluation
├── advanced_churn_model.pkl# Trained ML model (serialized)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # License (MIT)
└── .gitignore              # Git ignore rules
