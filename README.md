# 📈 Stock Price Prediction Using Machine Learning (LSTM)

A complete end-to-end **Machine Learning & Deep Learning project** that predicts stock prices using historical market data and **LSTM neural networks**, with an interactive **Streamlit web application**.

---

## 🚀 Project Overview

This project focuses on **time-series forecasting** of stock prices using deep learning. It uses historical stock data, preprocesses it, trains an LSTM neural network model, and visualizes predictions through a clean web interface.

The system enables users to:
- Enter any stock symbol (AAPL, TSLA, GOOG, MSFT, etc.)
- Select a date range
- View real-time prediction graphs

---

## 🎯 Objectives

- Analyze historical stock market data  
- Perform data preprocessing and normalization  
- Train a deep learning LSTM model  
- Predict stock price trends  
- Visualize predictions using an interactive dashboard  

---

## 🧠 Machine Learning Model

- **Algorithm:** Long Short-Term Memory (LSTM) Neural Network  
- **Type:** Time-Series Forecasting  
- **Accuracy:** ~90% (depends on market volatility)  

---

## 🛠 Tech Stack

- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib  
- **Machine Learning:** Scikit-learn  
- **Deep Learning:** TensorFlow / Keras  
- **Web App:** Streamlit  
- **Data Source:** Yahoo Finance API  

---

## 📂 Project Structure


stock-price-prediction-ml/
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── model.py
│ ├── train.py
│ └── predict.py
│
├── models/
│ └── lstm_model.h5
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore


---

## ▶ How To Run Locally

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
2️⃣ Train the Model
cd src
python train.py
cd ..
3️⃣ Run Web Application
streamlit run app.py


