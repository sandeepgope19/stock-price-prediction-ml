import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_stock_data
from src.preprocessing import scale_data, create_sequences
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("📈 Stock Price Prediction Using LSTM")

symbol = st.text_input("Enter Stock Symbol:", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))

if st.button("Predict"):

    df = load_stock_data(symbol, start_date, end_date)
    data = df[['Close']]

    scaled, scaler = scale_data(data)
    X, y = create_sequences(scaled, 60)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = load_model("models/lstm_model.h5")
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    valid = data.iloc[60:]
    valid['Predicted'] = predictions

    st.subheader("📊 Stock Price Prediction Graph")

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data['Close'], label="Actual Price")
    ax.plot(valid['Predicted'], label="Predicted Price")
    ax.legend()
    st.pyplot(fig)