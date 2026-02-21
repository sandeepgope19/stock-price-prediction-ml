import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from model import train_predict_model
from sentiment import get_news_sentiment

st.set_page_config(page_title="AI Stock Predictor", layout="wide")

st.title("🤖 AI-Powered Real-Time Stock Prediction System")

stock_dict = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Microsoft": "MSFT",
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS"
}

company = st.selectbox("Select Company:", list(stock_dict.keys()))
stock = stock_dict[company]

start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

if st.button("Analyze & Predict"):
    data = yf.download(stock, start=start_date, end=end_date)

    st.subheader("📊 Stock Data")
    st.dataframe(data.tail())

    predicted_price = train_predict_model(data)
    st.success(f"📌 Predicted Next Day Closing Price: {predicted_price:.2f}")

    st.subheader("📰 News Sentiment Analysis")
    sentiment, headlines = get_news_sentiment(company)

    if sentiment > 0:
        st.success(f"Positive Market Sentiment 😊  (Score: {sentiment:.2f})")
    elif sentiment < 0:
        st.error(f"Negative Market Sentiment 😟 (Score: {sentiment:.2f})")
    else:
        st.warning("Neutral Market Sentiment 😐")

    for news in headlines[:5]:
        st.write("•", news)

    st.subheader("📈 Stock Trend")
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label="Closing Price")
    ax.legend()
    st.pyplot(fig)