import numpy as np
from sklearn.linear_model import LinearRegression

def train_predict_model(df):
    df = df[['Close']].copy()
    df['Prediction'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = np.array(df[['Close']])
    y = np.array(df['Prediction'])

    model = LinearRegression()
    model.fit(X, y)

    last_price = np.array(df[['Close']].tail(1))
    predicted_price = model.predict(last_price)

    return predicted_price[0]