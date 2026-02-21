import os
from data_loader import load_stock_data
from preprocessing import scale_data, create_sequences
from model import build_lstm

TIME_STEP = 60

df = load_stock_data("AAPL", "2015-01-01", "2025-01-01")
data = df[['Close']]

scaled, scaler = scale_data(data)

X, y = create_sequences(scaled, TIME_STEP)
X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = build_lstm((TIME_STEP, 1))

model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))

os.makedirs("../models", exist_ok=True)
model.save("../models/lstm_model.h5")

print("✅ Model trained and saved successfully")