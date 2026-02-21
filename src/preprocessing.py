import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(data)
    return scaled, scaler

def create_sequences(dataset, step=60):
    X, y = [], []
    for i in range(len(dataset) - step):
        X.append(dataset[i:i+step, 0])
        y.append(dataset[i+step, 0])
    return np.array(X), np.array(y)
