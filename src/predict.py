import numpy as np
from tensorflow.keras.models import load_model

def predict(model_path, X):
    model = load_model(model_path)
    return model.predict(X)