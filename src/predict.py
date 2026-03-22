import joblib
import pandas as pd

def predict(data: dict):
    model = joblib.load("models/model.pkl")
    df = pd.DataFrame([data])
    return model.predict(df)[0]