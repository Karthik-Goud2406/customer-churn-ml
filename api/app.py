
from src.data_preprocessing import preprocess
from fastapi import FastAPI
import joblib
import pandas as pd
from src.feature_engineering import create_features

app = FastAPI()

model = joblib.load("models/model.pkl")
columns = joblib.load("models/columns.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    df = create_features(df)

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]

    return {"churn": int(prediction)}