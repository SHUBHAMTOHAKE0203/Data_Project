from fastapi import FastAPI
from src.predict import get_prediction

app = FastAPI(title="Sales Forecast API 🚀")

@app.get("/")
def home():
    return {"message": "API Running"}

@app.get("/predict/{state}")
def predict(state: str):
    return get_prediction(state)