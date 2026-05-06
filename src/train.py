import os
import joblib
import pandas as pd

from src.preprocessing import load_data, handle_missing
from src.features import create_features
from src.evaluate import evaluate

from src.models.arima_model import train_arima, forecast_arima
from src.models.prophet_model import train_prophet, forecast_prophet
from src.models.xgb_model import train_xgb, predict_xgb
from src.models.lstm_model import train_lstm

DATA_PATH = "data/sales.xlsx"
MODEL_DIR = "models/"


def train_pipeline():
    # ✅ Ensure models folder exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    df = handle_missing(df)

    states = df['state'].unique()

    for state in states:
        print(f"Training for {state}...")

        state_df = df[df['state'] == state].copy()

        # Feature engineering
        feat_df = create_features(state_df)

        # Time-based split
        train = feat_df.iloc[:-56]
        test = feat_df.iloc[-56:]

        y_true = test['sales']

        scores = {}
        models = {}

        # ================= ARIMA =================
        try:
            arima_model = train_arima(train['sales'])
            arima_pred = forecast_arima(arima_model, 56)

            scores['arima'] = evaluate(y_true, arima_pred)
            models['arima'] = arima_model
        except Exception as e:
            print("ARIMA failed:", e)
            scores['arima'] = float('inf')

        # ================= PROPHET =================
        try:
            prophet_model = train_prophet(train[['date', 'sales']])
            prophet_pred = forecast_prophet(prophet_model, 56)['yhat']

            scores['prophet'] = evaluate(y_true, prophet_pred)
            models['prophet'] = prophet_model
        except Exception as e:
            print("Prophet failed:", e)
            scores['prophet'] = float('inf')

        # ================= XGBOOST =================
        try:
            xgb_model = train_xgb(train)
            xgb_pred = predict_xgb(xgb_model, test)

            scores['xgb'] = evaluate(y_true, xgb_pred)
            models['xgb'] = xgb_model
        except Exception as e:
            print("XGB failed:", e)
            scores['xgb'] = float('inf')

        # ================= LSTM =================
        try:
            lstm_model, scaler = train_lstm(train['sales'])

            # Not fully evaluated → assign high error
            scores['lstm'] = float('inf')
            models['lstm'] = (lstm_model, scaler)
        except Exception as e:
            print("LSTM failed:", e)
            scores['lstm'] = float('inf')

        # ================= BEST MODEL =================
        best_model_name = min(scores, key=scores.get)
        best_model = models.get(best_model_name)

        print(f"Best model for {state}: {best_model_name}")

        # ✅ Save EVERYTHING needed for prediction
        save_object = {
            "model_name": best_model_name,
            "model": best_model,
            "last_data": state_df.tail(100)  # needed for future prediction
        }

        joblib.dump(save_object, f"{MODEL_DIR}/{state}.pkl")

    print("\nTraining complete ✅")


if __name__ == "__main__":
    train_pipeline()