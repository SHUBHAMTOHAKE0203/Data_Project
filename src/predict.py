import joblib

from src.models.arima_model import forecast_arima
from src.models.prophet_model import forecast_prophet
from src.models.xgb_model import predict_xgb


def get_prediction(state):
    model_info = joblib.load(f"models/{state}.pkl")

    model_name = model_info["model_name"]
    model = model_info["model"]
    last_data = model_info["last_data"]

    # ================= REAL FORECAST =================
    if model_name == "arima":
        forecast = forecast_arima(model, 8)

    elif model_name == "prophet":
        forecast_df = forecast_prophet(model, 8)
        forecast = forecast_df['yhat'].tail(8)

    elif model_name == "xgb":
        forecast = predict_xgb(model, last_data.tail(8))

    else:
        forecast = []

    # convert to normal list (important for JSON)
    forecast = list(map(float, forecast))

    return {
        "state": state,
        "model_used": model_name,
        "forecast": forecast
    }