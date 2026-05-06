from prophet import Prophet

def train_prophet(df):
    df = df.rename(columns={'date': 'ds', 'sales': 'y'})
    model = Prophet()
    model.fit(df)
    return model

def forecast_prophet(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(periods)