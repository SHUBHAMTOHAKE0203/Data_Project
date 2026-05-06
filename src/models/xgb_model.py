from xgboost import XGBRegressor

FEATURE_COLS = [
    'lag_1','lag_7','lag_30',
    'rolling_mean','rolling_std',
    'day_of_week','month','is_holiday'
]

def train_xgb(df):
    X = df[FEATURE_COLS]
    y = df['sales']
    
    model = XGBRegressor(n_estimators=100)
    model.fit(X, y)
    
    return model

def predict_xgb(model, df):
    return model.predict(df[FEATURE_COLS])