import pandas as pd
import holidays

def create_features(df):
    df = df.copy()

    # ================= FIX 1: Ensure datetime =================
    df['date'] = pd.to_datetime(df['date'])

    # ================= FIX 2: Sort properly =================
    df = df.sort_values(['state', 'date'])

    # ================= TIME FEATURES =================
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # ================= HOLIDAY FLAG =================
    us_holidays = holidays.US()   
    df['is_holiday'] = df['date'].isin(us_holidays).astype(int)

    # ================= LAG FEATURES =================
    df['lag_1'] = df.groupby('state')['sales'].shift(1)
    df['lag_7'] = df.groupby('state')['sales'].shift(7)
    df['lag_30'] = df.groupby('state')['sales'].shift(30)

    # ================= ROLLING FEATURES (FIXED) =================
    df['rolling_mean_7'] = df.groupby('state')['sales'].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    df['rolling_std_7'] = df.groupby('state')['sales'].transform(
        lambda x: x.shift(1).rolling(7).std()
    )

    # ================= DROP NA =================
    df = df.dropna()

    return df