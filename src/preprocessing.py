import pandas as pd

def load_data(path):
    df = pd.read_excel(path)
    df.columns = ['state', 'date', 'sales', 'category']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['state', 'date'])
    return df


def handle_missing(df):
    all_states = []

    for state in df['state'].unique():
        temp = df[df['state'] == state].copy()
        temp = temp.set_index('date').asfreq('D')

        temp['state'] = state
        temp['sales'] = temp['sales'].ffill()

        all_states.append(temp)

    df = pd.concat(all_states).reset_index()
    return df