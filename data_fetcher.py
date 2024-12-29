import ccxt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Fetch historical OHLCV data from Kraken
def fetch_ohlcv_data(symbol, timeframe='1d', since=None):
    exchange = ccxt.kraken()
    ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Preprocess the data (scaling)
def preprocess_data(data):
    data = data.sort_values('timestamp')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['scaled_close'] = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    return data, scaler

# Create LSTM dataset
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
