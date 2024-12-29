import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tkinter as tk
from tkinter import messagebox

# Function to fetch OHLCV data
def fetch_ohlcv_data(symbol, timeframe='1d', since=None):
    exchange = ccxt.kraken()  # Change to binanceus() if needed
    ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Function to create dataset for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Function to train the model
def train_model(symbol):
    try:
        # Fetch historical data
        data = fetch_ohlcv_data(symbol)
        data = data.sort_values('timestamp')
        
        # Normalize the 'close' price
        scaler = MinMaxScaler(feature_range=(0, 1))
        data['scaled_close'] = scaler.fit_transform(data['close'].values.reshape(-1, 1))
        
        # Prepare the dataset
        scaled_data = data['scaled_close'].values.reshape(-1, 1)
        X, y = create_dataset(scaled_data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split into train and test sets
        split_ratio = 0.8
        split = int(len(X) * split_ratio)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Building the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # Make predictions
        predicted_prices = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate accuracy metrics
        mse = mean_squared_error(real_prices, predicted_prices)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(real_prices, predicted_prices)
        explained_var = explained_variance_score(real_prices, predicted_prices)

        # Display results
        messagebox.showinfo("Training Complete", f"Model trained for {symbol}!\n"
                                                  f"MSE: {mse:.4f}\n"
                                                  f"RMSE: {rmse:.4f}\n"
                                                  f"MAE: {mae:.4f}\n"
                                                  f"Explained Variance: {explained_var * 100:.2f}%")

        # Visualization
        plt.figure(figsize=(14, 5))
        plt.plot(real_prices, color='blue', label='Actual Price')
        plt.plot(predicted_prices, color='red', label='Predicted Price')
        plt.title(f'{symbol} Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to create the GUI
def create_gui():
    root = tk.Tk()
    root.title("Cryptocurrency Price Prediction")

    tk.Label(root, text="Enter Cryptocurrency Symbol (e.g., BTC/USD):").pack(pady=10)
    symbol_entry = tk.Entry(root)
    symbol_entry.pack(pady=10)

    tk.Button(root, text="Train Model", command=lambda: train_model(symbol_entry.get())).pack(pady=20)

    root.mainloop()

# Run the GUI
create_gui()