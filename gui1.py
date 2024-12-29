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
import streamlit as st

# Function to fetch OHLCV data from Kraken
def fetch_ohlcv_data(symbol, timeframe='1d', since=None):
    exchange = ccxt.binanceus()
    ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Function to preprocess the data
def preprocess_data(data):
    data = data.sort_values('timestamp')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['scaled_close'] = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    return data, scaler

# Function to create the dataset for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Function to build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to visualize results
def visualize_results(real_prices, predicted_prices):
    plt.figure(figsize=(14, 5))
    plt.plot(real_prices, color='blue', label='Actual Price')
    plt.plot(predicted_prices, color='red', label='Predicted Price')
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    # Calculate accuracy metrics
    mse = mean_squared_error(real_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_prices, predicted_prices)
    explained_var = explained_variance_score(real_prices, predicted_prices)

    st.write(f'Mean Squared Error (MSE): {mse:.4f}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    st.write(f'Mean Absolute Error (MAE): {mae:.4f}')
    st.write(f'Explained Variance Score: {explained_var * 100:.2f}%')

    # Let's assume the "accuracy" as the complement of the normalized RMSE
    accuracy = 100 - (rmse / np.mean(real_prices)) * 100
    st.write(f'Model Accuracy: {accuracy:.2f}%')

    # Heatmap of error distribution
    errors = real_prices - predicted_prices
    plt.figure(figsize=(10, 6))
    sns.heatmap(errors.reshape(-1, 1), annot=False, cmap="coolwarm")
    plt.title('Error Distribution Heatmap')
    st.pyplot(plt)

# Streamlit application
def main():
    st.title("Cryptocurrency Price Prediction")
    st.write("This application predicts cryptocurrency prices using LSTM neural networks.")

    # Dropdown for selecting cryptocurrency symbols
    exchange = ccxt.binanceus()
    markets = exchange.load_markets()
    crypto_symbols = [market for market in markets if '/' in market and 'USD' in market]
    
    selected_symbol = st.selectbox("Select Cryptocurrency Symbol:", crypto_symbols)

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file with OHLCV data (optional)", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())
    else:
        # Fetch default data for the selected cryptocurrency
        st.write(f"Fetching data for {selected_symbol}...")
        data = fetch_ohlcv_data(selected_symbol)
        st.write("Data Preview:")
        st.write(data.head())

    # Preprocess the data
    data, scaler = preprocess_data(data)

    # Prepare the dataset
    scaled_data = data['scaled_close'].values.reshape(-1, 1)
    X, y = create_dataset(scaled_data)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split into train and test sets
    split_ratio = 0.8
    split = int(len(X) * split_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train the model
    model = build_model((X_train.shape[1], 1))
    st.write("Training the model...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Make predictions
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Visualize results
    visualize_results(real_prices, predicted_prices)

if __name__ == "__main__":
    main()