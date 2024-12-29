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
from tensorflow.keras.models import load_model
import os

# Step 1: Fetch OHLCV Data from Kraken
def fetch_ohlcv_data(symbol, timeframe='1d', since=None):
    exchange = ccxt.kraken()  # Change to binanceus() if needed
    ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Step 2: Data Preprocessing
def preprocess_data(data):
    data = data.sort_values('timestamp')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['scaled_close'] = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    
    return data, scaler

# Prepare the dataset for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Step 3: Build the LSTM Model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Save the Model
def save_model(model, filename='crypto_model.h5'):
    model.save(filename)
    print(f'Model saved as {filename}')

# Step 5: Load the Model
def load_existing_model(filename='crypto_model.h5'):
    if os.path.exists(filename):
        model = load_model(filename)
        print(f'Model loaded from {filename}')
        return model
    else:
        print("Model file not found.")
        return None

# Step 6: Visualization
def visualize_results(real_prices, predicted_prices):
    plt.figure(figsize=(14, 5))
    plt.plot(real_prices, color='blue', label='Actual BTC/USD Price')
    plt.plot(predicted_prices, color='red', label='Predicted BTC/USD Price')
    plt.title('BTC/USD Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Accuracy Metrics
    mse = mean_squared_error(real_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_prices, predicted_prices)
    explained_var = explained_variance_score(real_prices, predicted_prices)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Explained Variance Score: {explained_var * 100:.2f}%')

    # Calculate model accuracy
    accuracy = 100 - (rmse / np.mean(real_prices)) * 100
    print(f'Model Accuracy: {accuracy:.2f}%')

    # Heatmap of error distribution
    errors = real_prices - predicted_prices
    plt.figure(figsize=(10, 6))
    sns.heatmap(errors.reshape(-1, 1), annot=False, cmap="coolwarm")
    plt.title('Error Distribution Heatmap')
    plt.show()

    # Scatter plot for actual vs predicted prices
    plt.figure(figsize=(6, 6))
    plt.scatter(real_prices, predicted_prices, c='blue')
    plt.plot(real_prices, real_prices, color='red', linewidth=2)
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.show()

# Main function to run the model
def main():
    # Fetch historical data for BTC/USD
    data = fetch_ohlcv_data('BTC/USD')

    # Data Preprocessing
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

    # Build the model
    model = build_model((X_train.shape[1], 1))

    # Train the model
    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

    # Save the model
    save_model(model)

    # Make predictions
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Visualize the results
    visualize_results(real_prices, predicted_prices)

if __name__ == "__main__":
    main()