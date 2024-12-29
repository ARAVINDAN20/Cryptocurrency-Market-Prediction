# lstm_model.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Function to build and train the LSTM model
def build_and_train_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
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

    accuracy = 100 - (rmse / np.mean(real_prices)) * 100
    st.write(f'Model Accuracy: {accuracy:.2f}%')

    # Error distribution heatmap
    errors = real_prices - predicted_prices
    plt.figure(figsize=(10, 6))
    sns.heatmap(errors.reshape(-1, 1), annot=False, cmap="coolwarm")
    plt.title('Error Distribution Heatmap')
    st.pyplot(plt)

# Function to predict next day’s price
def predict_next_day(model, X, scaler):
    next_day_prediction = model.predict(X[-1].reshape(1, X.shape[1], 1))
    next_day_price = scaler.inverse_transform(next_day_prediction)
    return next_day_price[0][0]
