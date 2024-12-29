import streamlit as st
import pandas as pd
from lstm_model import build_and_train_model, visualize_results, predict_next_day
from data_fetcher import fetch_ohlcv_data, preprocess_data, create_dataset
import ccxt


def main():
    st.title("Cryptocurrency Price Prediction with LSTM")
    st.write("Predict future cryptocurrency prices using real-time data from Kraken.")

    # Initialize Kraken exchange
    exchange = ccxt.kraken()
    st.write("Fetching available symbols from Kraken...")

    # Get trading pairs with USD
    symbols = [symbol for symbol in exchange.load_markets() if '/' in symbol and 'USD' in symbol]

    # Select symbol via dropdown
    selected_symbol = st.selectbox("Select Cryptocurrency Symbol:", symbols)

    # Option to upload CSV
    uploaded_file = st.file_uploader("Or upload a CSV file with OHLCV data", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(data.head())
    else:
        # Fetch historical data for selected symbol
        st.write(f"Fetching historical data for {selected_symbol}...")
        data = fetch_ohlcv_data(selected_symbol)
        st.write("Data Preview:")
        st.write(data.head())

    # Preprocess the data
    data, scaler = preprocess_data(data)

    # Prepare dataset for LSTM
    time_step = 60
    scaled_data = data['scaled_close'].values.reshape(-1, 1)
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train-test split
    split_ratio = 0.8
    split = int(len(X) * split_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train LSTM model
    st.write("Training the LSTM model...")
    model, history = build_and_train_model(X_train, y_train, X_test, y_test)

    # Make predictions
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Visualize results
    st.write("Visualizing results...")
    visualize_results(real_prices, predicted_prices)

    # Predict next day's price
    next_day_price = predict_next_day(model, X[-1], scaler)
    st.write(f"Predicted Price for the next day: ${next_day_price:.2f}")


if __name__ == "__main__":
    main()
