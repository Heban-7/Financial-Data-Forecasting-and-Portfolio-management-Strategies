# LSTM Forecasting 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def LSTM_model(data, asset):
    data = data['Close']
    # Prepare data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[asset].values.reshape(-1,1))

    # Create sequences
    def create_sequences(data, window=60):
        X, y = [], []
        for i in range(len(data)-window):
            X.append(data[i:i+window])
            y.append(data[i+window])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data)
    X_train, X_test = X[:-60], X[-60:]
    y_train, y_test = y[:-60], y[-60:]

    # Build LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train
    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.2, verbose=0)

    # Plot training loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Training Progress')
    plt.legend()
    plt.show()

    # Predict
    lstm_forecast = model.predict(X_test)
    lstm_forecast = scaler.inverse_transform(lstm_forecast).flatten()

    # Plot LSTM results
    plt.figure(figsize=(12,6))
    plt.plot(data[asset].index[-60:], data[asset][-60:], label='Actual')
    plt.plot(data[asset].index[-60:], lstm_forecast, label='LSTM Forecast', linestyle='--')
    plt.title(f'{asset} Stock Price Forecast (LSTM)')
    plt.legend()
    plt.show()
