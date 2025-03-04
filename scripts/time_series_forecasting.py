import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_train_test_split(asset_returns):
    # Prepare training data (use 90% for training)
    train_size = int(len(asset_returns) * 0.9)
    train_data = asset_returns.iloc[:train_size]
    test_data = asset_returns.iloc[train_size:]
    return train_data, test_data


class ARIMATimeSeriesModel():
    def __init__(self):
        self.p, self.d, self.q = 1, 0, 1

    def arima_model_trainning(self, train_data):
        model = ARIMA(train_data = train_data, order=(1,0,1))
        self.model = model.fit()
        print(self.model.summary())

    def evaluation(self, test_data):
        forecast_len = len(test_data)
        self.forecast = self.model_fit.forecast(steps = forecast_len)
        self.mae = mean_absolute_error(test_data, self.forecast)
        self.mse = mean_squared_error(test_data, self.forecast)
        print(f"\nModel Evaluation on Test Data:")
        print(f"Mean Absolute Error (MAE): {self.mae:.6f}")
        print(f"Root Mean Squared Error (RMSE): {self.rmse:.6f}")

    def plot_forecast(self, test_data, forecast_asset):
        # Plot actual vs forecast returns
        plt.figure(figsize=(15, 8))
        plt.plot(test_data.index, test_data, label='Actual Returns', color='blue')
        plt.plot(test_data.index, self.forecast, label='Forecasted Returns', color='red', linestyle='--')
        plt.title(f'{forecast_asset} - Actual vs Forecasted Returns')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.tight_layout()

    def future_forecast(self, forecast_asset, asset_returns):
        # Generate and plot future forecasts
        future_steps = 30  # Forecast 30 days ahead
        future_forecast = self.model_fit.forecast(steps=future_steps)
        future_dates = pd.date_range(start=asset_returns.index[-1], periods=future_steps)

        plt.figure(figsize=(15, 8))
        plt.plot(asset_returns.index[-90:], asset_returns.iloc[-90:], label='Historical Returns', color='blue')
        plt.plot(future_dates, future_forecast, label='Future Returns Forecast', color='red', linestyle='--')
        plt.title(f'{forecast_asset} - Returns Forecast for Next {future_steps} Trading Days')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.axvline(x=asset_returns.index[-1], color='green', linestyle='-', label='Forecast Start')
        plt.legend()
        plt.tight_layout()



