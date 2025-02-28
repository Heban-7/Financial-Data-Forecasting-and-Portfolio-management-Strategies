import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import os, sys

tickers = ["TSLA", "BND", "SPY"]

# Plot the Close price trends over time
def close_price_trends(close_prices, tickers):
    plt.figure(figsize=(15, 10))
    for ticker in tickers:
        plt.plot(close_prices.index, close_prices[ticker]/close_prices[ticker].iloc[0], label=ticker)
    plt.title('Normalized Price Trends')
    plt.ylabel('Normalized Price')
    plt.xlabel('Date')
    plt.legend()
    plt.tight_layout()

# Plote daily change percentage
def daily_returns(close_prices, tickers):
    # Calculate daily returns
    returns = close_prices.pct_change().dropna()
    plt.figure(figsize=(15,10))
    for ticker in tickers:
        plt.plot(returns.index, returns[ticker], label=ticker)
    plt.title("Daily Percentage Change")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.tight_layout()

# rolling mean and standard devation for volatility analysis
def rolling_statistics(close_prices, tickers):
    daily_returns= close_prices.pct_change().dropna()
    rolling_window = 30
    plt.figure(figsize=(14, 6))
    for ticker in tickers:
        rolling_mean = daily_returns[ticker].rolling(rolling_window).mean()
        rolling_std = daily_returns[ticker].rolling(rolling_window).std()

        # Plot rolling statistics
        plt.plot(rolling_mean, label=f"Rolling Mean of {ticker}")
        plt.plot(rolling_std, label=f"Rolling Std Dev of {ticker}")
        plt.title("Rolling Mean and Standard Deviation")
    plt.legend()
    plt.show()

# Seasonality and Trend Analysis
def seasonality_trend_analysis(close_prices, tickers):
    for ticker in tickers:
        print(f"\nDecomposing Time Series for {ticker}")
        result = seasonal_decompose(close_prices[ticker], model='multiplicative', period=252)
        result.plot()
        plt.show()

# key statistics
def key_statistics(close_prices):
    returns= close_prices.pct_change().dropna()
    print("\nSummary statistics of daily returns:")
    summary_stats = returns.describe().T
    summary_stats['annualized_return'] = returns.mean() * 252
    summary_stats['annualized_volatility'] = returns.std() * np.sqrt(252)
    summary_stats['sharpe_ratio'] = summary_stats['annualized_return'] / summary_stats['annualized_volatility']
    print(summary_stats[['annualized_return', 'annualized_volatility', 'sharpe_ratio']])

# Stationarity Check using Augmented Dickey-Fuller (ADF) Test
def adf_test(close_price, ticker):
    result = adfuller(close_price.dropna())
    print(f"\nADF Test for {ticker} Closing Prices:")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value}")
    if result[1] < 0.05:
        print("Stationary (Reject H0)")
    else:
        print("Non-Stationary (Fail to Reject H0)")

# Autocorrelation and Partial Autocorrelation Plots for ARIMA

def autocorrelation(close_price, tickers):
    for ticker in tickers:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sm.graphics.tsa.plot_acf(close_price[ticker].dropna(), lags=40, ax=axes[0])
        sm.graphics.tsa.plot_pacf(close_price[ticker].dropna(), lags=40, ax=axes[1])
        plt.suptitle(f"ACF and PACF for {ticker} Closing Prices")
        plt.show()