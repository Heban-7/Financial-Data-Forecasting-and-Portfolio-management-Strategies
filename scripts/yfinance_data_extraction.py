import yfinance as yf
import pandas as pd

def fetch_data(tickers):
    # Define the stick tickers and date range
    start_date = "2015-01-01"
    end_date = "2025-01-31"

    # Fetch the data 
    data = yf.download(tickers, start=start_date, end=end_date)
    data.to_csv('../data/stock_data_market.csv')
    return data


