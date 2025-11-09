import yfinance as yf
import pandas as pd

def load_data(period="7y"):
    """Load NVDA stock data from Yahoo Finance."""

    # Lets first fetch data from Yahoo Finance
    d = yf.download("NVDA", period=period, auto_adjust=True, progress=False)
    # print(f"Data loaded: {d.shape[0]} rows, {d.shape[1]} columns.") SHould be about 7 years of data meaning 1750 rows give or take
    d = d[['Close']]
    # d.rename(columns={'Close':'price'}, inplace=True)
    d.index = pd.to_datetime(d.index)
    return d

def pretty_print(data):
    """Pretty print basic statistics of the data."""
    # Mainly for debugging purposes to visualize and understand the data
    print(data.head())
    print(data.describe())

def describe_data(data, dates=None, years=4):
    """Describe the data with basic statistics.
    Either provide a date range (dates) or a number of years (years) to filter the data.
    
    Args:
        data (pd.DataFrame): The stock data.
        dates (tuple): A tuple of (start_date, end_date) as strings.
        years (int): Number of years to look back from the latest date.
    
    Returns:
        dict: A dictionary with min, max, avg, and volatility of the 'Close' prices.
    """
    # We can either filter by dates or by number of years
    # For example I want to see stats for the last 5 years or between 2020 and 2023
    if dates:
        data = data.loc[dates[0]:dates[1]]
        start_date, end_date = dates
        # cast to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        data = data.loc[start_date:end_date]
        
    # default to years especially for 4 years ~ user can specify 1 year or 3 years if they wanna change this
    # 4 years is a good default for AI boom
    else:
        end_date = data.index.max()
        start_date = end_date - pd.DateOffset(years=years)
        data = data.loc[start_date:end_date]

    # print(f"\n=== STATS ({start_date.date()} to {end_date.date()}) ===")

    stats = {
        "min_price": float(data['Close'].min().iloc[0]),
        "max_price": float(data['Close'].max().iloc[0]),
        "avg_price": float(data['Close'].mean().iloc[0]),
        "volatility": float(data['Close'].std().iloc[0])
    }
    return stats

if __name__ == "__main__":
    data = load_data()
    # print(data)
    # pretty_print(data)
    # describe_data(data) # FOr 7 years
    # describe_data(data, dates=("2020-01-01", "2025-01-01")) # For specific date range
    # describe_data(data, years=1) # For 3 years
    # print("=== DATA PREVIEW ===")
    # print(data.head())
