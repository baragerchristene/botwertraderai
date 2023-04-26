# data_collection.py

import os
import requests
import pandas as pd
from datetime import datetime

def fetch_binance_historical_data(symbol, interval, start_time, end_time):
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    else:
        print(f"Error fetching historical data from Binance API: {response.status_code} - {response.text}")
        return None

def fetch_binance_ticker_24hr():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data)
    else:
        print(f"Error fetching ticker data from Binance API: {response.status_code} - {response.text}")
        return None

def find_top_trading_opportunities(df, metric="priceChangePercent", top_n=10):
    df[metric] = df[metric].astype(float)
    df.sort_values(by=metric, ascending=False, inplace=True)
    return df.head(top_n)["symbol"].tolist()

def main():
    # Fetch the 24-hour ticker data
    ticker_data = fetch_binance_ticker_24hr()

    if ticker_data is not None:
        # Find the top 10 trading pairs by price change percentage
        top_trading_pairs = find_top_trading_opportunities(ticker_data, metric="priceChangePercent", top_n=10)

        print("Top 10 trading pairs by price change percentage:")
        print(top_trading_pairs)

        # Fetch historical price data for each trading pair
        interval = "1h"
        start_time = int(datetime(2021, 1, 1).timestamp() * 1000)
        end_time = int(datetime(2021, 12, 31).timestamp() * 1000)

        for symbol in top_trading_pairs:
            historical_data = fetch_binance_historical_data(symbol, interval, start_time, end_time)
            if historical_data is not None:
                # Save historical data to a CSV file or process it further
                pass

if __name__ == "__main__":
    main()
