#binance_connector.py

import os
from binance import Client
import pandas as pd
import numpy as np

API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_SECRET_KEY')

client = Client(API_KEY, API_SECRET)

def fetch_historical_data(symbol, interval, limit):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                         'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    data = data.astype(float)
    return data

def execute_trade(symbol, signal, quantity):
    if signal == 1:
        # Buy
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
    elif signal == -1:
        # Sell
        order = client.order_market_sell(symbol=symbol, quantity=quantity)
    return order
