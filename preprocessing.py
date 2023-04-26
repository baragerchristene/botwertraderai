# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# Additional imports for feature calculation
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, MACDIndicator

def preprocess_data(data):
    # Data validation and cleaning
    # Check if data has required columns
    required_columns = ['close', 'volume']
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Data is missing required column: {column}")

    # Clean missing or invalid data
    data = data.dropna()

    # Feature Engineering
    # Calculate Simple Moving Averages
    data['sma_short'] = SMAIndicator(data['close'], window=10).sma_indicator()
    data['sma_long'] = SMAIndicator(data['close'], window=50).sma_indicator()

    # Calculate Exponential Moving Averages
    data['ema_short'] = EMAIndicator(data['close'], window=10).ema_indicator()
    data['ema_long'] = EMAIndicator(data['close'], window=50).ema_indicator()

    # Calculate RSI
    data['rsi'] = RSIIndicator(data['close'], window=14).rsi()

    # Calculate MACD
    macd_ind = MACDIndicator(data['close'], window_slow=26, window_fast=12, window_sign=9)
    data['macd'] = macd_ind.macd()
    data['macd_signal'] = macd_ind.macd_signal()
    data['macd_diff'] = macd_ind.macd_diff()

    # Drop any rows with NaN values generated during feature calculation
    data = data.dropna()

    # Extract features (X) and target (y) columns
    X = data.drop(columns=['target'])
    y = data['target']

    # Scale features
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    return X, y
