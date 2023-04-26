# feature_extraction.py

import pandas as pd
import numpy as np
import talib
from datetime import datetime
from sentiment_analysis import get_sentiment
from order_book_data import get_order_book_features

def technical_indicators(df):
    # Add various technical indicators
    df['RSI'] = talib.RSI(df['Close'])
    df['SMA'] = talib.SMA(df['Close'])
    df['EMA'] = talib.EMA(df['Close'])
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'])
    
    return df

def sentiment_features(df):
    df['sentiment'] = df['timestamp'].apply(lambda x: get_sentiment(datetime.fromtimestamp(x)))
    return df

def order_book_features(df):
    df['bid_ask_spread'], df['order_book_imbalance'] = zip(*df['timestamp'].apply(lambda x: get_order_book_features(datetime.fromtimestamp(x))))
    return df

def preprocess_data(df):
    # Apply technical indicators
    df = technical_indicators(df)
    
    # Add sentiment features
    df = sentiment_features(df)

    # Add order book features
    df = order_book_features(df)

    # Drop missing values
    df.dropna(inplace=True)
    
    return df
