# main.py

import argparse
from binance_connector import BinanceConnector
from data_collection import collect_historical_data
from preprocessing import preprocess_data
from ai_strategy import AIStrategy
import dependencies

def main(args):
    # Check and install dependencies
    dependencies.check_install_dependencies()

    # Initialize Binance connector
    connector = BinanceConnector(api_key=args.api_key, secret_key=args.secret_key)

    # Collect historical data
    historical_data = collect_historical_data(connector, args.symbol, args.interval, args.start_time, args.end_time)

    # Preprocess the data
    preprocessed_data = preprocess_data(historical_data)

    # Initialize AI strategy
    ai_strategy = AIStrategy()

    # Train the AI strategy with preprocessed data
    ai_strategy.train(preprocessed_data)

    # Execute the AI strategy
    ai_strategy.execute(connector, args.symbol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Crypto Trading Bot")
    parser.add_argument("--api_key", required=True, help="Binance API Key")
    parser.add_argument("--secret_key", required=True, help="Binance Secret Key")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", default="1m", help="Data interval")
    parser.add_argument("--start_time", help="Start time for historical data")
    parser.add_argument("--end_time", help="End time for historical data")

    args = parser.parse_args()
    main(args)
