import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime
import threading
import webbrowser
import base64
import json
import urllib.parse
import os
import yaml

# Kalman Filter implementation
class KalmanFilter:
    def __init__(self, A=1, H=1, Q=1e-5, R=0.1, initial_state=0, initial_covariance=1.0):
        self.A = A  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = initial_state  # Initial state estimate
        self.P = initial_covariance  # Initial covariance estimate

    def predict(self):
        # Predict the next state
        self.x = self.A * self.x
        self.P = self.A * self.P * self.A + self.Q

    def update(self, z):
        # Update the state with new measurement z
        K = self.P * self.H / (self.H * self.P * self.H + self.R)  # Kalman Gain
        self.x = self.x + K * (z - self.H * self.x)
        self.P = (1 - K * self.H) * self.P

    def get_state(self):
        return self.x

# Schwab API Client
# Updated SchwabClient class with corrected URLs
class SchwabClient:
    def __init__(self, app_key, app_secret, callback_url="https://127.0.0.1", tokens_file='tokens.json', timeout=5, verbose=False):
        self.app_key = app_key
        self.app_secret = app_secret
        self.callback_url = callback_url
        self.tokens_file = tokens_file
        self.timeout = timeout
        self.verbose = verbose
        self.access_token = None
        self.refresh_token = None
        self.account_hash = None  # To be set after fetching account details
        self._base_api_url = "https://api.schwabapi.com"

        # Initialize tokens
        self._initialize_tokens()

    def _initialize_tokens(self):
        # Load tokens from file or authenticate
        tokens = self._read_tokens_file()
        if tokens:
            self.access_token = tokens.get('access_token')
            self.refresh_token = tokens.get('refresh_token')
            if self.verbose:
                print("Tokens loaded from file.")
        else:
            self._authenticate_user()

        # Fetch account hash
        self.account_hash = self._get_account_hash()

    def _authenticate_user(self):
        # Step 1: Get authorization code
        auth_url = f'{self._base_api_url}/v1/oauth/authorize?client_id={self.app_key}&redirect_uri={self.callback_url}'
        print(f"Please authorize the application by visiting this URL: {auth_url}")
        webbrowser.open(auth_url)
        response_url = input("Paste the full redirect URL after authorization: ")
        parsed_url = urllib.parse.urlparse(response_url)
        code = urllib.parse.parse_qs(parsed_url.query)['code'][0]

        # Step 2: Exchange authorization code for tokens
        token_url = f'{self._base_api_url}/v1/oauth/token'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Basic ' + base64.b64encode(f"{self.app_key}:{self.app_secret}".encode()).decode()
        }
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': self.callback_url
        }
        response = requests.post(token_url, headers=headers, data=data)
        if response.ok:
            tokens = response.json()
            self.access_token = tokens['access_token']
            self.refresh_token = tokens['refresh_token']
            self._write_tokens_file(tokens)
            if self.verbose:
                print("Authentication successful.")
        else:
            raise Exception(f"Authentication failed: {response.text}")

    def _read_tokens_file(self):
        try:
            with open(self.tokens_file, 'r') as f:
                tokens = json.load(f)
                return tokens
        except FileNotFoundError:
            return None

    def _write_tokens_file(self, tokens):
        with open(self.tokens_file, 'w') as f:
            json.dump(tokens, f)

    def _get_account_hash(self):
        url = f'{self._base_api_url}/trader/v1/accounts/accountNumbers'
        headers = {'Authorization': f'Bearer {self.access_token}'}
        response = requests.get(url, headers=headers)
        if response.ok:
            accounts = response.json()
            account_hash = accounts[0]['hashValue']
            return account_hash
        else:
            raise Exception(f"Failed to get account hash: {response.text}")

    def place_order(self, symbol, quantity, side):
        url = f'{self._base_api_url}/trader/v1/accounts/{self.account_hash}/orders'
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        order = {
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": side.upper(),
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "EQUITY"
                    }
                }
            ]
        }
        response = requests.post(url, headers=headers, json=order)
        if response.ok:
            if self.verbose:
                print(f"Order placed successfully: {side} {quantity} shares of {symbol}")
        else:
            raise Exception(f"Order placement failed: {response.text}")


# Function to get price data from Polygon.io
def get_price(symbol, api_key):
    # Get the latest price data for the symbol
    url = f"https://api.polygon.io/v2/last/trade/{symbol}?apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    if 'status' in data and data['status'] == 'NOT_FOUND':
        print(f"Symbol {symbol} not found.")
        return None
    if 'error' in data:
        print(f"Error fetching data for {symbol}: {data['error']}")
        return None
    price = data['results']['p']
    return price

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    polygon_api_key = config['polygon_api_key']
    excel_file_path = config['excel_file_path']
    schwab_config = config['schwab']
    app_key = schwab_config['app_key']
    app_secret = schwab_config['app_secret']
    callback_url = schwab_config['callback_url']
    tokens_file = schwab_config.get('tokens_file', 'tokens.json')

    # Initialize Schwab Client
    schwab_client = SchwabClient(
        app_key=app_key,
        app_secret=app_secret,
        callback_url=callback_url,
        tokens_file=tokens_file,
        verbose=True
    )

    # Read tickers from Excel spreadsheet
    try:
        tickers_df = pd.read_excel(excel_file_path)
        tickers = tickers_df['Ticker'].dropna().unique().tolist()
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Initialize Kalman Filters for each ticker
    kf_dict = {ticker: KalmanFilter() for ticker in tickers}

    # Position tracking for each ticker
    positions = {ticker: 0 for ticker in tickers}
    position_limit = 100  # Maximum number of shares to hold for each ticker
    time_interval = 60  # Time interval in seconds between each data fetch
    threshold = 0.5  # Threshold for trading signals
    quantity = 10  # Number of shares to trade per signal

    try:
        while True:
            for ticker in tickers:
                price = get_price(ticker, polygon_api_key)
                if price is None:
                    continue

                kf = kf_dict[ticker]
                kf.predict()
                kf.update(price)
                estimated_price = kf.get_state()
                position = positions[ticker]

                # Trading logic
                if estimated_price - price > threshold and position < position_limit:
                    # Buy signal
                    schwab_client.place_order(ticker, quantity, 'buy')
                    positions[ticker] += quantity
                    print(f"Bought {quantity} shares of {ticker}")
                elif price - estimated_price > threshold and position > 0:
                    # Sell signal
                    schwab_client.place_order(ticker, quantity, 'sell')
                    positions[ticker] -= quantity
                    print(f"Sold {quantity} shares of {ticker}")
                else:
                    print(f"No trade for {ticker}. Current position: {position} shares.")

                # Print status
                print(f"Time: {datetime.now()}, Ticker: {ticker}, Price: {price}, Estimated: {estimated_price}, Position: {positions[ticker]}")

            # Wait for the next time interval
            time.sleep(time_interval)

    except KeyboardInterrupt:
        print("Trading stopped by user.")

if __name__ == "__main__":
    main()
