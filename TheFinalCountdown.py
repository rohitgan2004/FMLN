import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import base64
import json
import urllib.parse
import yaml
from pytz import timezone
import asyncio
import aiohttp
import webbrowser
import concurrent.futures

# Function to fetch prices for multiple tickers asynchronously
async def get_prices_async(symbols, api_key):
    # symbols: list of ticker symbols
    all_prices = {}
    async with aiohttp.ClientSession() as session:
        tasks = []
        max_symbols_per_request = 50  # Adjust based on API limitations
        for i in range(0, len(symbols), max_symbols_per_request):
            chunk = symbols[i:i+max_symbols_per_request]
            symbols_str = ','.join(chunk)
            url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers?tickers={symbols_str}&apiKey={api_key}"
            tasks.append(fetch_data(session, url))
        responses = await asyncio.gather(*tasks)
        for data in responses:
            if data and 'tickers' in data:
                for ticker_data in data['tickers']:
                    ticker = ticker_data['ticker']
                    price = ticker_data['lastTrade']['p']
                    all_prices[ticker] = price
            else:
                print(f"Error fetching data: {data}")
    return all_prices  # Returns a dictionary {ticker: price}

async def fetch_data(session, url):
    try:
        async with session.get(url) as response:
            return await response.json()
    except Exception as e:
        print(f"Exception during data fetch: {e}")
        return None
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



# Vectorized Kalman Filter class
class VectorizedKalmanFilter:
    def __init__(self, num_tickers, initial_states, Q=1e-3, R=0.1):
        self.num_tickers = num_tickers
        self.A = np.eye(num_tickers)  # State transition matrix
        self.H = np.eye(num_tickers)  # Observation matrix
        self.Q = np.eye(num_tickers) * Q  # Process noise covariance
        self.R = np.eye(num_tickers) * R  # Measurement noise covariance
        self.x = np.array(initial_states)  # Initial state estimates
        self.P = np.eye(num_tickers) * 1.0  # Initial covariance estimates

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        z = np.array(z)
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman Gain
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P

    def get_states(self):
        return self.x

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
    # Assume SchwabClient is defined elsewhere in the code
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

    num_tickers = len(tickers)

    # Fetch initial prices for all tickers asynchronously
    loop = asyncio.get_event_loop()
    prices = loop.run_until_complete(get_prices_async(tickers, polygon_api_key))

    # Initialize initial prices
    initial_prices = np.zeros(num_tickers)

    # Initialize Vectorized Kalman Filter
    vkf = VectorizedKalmanFilter(num_tickers=num_tickers, initial_states=initial_prices)

    # Position tracking
    positions = {ticker: 0 for ticker in tickers}
    average_cost = {ticker: 0.0 for ticker in tickers}
    current_price = {ticker: 0.0 for ticker in tickers}
    position_limit = 3  # Maximum number of shares to hold for each ticker
    time_interval = 5  # Time interval in seconds between each data fetch
    threshold = 0.005*current_price  # Threshold for trading signals

    # Trading session parameters
    est = timezone('US/Eastern')
    initial_capital = 1000.0  # Increased initial capital for 1000 tickers
    net_profit_loss = 0.0  # Realized P/L

    try:
        while True:
            now_est = datetime.now(est)
            start_time = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
            end_time = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

            # Check if current time is within trading hours
            if start_time <= now_est <= end_time:
                # Fetch prices for all tickers asynchronously
                prices = loop.run_until_complete(get_prices_async(tickers, polygon_api_key))
                # Update current prices
                for ticker in tickers:
                    price = prices.get(ticker)
                    if price is None:
                        continue
                    current_price[ticker] = price

                # Update Kalman Filter
                vkf.predict()
                # Prepare measurements
                z = []
                for ticker in tickers:
                    price = current_price.get(ticker, 0.0)
                    z.append(price)
                vkf.update(z)
                estimated_prices = vkf.get_states()

                # Trading logic
                # Prepare data for parallel processing
                ticker_data_list = []
                for idx, ticker in enumerate(tickers):
                    ticker_data_list.append((idx, ticker, estimated_prices[idx]))

                # Process tickers in parallel using ThreadPoolExecutor
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    executor.map(lambda data: process_ticker(data, positions, average_cost, current_price, net_profit_loss, initial_capital, threshold, position_limit, schwab_client), ticker_data_list)

                # Compute total invested and P/L
                total_invested = sum(average_cost[ticker] * positions[ticker] for ticker in tickers)
                unrealized_pl = sum((current_price[ticker] - average_cost[ticker]) * positions[ticker] for ticker in tickers)
                total_pl = net_profit_loss + unrealized_pl

                print(f"Total Invested: {total_invested}, Net P/L: {total_pl}")

                # Check for net loss exceeding 10% of initial capital
                if total_pl <= -0.1 * initial_capital:
                    print("Net loss exceeded 10% of initial capital. Stopping trading.")
                    break

                # Wait for the next time interval
                time.sleep(time_interval)
            else:
                # Calculate time until next trading session
                if now_est > end_time:
                    # After trading hours, wait until next day's start time
                    next_day = now_est + timedelta(days=1)
                    start_time = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
                elif now_est < start_time:
                    # Before trading hours, wait until start time
                    start_time = start_time

                time_until_start = (start_time - now_est).total_seconds()

                # Handle weekends
                while start_time.weekday() >= 5:  # Saturday=5, Sunday=6
                    start_time += timedelta(days=1)
                    time_until_start = (start_time - now_est).total_seconds()

                print(f"Not trading hours. Sleeping for {time_until_start} seconds.")
                time.sleep(time_until_start)
                continue

    except KeyboardInterrupt:
        print("Trading stopped by user.")

def process_ticker(data, positions, average_cost, current_price, net_profit_loss, initial_capital, threshold, position_limit, schwab_client):
    idx, ticker, estimated_price = data
    price = current_price.get(ticker)
    if price is None or price <= 0:
        return
    position = positions[ticker]

    # Log the estimates for debugging
    print(f"Ticker: {ticker}, Actual Price: {price}, Estimated Price: {estimated_price}")

    # Compute total invested amount
    total_invested = sum(average_cost[ticker] * positions[ticker] for ticker in positions)

    # Buy logic
    if estimated_price - price > threshold and position < position_limit:
        # Calculate maximum quantity based on $150 expenditure limit
        max_quantity = int(150 // price)
        # Ensure at least 1 share is bought if possible
        if max_quantity < 1:
            print(f"Price of {ticker} is too high to buy with $150 limit.")
            return
        # Adjust quantity if it exceeds position limit
        available_quantity = position_limit - position
        quantity = min(max_quantity, available_quantity)
        if quantity <= 0:
            print(f"Position limit reached for {ticker}. Cannot buy more shares.")
            return
        # Check if we have enough capital
        if total_invested + price * quantity > initial_capital:
            print(f"Not enough capital to buy {quantity} shares of {ticker}")
            return

        # Place order
        schwab_client.place_order(ticker, quantity, 'buy')  # Uncomment for live trading

        # Update positions and average cost
        previous_position = positions[ticker]
        positions[ticker] += quantity
        average_cost[ticker] = ((average_cost[ticker] * previous_position) + (price * quantity)) / positions[ticker]

        print(f"Bought {quantity} shares of {ticker}")

    # Sell logic
    elif price - estimated_price > threshold and position >= 1:
        quantity = position  # Sell all shares held
        # Place order
        schwab_client.place_order(ticker, quantity, 'sell')  # Uncomment for live trading

        # Update net_profit_loss
        profit = (price - average_cost[ticker]) * quantity
        net_profit_loss += profit

        # Update positions
        positions[ticker] -= quantity

        # Reset average cost if no more positions
        if positions[ticker] == 0:
            average_cost[ticker] = 0.0

        print(f"Sold {quantity} shares of {ticker}, Profit: {profit}")

    else:
        print(f"No trade for {ticker}. Current position: {position} shares.")

    # Print status
    print(f"Time: {datetime.now()}, Ticker: {ticker}, Price: {price}, Estimated: {estimated_price}, Position: {positions[ticker]}")

if __name__ == "__main__":
    main()
