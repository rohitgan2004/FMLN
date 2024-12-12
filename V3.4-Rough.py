import json
import time
import base64
import requests
import threading
import webbrowser
import urllib.parse
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import yaml
from pytz import timezone
import asyncio
import aiohttp
import concurrent.futures
import warnings

# Tyler Stuff
class SchwabClient:
    def __init__(self, schwab_app_key, schwab_app_secret, callback_url="https://127.0.0.1", tokens_file="tokens.json", timeout=5, verbose=False, update_tokens_auto=True):
        if schwab_app_key is None:
            raise Exception("schwab_app_key cannot be None.")
        elif schwab_app_secret is None:
            raise Exception("schwab_app_secret cannot be None.")
        elif callback_url is None:
            raise Exception("callback_url cannot be None.")
        elif tokens_file is None:
            raise Exception("tokens_file cannot be None.")
        elif len(schwab_app_key) != 32 or len(schwab_app_secret) != 16:
            raise Exception("App key or app secret invalid length.")
        elif callback_url[0:5] != "https":
            raise Exception("callback_url must be https.")
        elif callback_url[-1] == "/":
            raise Exception("callback_url cannot be path (ends with \"/\").")
        elif tokens_file[-1] == '/':
            raise Exception("Tokens file cannot be path.")
        elif timeout <= 0:
            raise Exception("Timeout must be greater than 0 and is recomended to be 5 seconds or more.")

        self._schwab_app_key = schwab_app_key
        self._schwab_app_secret = schwab_app_secret
        self._callback_url = callback_url
        self.access_token = None
        self.refresh_token = None
        self.id_token = None
        self._access_token_issued = None
        self._refresh_token_issued = None
        self._access_token_timeout = 1680
        self._refresh_token_timeout = 6
        self._tokens_file = tokens_file
        self.timeout = timeout
        self.verbose = verbose
        self.awaiting_input = False

        at_issued, rt_issued, token_dictionary = self._read_tokens_file()
        if None not in [at_issued, rt_issued, token_dictionary]:
            self.access_token = token_dictionary.get("access_token")
            self.refresh_token = token_dictionary.get("refresh_token")
            self.id_token = token_dictionary.get("id_token")
            self._access_token_issued = at_issued
            self._refresh_token_issued = rt_issued
            if self.verbose:
                print(self._access_token_issued.strftime("Access token last updated: %Y-%m-%d %H:%M:%S") + f" (expires in {self._access_token_timeout - (datetime.now() - self._access_token_issued).seconds} seconds)")
                print(self._refresh_token_issued.strftime("Refresh token last updated: %Y-%m-%d %H:%M:%S") + f" (expires in {self._refresh_token_timeout - (datetime.now() - self._refresh_token_issued).days} days)")
            self.update_tokens()
        else:
            if self.verbose:
                print(f"Token file does not exist or invalid formatting, creating \"{str(self._tokens_file)}\"")
            open(self._tokens_file, 'w').close()
            self._update_refresh_token()

        if update_tokens_auto:
            def checker():
                while True:
                    self.update_tokens()
                    time.sleep(60)
            threading.Thread(target=checker, daemon=True).start()
        elif not self.verbose:
            print("Warning: Tokens will not be updated automatically.")

        if self.verbose:
            print("Initialization Complete")

    def update_tokens(self, force=False):
        if (datetime.now() - self._refresh_token_issued).days >= (self._refresh_token_timeout - 1) or force:
            print("The refresh token has expired, please update!")
            self._update_refresh_token()
        elif ((datetime.now() - self._access_token_issued).days >= 1) or (
                (datetime.now() - self._access_token_issued).seconds > (self._access_token_timeout - 61)):
            if self.verbose: print("The access token has expired, updating automatically.")
            self._update_access_token()

    def update_tokens_auto(self):
        warnings.warn("update_tokens_auto() is deprecated and is now started when the client is created (if update_tokens_auto=True (default)).", DeprecationWarning, stacklevel=2)

    def _update_access_token(self):
        access_token_time_old, refresh_token_issued, token_dictionary_old = self._read_tokens_file()
        for i in range(3):
            response = self._post_oauth_token('refresh_token', token_dictionary_old.get("refresh_token"))
            if response.ok:
                self._access_token_issued = datetime.now()
                self._refresh_token_issued = refresh_token_issued
                new_td = response.json()
                self.access_token = new_td.get("access_token")
                self.refresh_token = new_td.get("refresh_token")
                self.id_token = new_td.get("id_token")
                self._write_tokens_file(self._access_token_issued, refresh_token_issued, new_td)
                if self.verbose:
                    print(f"Access token updated: {self._access_token_issued}")
                break
            else:
                print(response.text)
                print(f"Could not get new access token ({i+1} of 3).")
                time.sleep(10)

    def _update_refresh_token(self):
        self.awaiting_input = True
        auth_url = f'https://api.schwabapi.com/v1/oauth/authorize?client_id={self._schwab_app_key}&redirect_uri={self._callback_url}'
        print(f"Open to authenticate: {auth_url}")
        webbrowser.open(auth_url)
        response_url = input("After authorizing, paste the address bar url here: ")
        code = f"{response_url[response_url.index('code=') + 5:response_url.index('%40')]}@"
        response = self._post_oauth_token('authorization_code', code)
        if response.ok:
            self._access_token_issued = self._refresh_token_issued = datetime.now()
            new_td = response.json()
            self.access_token = new_td.get("access_token")
            self.refresh_token = new_td.get("refresh_token")
            self.awaiting_input = False
            self.id_token = new_td.get("id_token")
            self._write_tokens_file(self._access_token_issued, self._refresh_token_issued, new_td)
            if self.verbose: print("Refresh and Access tokens updated")
        else:
            print(response.text)
            print("Could not get new refresh and access tokens.")

    def _post_oauth_token(self, grant_type, code):
        headers = {
            'Authorization': f'Basic {base64.b64encode(bytes(f"{self._schwab_app_key}:{self._schwab_app_secret}", "utf-8")).decode("utf-8")}',
            'Content-Type': 'application/x-www-form-urlencoded'}
        if grant_type == 'authorization_code':
            data = {'grant_type': 'authorization_code', 'code': code, 'redirect_uri': self._callback_url}
        elif grant_type == 'refresh_token':
            data = {'grant_type': 'refresh_token', 'refresh_token': code}
        else:
            raise Exception("Invalid grant type.")
        return requests.post('https://api.schwabapi.com/v1/oauth/token', headers=headers, data=data)

    def _write_tokens_file(self, at_issued, rt_issued, token_dictionary):
        try:
            with open(self._tokens_file, 'w') as f:
                toWrite = {"access_token_issued": at_issued.isoformat(), "refresh_token_issued": rt_issued.isoformat(),
                           "token_dictionary": token_dictionary}
                json.dump(toWrite, f, ensure_ascii=False, indent=4)
                f.flush()
        except Exception as e:
            print(e)

    def _read_tokens_file(self):
        try:
            with open(self._tokens_file, 'r') as f:
                d = json.load(f)
                return datetime.fromisoformat(d.get("access_token_issued")), datetime.fromisoformat(d.get("refresh_token_issued")), d.get("token_dictionary")
        except Exception as e:
            print(e)
            return None, None, None

    def _params_parser(self, params):
        for key in list(params.keys()):
            if params[key] is None: del params[key]
        return params

    def _time_convert(self, dt=None, form="8601"):
        if dt is None or isinstance(dt, str):
            return dt
        elif form == "8601":
            return f'{dt.isoformat()[:-3]}Z'
        elif form == "epoch":
            return int(dt.timestamp())
        elif form == "epoch_ms":
            return int(dt.timestamp() * 1000)
        elif form == "YYYY-MM-DD":
            return dt.strftime("%Y-%m-%d")
        else:
            return dt

    def _format_list(self, l):
        if l is None:
            return None
        elif type(l) is list:
            return ",".join(l)
        else:
            return l

    _base_api_url = "https://api.schwabapi.com"

    def account_linked(self):
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/accountNumbers',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            timeout=self.timeout)

    def account_details_all(self, fields=None):
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser({'fields': fields}),
                            timeout=self.timeout)

    def account_details(self, accountHash, fields=None):
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/{accountHash}',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser({'fields': fields}),
                            timeout=self.timeout)

    def account_orders(self, accountHash, fromEnteredTime, toEnteredTime, maxResults=None, status=None):
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/orders',
                            headers={"Accept": "application/json", 'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser(
                                {'maxResults': maxResults, 'fromEnteredTime': self._time_convert(fromEnteredTime, "8601"),
                                 'toEnteredTime': self._time_convert(toEnteredTime, "8601"), 'status': status}),
                            timeout=self.timeout)

    def order_place(self, accountHash, order):
        return requests.post(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/orders',
                             headers={"Accept": "application/json", 'Authorization': f'Bearer {self.access_token}',
                                      "Content-Type": "application/json"},
                             json=order,
                             timeout=self.timeout)

    def order_details(self, accountHash, orderId):
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/orders/{orderId}',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            timeout=self.timeout)

    def account_orders_all(self, fromEnteredTime, toEnteredTime, maxResults=None, status=None):
        return requests.get(f'{self._base_api_url}/trader/v1/orders',
                            headers={"Accept": "application/json", 'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser(
                                {'maxResults': maxResults, 'fromEnteredTime': self._time_convert(fromEnteredTime, "8601"),
                                 'toEnteredTime': self._time_convert(toEnteredTime, "8601"), 'status': status}),
                            timeout=self.timeout)

    def transactions(self, accountHash, startDate, endDate, types, symbol=None):
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/transactions',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser(
                                {'accountNumber': accountHash, 'startDate': self._time_convert(startDate, "8601"),
                                 'endDate': self._time_convert(endDate, "8601"), 'symbol': symbol, 'types': types}),
                            timeout=self.timeout)

    def transaction_details(self, accountHash, transactionId):
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/transactions/{transactionId}',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            params={'accountNumber': accountHash, 'transactionId': transactionId},
                            timeout=self.timeout)

    def preferences(self):
        return requests.get(f'{self._base_api_url}/trader/v1/userPreference',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            timeout=self.timeout)

class VectorizedKalmanFilter:
    def __init__(self, num_tickers, initial_states, Q=1e-3, R=0.1):
        self.num_tickers = num_tickers
        self.A = np.eye(num_tickers)
        self.H = np.eye(num_tickers)
        self.Q = np.eye(num_tickers) * Q
        self.R = np.eye(num_tickers) * R
        self.x = np.array(initial_states)
        self.P = np.eye(num_tickers) * 1.0

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        z = np.array(z)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P

    def get_states(self):
        return self.x

async def fetch_data(session, url):
    try:
        async with session.get(url) as response:
            return await response.json()
    except Exception as e:
        print(f"Exception during data fetch: {e}")
        return None

async def get_quotes_async(symbols, api_key):
    all_quotes = {}
    async with aiohttp.ClientSession() as session:
        tasks = []
        max_symbols_per_request = 50
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
                    last_trade_price = ticker_data['lastTrade']['p']
                    bid_price = ticker_data['lastQuote']['p']
                    ask_price = ticker_data['lastQuote']['P']
                    all_quotes[ticker] = {
                        'price': last_trade_price,
                        'bid': bid_price,
                        'ask': ask_price
                    }
            else:
                print(f"Error fetching data: {data}")
    return all_quotes

def json_inator(value):
    if value.status_code == 200:
        return value.json()
    else:
        return "error"

def order_json_inator(value):
    if value.status_code == 201:
        return "Success"
    else:
        return "error"

def verify_price_consistency(ticker, polygon_price, schwab_client, max_diff=0.05):
    schwab_q = json_inator(schwab_client.quote(ticker))
    if schwab_q == "error":
        return False
    try:
        schwab_price = schwab_q[ticker]["quote"]["lastPrice"]
    except:
        return False
    if schwab_price == 0:
        return False
    diff_ratio = abs(schwab_price - polygon_price) / schwab_price
    return diff_ratio < max_diff

def compute_variation(bid, ask):
    mid = (bid + ask) / 2
    if mid == 0:
        return 0
    return (ask - bid) / mid

def update_and_rank_tickers(bid_ask_data, max_variation_threshold=0.2):
    # Get latest row per ticker
    latest_data = bid_ask_data.groupby('ticker').tail(1).copy()
    latest_data['variation'] = latest_data.apply(lambda row: compute_variation(row['bid'], row['ask']), axis=1)

    # Separate tickers into those within threshold and those to skip
    within_threshold = latest_data[latest_data['variation'] <= max_variation_threshold]
    skipped = latest_data[latest_data['variation'] > max_variation_threshold]

    # Sort the ones within threshold by variation descending
    within_threshold.sort_values(by='variation', ascending=False, inplace=True)
    ranked_tickers = within_threshold['ticker'].tolist()
    skipped_tickers = skipped['ticker'].tolist()

    return ranked_tickers, skipped_tickers

def settled_cash_balance():
    account_info = json_inator(schwab_client.account_details(account_hash))
    if account_info == "error":
        return 0
    cashAvailableForTrading = account_info['securitiesAccount']['currentBalances']['cashAvailableForTrading']
    unsettled_cash = account_info['securitiesAccount']['currentBalances']['unsettledCash']
    settled_cash = cashAvailableForTrading - unsettled_cash - buffer_amount
    return settled_cash

def open_position(ticker):
    settled_cash = settled_cash_balance()
    partial_target = position_size_target / 2.0
    if settled_cash > partial_target:
        quote = json_inator(schwab_client.quote(ticker))
        if quote == "error":
            print("quote_error")
            return 0
        else:
            askPrice = quote[ticker]["quote"]["askPrice"]
            if not (1 <= askPrice <= 50):
                print(f"{ticker} askPrice {askPrice} not in $1-$50 range, skipping buy.")
                return 0
            buyQuantity = math.floor(partial_target / askPrice)
            if buyQuantity <= 0:
                return 0

            order = {
                "orderType": "MARKET",
                "session": "NORMAL",
                "duration": "DAY",
                "orderStrategyType": "SINGLE",
                "orderLegCollection": [
                    {"instruction": "BUY", "quantity": buyQuantity, "instrument": {"symbol": ticker, "assetType": "EQUITY"}}
                ]
            }
            resp = schwab_client.order_place(account_hash, order)
            if order_json_inator(resp) == "error":
                print("Buy order error")
                return 0
            else:
                print(f"Placed buy order for {buyQuantity} shares of {ticker}")
                return buyQuantity
    else:
        print("insufficient_funds")
        return 0

def close_position(ticker):
    sellQuantity = positions[ticker]
    if sellQuantity <= 0:
        return
    order = {
        "orderType": "MARKET",
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [
            {"instruction": "SELL", "quantity": sellQuantity, "instrument": {"symbol": ticker, "assetType": "EQUITY"}}
        ]
    }
    resp = schwab_client.order_place(account_hash, order)
    if order_json_inator(resp) != "error":
        print("Success")
    else:
        print("SELL ORDER ERROR")

def BIG_RED_BUTTON():
    SELL_TICKERS = list(positions.keys())
    for ticker in SELL_TICKERS:
        if positions[ticker] > 0:
            close_position(ticker)
            positions[ticker] = 0
    SELL_TICKERS.clear()

def process_ticker(data, positions, current_price, threshold, skipped_tickers):
    idx, ticker, estimated_price = data
    # If ticker is in skipped_tickers, skip this iteration
    if ticker in skipped_tickers:
        print(f"Skipping {ticker} due to high variation this iteration.")
        return

    price_data = current_price.get(ticker, None)
    if price_data is None:
        return
    price = price_data['price']
    if price is None or price <= 0:
        return
    position = positions[ticker]

    print(f"Ticker: {ticker}, Actual Price: {price}, Estimated Price: {estimated_price}")

    if not (1 <= price <= 50):
        print(f"{ticker} price {price} not in $1-$50 range, no trade.")
        return

    # Buy logic
    if (estimated_price - price) > threshold and position == 0:
        if verify_price_consistency(ticker, price, schwab_client):
            quantity = open_position(ticker)
            positions[ticker] += quantity
            print(f"Bought {quantity} shares of {ticker}")
        else:
            print(f"Price discrepancy too large for {ticker}, skipping buy.")
    # Sell logic
    elif (price - estimated_price) > threshold and position > 0:
        quantity = position
        close_position(ticker)
        positions[ticker] -= quantity
        print(f"Sold {quantity} shares of {ticker}")
    else:
        print(f"No trade for {ticker}. Current position: {position} shares.")

    print(f"Time: {datetime.now()}, Ticker: {ticker}, Price: {price}, Estimated: {estimated_price}, Position: {positions[ticker]}")

# Parameters
min_price_threshold = 1.01
max_price_threshold = 50
buffer_amount = 3500
position_size_target = 50
time_interval = 120  # 2 minutes

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

polygon_api_key = config['polygon_api_key']
excel_file_path = config['excel_file_path']
schwab_config = config['schwab']
schwab_app_key = schwab_config['schwab_app_key']
schwab_app_secret = schwab_config['schwab_app_secret']

schwab_client = SchwabClient(schwab_app_key, schwab_app_secret)
hashes = json_inator(schwab_client.account_linked())
account_hash = hashes[0]['hashValue']

print(json_inator(schwab_client.account_details_all()))

# Read tickers
try:
    tickers_df = pd.read_excel(excel_file_path)
    tickers = tickers_df['Ticker'].dropna().unique().tolist()
except Exception as e:
    print(f"Error reading Excel file: {e}")
    tickers = []

num_tickers = len(tickers)

loop = asyncio.get_event_loop()

# Initialize a DataFrame to store bid-ask data
bid_ask_data = pd.DataFrame(columns=['timestamp', 'ticker', 'bid', 'ask'])

positions = {ticker: 0 for ticker in tickers}
current_price = {ticker: {'price':0.0, 'bid':0.0, 'ask':0.0} for ticker in tickers}

for ticker in tickers:
    price = current_price.get(ticker, {'price':0.0})['price']

# Fetch initial prices
quotes = loop.run_until_complete(get_quotes_async(tickers, polygon_api_key))
for t in tickers:
    q = quotes.get(t, None)
    if q:
        current_price[t] = q

initial_prices = []
 
for ticker in tickers:
    price = current_price.get(ticker, {'price':0})['price']
    initial_prices.append(price)

vkf = VectorizedKalmanFilter(num_tickers=num_tickers, initial_states=initial_prices)

threshold = 0.005 * price  # Example threshold
est = timezone('US/Eastern')
initial_capital = 1000.0
net_profit_loss = 0.0

try:
    while True:
        now_est = datetime.now(est)
        start_time = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
        end_time = now_est.replace(hour=15, minute=59, second=0, microsecond=0)
        BIG_RED_BUTTON_EndTime = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

        if start_time <= now_est <= end_time:
            # Fetch quotes again
            quotes = loop.run_until_complete(get_quotes_async(tickers, polygon_api_key))
            timestamp = datetime.now()
            for t in tickers:
                q = quotes.get(t, None)
                if q is not None:
                    current_price[t] = q
                    bid_ask_data = pd.concat([bid_ask_data, pd.DataFrame([{
                        'timestamp': timestamp,
                        'ticker': t,
                        'bid': q['bid'],
                        'ask': q['ask']
                    }])], ignore_index=True)

            # Rank and filter tickers by variation
            if len(bid_ask_data) > 0:
                ranked_tickers, skipped_tickers = update_and_rank_tickers(bid_ask_data, max_variation_threshold=0.2)
            else:
                ranked_tickers = tickers[:]  # no ranking yet
                skipped_tickers = []

            # Update Kalman Filter
            vkf.predict()
            z = []
            # z must always include all tickers in their original order
            for ticker in tickers:
                p = current_price.get(ticker, {'price':0})
                z.append(p['price'])
            vkf.update(z)
            estimated_prices = vkf.get_states()

            # Prepare data for thread pool
            ticker_data_list = [(idx, ticker, estimated_prices[idx]) for idx, ticker in enumerate(tickers)]

            # Only skip signals in process_ticker if they're in skipped_tickers
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(lambda data: process_ticker(data, positions, current_price, threshold, skipped_tickers), ticker_data_list)

            time.sleep(time_interval)

        elif end_time < now_est <= BIG_RED_BUTTON_EndTime:
            BIG_RED_BUTTON()
            time.sleep(60)  # Wait till session fully ends
        else:
            # Sleep until next session
            if now_est > end_time:
                next_day = now_est + timedelta(days=1)
                start_time = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
            time_until_start = (start_time - now_est).total_seconds()
            # Handle weekends
            while start_time.weekday() >= 5:
                start_time += timedelta(days=1)
                time_until_start = (start_time - now_est).total_seconds()

            print(f"Not trading hours. Sleeping for {time_until_start} seconds.")
            time.sleep(time_until_start)
            continue

except KeyboardInterrupt:
    print("Trading stopped by user.")
    BIG_RED_BUTTON()
