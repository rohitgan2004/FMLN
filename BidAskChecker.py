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

# -----------------------------------
# SchwabClient Class
# -----------------------------------
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
    """
    Market Data -- Kept because we might want some of this for spread stuff
    """
    
    def quotes(self, symbols, fields=None, indicative=False):
        """
        Get quotes for a list of tickers
        :param symbols: list of symbols strings (    e.g. "AMD,INTC" or ["AMD", "INTC"])
        :type symbols: [str] | str
        :param fields: list of fields to get ("all", "quote", "fundamental")
        :type fields: list
        :param indicative: whether to get indicative quotes (True/False)
        :type indicative: boolean
        :return: list of quotes
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/marketdata/v1/quotes',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser(
                                {'symbols': self._format_list(symbols), 'fields': fields, 'indicative': indicative}),
                            timeout=self.timeout)

    def quote(self, symbol_id, fields="all"):
        """
        Get quote for a single symbol
        :param symbol_id: ticker symbol
        :type symbol_id: str (e.g. "AAPL", "/ES", "USD/EUR")
        :param fields: list of fields to get ("all", "quote", "fundamental")
        :type fields: list
        :return: quote for a single symbol
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/marketdata/v1/{urllib.parse.quote(symbol_id)}/quotes',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser({'fields': fields}),
                            timeout=self.timeout)

    def preferences(self):
        return requests.get(f'{self._base_api_url}/trader/v1/userPreference',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            timeout=self.timeout)

def json_inator(value):
    if value.status_code == 200:
        return value.json()
    else:
        return "error"

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

df =  pd.DataFrame(columns=["ticker", "polygon_askPrice", "polygon_bidPrice","polygon_time", "schwab_askPrice", "schwab_bidPrice", "schwab_time"])
tickers_df = pd.read_excel(excel_file_path)
tickers_column = tickers_df['Ticker']
df['tickers'] = tickers_column

tickers = ["GOOG", "AAPL", "MSFT"]

quotes = schwab_client.quotes(tickers).json()

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
                    price = ticker_data['lastTrade']['p']
                    bid_price = ticker_data['lastQuote']['p']
                    ask_price = ticker_data['lastQuote']['P']
                    all_quotes[ticker] = {
                        'price': price,
                        'bid': bid_price,
                        'ask': ask_price
                    }
            else:
                print(f"Error fetching data: {data}")
    return all_quotes

loop = asyncio.get_event_loop()
quotes_poly = loop.run_until_complete(get_quotes_async(tickers, polygon_api_key))
timestamp = datetime.now()
for ticker in tickers:
    ticker = ticker
    q = quotes_poly.get(ticker, None)
    polyAskPrice = q['ask']
    polyBidPrice = q['bid']
    polyTime = timestamp

    df.loc[df['tickers'] == ticker, ['polygon_askPrice', 'polygon_bidPrice', 'polygon_time']] = [
            polyAskPrice, polyBidPrice, polyTime]
               

for ticker in tickers:
    ticker = ticker
    schwab_askPrice = quotes[ticker]["quote"]["askPrice"]
    schwab_bidPrice = quotes[ticker]["quote"]["bidPrice"]
    schwab_time = datetime.now()

    df.loc[df['tickers'] == ticker, ['schwab_askPrice', 'schwab_bidPrice', 'schwab_time']] = [
            schwab_askPrice, schwab_bidPrice, schwab_time]
    
    
df.to_excel("Book1.xlsx", index=False)    