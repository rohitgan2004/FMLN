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
# Extended Kalman Filter with Mean Reversion
# -----------------------------------
class ExtendedKalmanFilter:
    def __init__(self, num_tickers, initial_states, Q=1e-3, R=0.1, alpha=0.01, mu=100.0):
        """
        Extended Kalman Filter with a mean-reverting state transition model.
        
        Model:
        x_{t+1} = x_t + alpha*(mu - x_t) + w_t
                 = (1 - alpha)*x_t + alpha*mu + w_t

        :param num_tickers: number of states (tickers)
        :param initial_states: initial price estimates (array)
        :param Q: Process noise covariance parameter
        :param R: Measurement noise covariance parameter
        :param alpha: mean reversion speed (0 < alpha ≤ 1)
        :param mu: mean reversion level (long-term mean), can be a scalar or vector
        """
        self.num_tickers = num_tickers
        self.Q = np.eye(num_tickers) * Q
        self.R = np.eye(num_tickers) * R
        self.x = np.array(initial_states, dtype=float)
        self.P = np.eye(num_tickers)
        self.alpha = alpha
        self.mu = mu  # scalar mean for simplicity

    def f(self, x):
        # Mean reverting function:
        return (1.0 - self.alpha) * x + self.alpha * self.mu

    def h(self, x):
        # Observation function (linear identity)
        return x

    def F_jacobian(self, x):
        # Jacobian of f w.r.t x: df/dx = (1 - alpha)*I
        return np.eye(self.num_tickers) * (1.0 - self.alpha)

    def H_jacobian(self, x):
        # Jacobian of h w.r.t x: h(x)=x, dh/dx=I
        return np.eye(self.num_tickers)

    def predict(self):
        # Predict step
        F = self.F_jacobian(self.x)
        self.x = self.f(self.x)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        # Update step
        z = np.array(z)
        H = self.H_jacobian(self.x)
        y = z - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.num_tickers) - K @ H) @ self.P

    def get_states(self):
        return self.x

    def predict_future(self, steps=120):
        # Predict the state 'steps' times without updating (forward simulation)
        original_x = self.x.copy()
        original_P = self.P.copy()

        for _ in range(steps):
            self.predict()

        future_price = self.x.copy()

        # Revert to original state
        self.x = original_x
        self.P = original_P
        return future_price

# -----------------------------------
# SchwabClient Class (Assumed unchanged from your original)
# -----------------------------------
lass SchwabClient: 

    """
    DON'T Mess with anything from the current lines 19 through 2943 -- those are VERY, VERY important!!!
    """

    def __init__(self, schwab_app_key, schwab_app_secret, callback_url="https://127.0.0.1", tokens_file="tokens.json", timeout=5, verbose=False, update_tokens_auto=True):
        """
        Initialize a client to access the Schwab API.
        :param schwab_app_key: app key credentials
        :type schwab_app_key: str
        :param schwab_app_secret: app secret credentials
        :type schwab_app_secret: str
        :param callback_url: url for callback
        :type callback_url: str
        :param tokens_file: path to tokens file
        :type tokens_file: str
        :param timeout: request timeout
        :type timeout: int
        :param verbose: print extra information
        :type verbose: bool
        :param show_linked: print linked accounts
        :type show_linked: bool
        """

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

        self._schwab_app_key = schwab_app_key             # app key credential
        self._schwab_app_secret = schwab_app_secret       # app secret credential
        self._callback_url = callback_url   # callback url to use
        self.access_token = None            # access token from auth
        self.refresh_token = None           # refresh token from auth
        self.id_token = None                # id token from auth
        self._access_token_issued = None    # datetime of access token issue
        self._refresh_token_issued = None   # datetime of refresh token issue
        self._access_token_timeout = 1680   # in seconds (from schwab) -- RYAN: is 30min, bumped down to 28min
        self._refresh_token_timeout = 6     # in days (from schwab) -- RYAN: is 7, bumped down to 6
        self._tokens_file = tokens_file     # path to tokens file
        self.timeout = timeout              # timeout to use in requests
        self.verbose = verbose              # verbose mode
        #self.stream = Stream(self)          # init the streaming object
        self.awaiting_input = False         # whether we are awaiting user input

        # Try to load tokens from the tokens file
        at_issued, rt_issued, token_dictionary = self._read_tokens_file()
        if None not in [at_issued, rt_issued, token_dictionary]:
            # show user when tokens were last updated and when they will expire
            self.access_token = token_dictionary.get("access_token")
            self.refresh_token = token_dictionary.get("refresh_token")
            self.id_token = token_dictionary.get("id_token")
            self._access_token_issued = at_issued
            self._refresh_token_issued = rt_issued
            if self.verbose:
                print(self._access_token_issued.strftime("Access token last updated: %Y-%m-%d %H:%M:%S") + f" (expires in {self._access_token_timeout - (datetime.now() - self._access_token_issued).seconds} seconds)")
                print(self._refresh_token_issued.strftime("Refresh token last updated: %Y-%m-%d %H:%M:%S") + f" (expires in {self._refresh_token_timeout - (datetime.now() - self._refresh_token_issued).days} days)")
            # check if tokens need to be updated and update if needed
            self.update_tokens()
        else:
            # The tokens file doesn't exist, so create it.
            if self.verbose:
                print(f"Token file does not exist or invalid formatting, creating \"{str(tokens_file)}\"")
            open(self._tokens_file, 'w').close()
            # Tokens must be updated.
            self._update_refresh_token()

        # Spawns a thread to check the access token and update if necessary
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
        """
        Checks if tokens need to be updated and updates if needed (only access token is automatically updated)
        :param force: force update of refresh token (also updates access token)
        :type force: bool
        """
        if (datetime.now() - self._refresh_token_issued).days >= (self._refresh_token_timeout - 1) or force:  # check if we need to update refresh (and access) token
            print("The refresh token has expired, please update!")
            self._update_refresh_token()
        elif ((datetime.now() - self._access_token_issued).days >= 1) or (
                (datetime.now() - self._access_token_issued).seconds > (self._access_token_timeout - 61)):  # check if we need to update access token
            if self.verbose: print("The access token has expired, updating automatically.")
            self._update_access_token()

    def update_tokens_auto(self):
        import warnings
        warnings.warn("update_tokens_auto() is deprecated and is now started when the client is created (if update_tokens_auto=True (default)).", DeprecationWarning, stacklevel=2)

    def _update_access_token(self):
        """
        "refresh" the access token using the refresh token
        """
        # get the token dictionary (we will need to rewrite the file)
        access_token_time_old, refresh_token_issued, token_dictionary_old = self._read_tokens_file()
        # get new tokens
        for i in range(3):
            response = self._post_oauth_token('refresh_token', token_dictionary_old.get("refresh_token"))
            if response.ok:
                # get and update to the new access token
                self._access_token_issued = datetime.now()
                self._refresh_token_issued = refresh_token_issued
                new_td = response.json()
                self.access_token = new_td.get("access_token")
                self.refresh_token = new_td.get("refresh_token")
                self.id_token = new_td.get("id_token")
                self._write_tokens_file(self._access_token_issued, refresh_token_issued, new_td)
                if self.verbose: # show user that we have updated the access token
                    print(f"Access token updated: {self._access_token_issued}")
                break
            else:
                print(response.text)
                print(f"Could not get new access token ({i+1} of 3).")
                time.sleep(10)

    def _update_refresh_token(self):
        """
        Get new access and refresh tokens using authorization code.
        """
        self.awaiting_input = True # set flag since we are waiting for user input
        # get authorization code (requires user to authorize)
        #print("Please authorize this program to access your schwab account.")
        auth_url = f'https://api.schwabapi.com/v1/oauth/authorize?client_id={self._schwab_app_key}&redirect_uri={self._callback_url}'
        print(f"Open to authenticate: {auth_url}")
        webbrowser.open(auth_url)
        response_url = input("After authorizing, paste the address bar url here: ")
        code = f"{response_url[response_url.index('code=') + 5:response_url.index('%40')]}@"  # session = responseURL[responseURL.index("session=")+8:]
        # get new access and refresh tokens
        response = self._post_oauth_token('authorization_code', code)
        if response.ok:
            # update token file and variables
            self._access_token_issued = self._refresh_token_issued = datetime.now()
            new_td = response.json()
            self.access_token = new_td.get("access_token")
            self.refresh_token = new_td.get("refresh_token")
            self.awaiting_input = False  # reset flag since tokens have been updated
            self.id_token = new_td.get("id_token")
            self._write_tokens_file(self._access_token_issued, self._refresh_token_issued, new_td)
            if self.verbose: print("Refresh and Access tokens updated")
        else:
            print(response.text)
            print("Could not get new refresh and access tokens, check these:\n    1. App status is "
                  "\"Ready For Use\".\n    2. App key and app secret are valid.\n    3. You pasted the "
                  "whole url within 30 seconds. (it has a quick expiration)")

    def _post_oauth_token(self, grant_type, code):
        """
        Makes API calls for auth code and refresh tokens
        :param grant_type: 'authorization_code' or 'refresh_token'
        :type grant_type: str
        :param code: authorization code
        :type code: str
        :return: response
        :rtype: requests.Response
        """
        headers = {
            'Authorization': f'Basic {base64.b64encode(bytes(f"{self._schwab_app_key}:{self._schwab_app_secret}", "utf-8")).decode("utf-8")}',
            'Content-Type': 'application/x-www-form-urlencoded'}
        if grant_type == 'authorization_code':  # gets access and refresh tokens using authorization code
            data = {'grant_type': 'authorization_code', 'code': code,
                    'redirect_uri': self._callback_url}
        elif grant_type == 'refresh_token':  # refreshes the access token
            data = {'grant_type': 'refresh_token', 'refresh_token': code}
        else:
            raise Exception("Invalid grant type; options are 'authorization_code' or 'refresh_token'")
        return requests.post('https://api.schwabapi.com/v1/oauth/token', headers=headers, data=data)

    def _write_tokens_file(self, at_issued, rt_issued, token_dictionary):
        """
        Writes token file
        :param at_issued: access token issued
        :type at_issued: datetime
        :param rt_issued: refresh token issued
        :type rt_issued: datetime
        :param token_dictionary: token dictionary
        :type token_dictionary: dict
        """
        try:
            with open(self._tokens_file, 'w') as f:
                toWrite = {"access_token_issued": at_issued.isoformat(), "refresh_token_issued": rt_issued.isoformat(),
                           "token_dictionary": token_dictionary}
                json.dump(toWrite, f, ensure_ascii=False, indent=4)
                f.flush()
        except Exception as e:
            print(e)


    def _read_tokens_file(self):
        """
        Reads token file
        :return: access token issued, refresh token issued, token dictionary
        :rtype: datetime, datetime, dict
        """
        try:
            with open(self._tokens_file, 'r') as f:
                d = json.load(f)
                return datetime.fromisoformat(d.get("access_token_issued")), datetime.fromisoformat(d.get("refresh_token_issued")), d.get("token_dictionary")
        except Exception as e:
            print(e)
            return None, None, None

    def _params_parser(self, params):
        """
        Removes None (null) values
        :param params: params to remove None values from
        :type params: dict
        :return: params without None values
        :rtype: dict
        """
        for key in list(params.keys()):
            if params[key] is None: del params[key]
        return params

    def _time_convert(self, dt=None, form="8601"):
        """
        Convert time to the correct format, passthrough if a string, preserve None if None for params parser
        :param dt: datetime object to convert
        :type dt: datetime
        :param form: what to convert input to
        :type form: str
        :return: converted time or passthrough
        :rtype: str | None
        """
        if dt is None or isinstance(dt, str):
            return dt
        elif form == "8601":  # assume datetime object from here on
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
        """
        Convert python list to string or passthough if already a string i.e ["a", "b"] -> "a,b"
        :param l: list to convert
        :type l: list | str | None
        :return: converted string or passthrough
        :rtype: str | None
        """
        if l is None:
            return None
        elif type(l) is list:
            return ",".join(l)
        else:
            return l
        
    _base_api_url = "https://api.schwabapi.com"

    """
    Accounts and Trading Production  -- either is currently being used, or we might want to integrate it in for a more robust P/L analysis system
    """

    def account_linked(self):
        """
        Account numbers in plain text cannot be used outside of headers or request/response bodies. As the first step consumers must invoke this service to retrieve the list of plain text/encrypted value pairs, and use encrypted account values for all subsequent calls for any accountNumber request.
        :return: All linked account numbers and hashes
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/accountNumbers',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            timeout=self.timeout)

    def account_details_all(self, fields=None):
        """
        All the linked account information for the user logged in. The balances on these accounts are displayed by default however the positions on these accounts will be displayed based on the "positions" flag.
        :param fields: fields to return (options: "positions")
        :type fields: str
        :return: details for all linked accounts
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser({'fields': fields}),
                            timeout=self.timeout)

    def account_details(self, accountHash, fields=None):
        """
        Specific account information with balances and positions. The balance information on these accounts is displayed by default but Positions will be returned based on the "positions" flag.
        :param accountHash: account hash from account_linked()
        :type accountHash: str
        :param fields: fields to return
        :type fields: str
        :return: details for one linked account
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/{accountHash}',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser({'fields': fields}),
                            timeout=self.timeout)

    def account_orders(self, accountHash, fromEnteredTime, toEnteredTime, maxResults=None, status=None):#Could potentially use this to compute the total P/L at the end of the day... check into response format
        """
        All orders for a specific account. Orders retrieved can be filtered based on input parameters below. Maximum date range is 1 year.
        :param accountHash: account hash from account_linked()
        :type accountHash: str
        :param fromEnteredTime: from entered time
        :type fromEnteredTime: datetime | str
        :param toEnteredTime: to entered time
        :type toEnteredTime: datetime | str
        :param maxResults: maximum number of results
        :type maxResults: int
        :param status: status ("AWAITING_PARENT_ORDER"|"AWAITING_CONDITION"|"AWAITING_STOP_CONDITION"|"AWAITING_MANUAL_REVIEW"|"ACCEPTED"|"AWAITING_UR_OUT"|"PENDING_ACTIVATION"|"QUEUED"|"WORKING"|"REJECTED"|"PENDING_CANCEL"|"CANCELED"|"PENDING_REPLACE"|"REPLACED"|"FILLED"|"EXPIRED"|"NEW"|"AWAITING_RELEASE_TIME"|"PENDING_ACKNOWLEDGEMENT"|"PENDING_RECALL"|"UNKNOWN")
        :type status: str
        :return: orders for one linked account hash
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/orders',
                            headers={"Accept": "application/json", 'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser(
                                {'maxResults': maxResults, 'fromEnteredTime': self._time_convert(fromEnteredTime, "8601"),
                                 'toEnteredTime': self._time_convert(toEnteredTime, "8601"), 'status': status}),
                            timeout=self.timeout)

    def order_place(self, accountHash, order):
        """
        Place an order for a specific account.
        :param accountHash: account hash from account_linked()
        :type accountHash: str
        :param order: order dictionary, examples in Schwab docs
        :type order: dict
        :return: order number in response header (if immediately filled then order number not returned)
        :rtype: request.Response
        """
        return requests.post(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/orders',
                             headers={"Accept": "application/json", 'Authorization': f'Bearer {self.access_token}',
                                      "Content-Type": "application/json"},
                             json=order,
                             timeout=self.timeout)

    def order_details(self, accountHash, orderId):#No current use, but we could use this to keep track of specific order details
        """
        Get a specific order by its ID, for a specific account
        :param accountHash: account hash from account_linked()
        :type accountHash: str
        :param orderId: order id
        :type orderId: str|int
        :return: order details
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/orders/{orderId}',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            timeout=self.timeout)

    def account_orders_all(self, fromEnteredTime, toEnteredTime, maxResults=None, status=None):#Could potentially use this to compute the total P/L at the end of the day... check into response format
        """
        Get all orders for all accounts
        :param fromEnteredTime: start date
        :type fromEnteredTime: datetime | str
        :param toEnteredTime: end date
        :type toEnteredTime: datetime | str
        :param maxResults: maximum number of results (set to None for default 3000)
        :type maxResults: int
        :param status: status ("AWAITING_PARENT_ORDER"|"AWAITING_CONDITION"|"AWAITING_STOP_CONDITION"|"AWAITING_MANUAL_REVIEW"|"ACCEPTED"|"AWAITING_UR_OUT"|"PENDING_ACTIVATION"|"QUEUED"|"WORKING"|"REJECTED"|"PENDING_CANCEL"|"CANCELED"|"PENDING_REPLACE"|"REPLACED"|"FILLED"|"EXPIRED"|"NEW"|"AWAITING_RELEASE_TIME"|"PENDING_ACKNOWLEDGEMENT"|"PENDING_RECALL"|"UNKNOWN")
        :type status: str
        :return: all orders
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/trader/v1/orders',
                            headers={"Accept": "application/json", 'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser(
                                {'maxResults': maxResults, 'fromEnteredTime': self._time_convert(fromEnteredTime, "8601"),
                                 'toEnteredTime': self._time_convert(toEnteredTime, "8601"), 'status': status}),
                            timeout=self.timeout)

    def transactions(self, accountHash, startDate, endDate, types, symbol=None):#Could potentially use this to compute the total P/L at the end of the day... check into response format
        """
        All transactions for a specific account. Maximum number of transactions in response is 3000. Maximum date range is 1 year.
        :param accountHash: account hash number
        :type accountHash: str
        :param startDate: start date
        :type startDate: datetime | str
        :param endDate: end date
        :type endDate: datetime | str
        :param types: transaction type ("TRADE, RECEIVE_AND_DELIVER, DIVIDEND_OR_INTEREST, ACH_RECEIPT, ACH_DISBURSEMENT, CASH_RECEIPT, CASH_DISBURSEMENT, ELECTRONIC_FUND, WIRE_OUT, WIRE_IN, JOURNAL, MEMORANDUM, MARGIN_CALL, MONEY_MARKET, SMA_ADJUSTMENT")
        :type types: str
        :param symbol: symbol
        :return: list of transactions for a specific account
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/transactions',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser(
                                {'accountNumber': accountHash, 'startDate': self._time_convert(startDate, "8601"),
                                 'endDate': self._time_convert(endDate, "8601"), 'symbol': symbol, 'types': types}),
                            timeout=self.timeout)

    def transaction_details(self, accountHash, transactionId): #No current use, but we could use this to keep track of specific transaction details
       
        """
        Get specific transaction information for a specific account
        :param accountHash: account hash number
        :type accountHash: str
        :param transactionId: transaction id
        :type transactionId: str|int
        :return: transaction details of transaction id using accountHash
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/transactions/{transactionId}',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            params={'accountNumber': accountHash, 'transactionId': transactionId},
                            timeout=self.timeout)

    def preferences(self): #Might be handy, kinda forgot what it did but seemed vaguely useful
        """
        Get user preference information for the logged in user.
        :return: User Preferences and Streaming Info
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/trader/v1/userPreference',
                            headers={'Authorization': f'Bearer {self.access_token}'},
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

    def price_history(self, symbol, periodType=None, period=None, frequencyType=None, frequency=None, startDate=None,
                      endDate=None, needExtendedHoursData=None, needPreviousClose=None):
        """
        Get price history for a ticker
        :param symbol: ticker symbol
        :type symbol: str
        :param periodType: period type ("day"|"month"|"year"|"ytd")
        :type periodType: str
        :param period: period
        :type period: int
        :param frequencyType: frequency type ("minute"|"daily"|"weekly"|"monthly")
        :type frequencyType: str
        :param frequency: frequency (1|5|10|15|30)
        :type frequency: int
        :param startDate: start date
        :type startDate: datetime | str
        :param endDate: end date
        :type endDate: datetime | str
        :param needExtendedHoursData: need extended hours data (True|False)
        :type needExtendedHoursData: boolean
        :param needPreviousClose: need previous close (True|False)
        :type needPreviousClose: boolean
        :return: dictionary of containing candle history
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/marketdata/v1/pricehistory',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser({'symbol': symbol, 'periodType': periodType, 'period': period,
                                                        'frequencyType': frequencyType, 'frequency': frequency,
                                                        'startDate': self._time_convert(startDate, 'epoch_ms'),
                                                        'endDate': self._time_convert(endDate, 'epoch_ms'),
                                                        'needExtendedHoursData': needExtendedHoursData,
                                                        'needPreviousClose': needPreviousClose}),
                            timeout=self.timeout)

# -----------------------------------
# Asynchronous price (quotes) fetching with bid/ask
# -----------------------------------
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

# -----------------------------------
# Trading Utility Functions
# -----------------------------------

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
    if settled_cash > position_size_target:
        quote = json_inator(schwab_client.quote(ticker))
        if quote == "error":
            print("quote_error")
            return "quote_error"
        else:
            askPrice = quote[ticker]["quote"]["askPrice"]
            buyQuantity = math.floor(position_size_target/askPrice)
            if buyQuantity > 0:
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
                if order_json_inator(resp) != "error":
                    print("Placed order")
                    return buyQuantity
                else:
                    print("Buy order error")
                    return 0
            else:
                print("Buy quantity <= 0, no trade.")
                return 0
    else:
        print("insufficient_funds")
        return 0

def close_position(ticker, quantity):
    if quantity <= 0:
        return
    order = {
        "orderType": "MARKET",
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [
            {"instruction": "SELL", "quantity": quantity, "instrument": {"symbol": ticker, "assetType": "EQUITY"}}
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
        position = positions[ticker]
        if position > 0:
            close_position(ticker, position)
            positions[ticker] = 0

# -----------------------------------
# Bid-Ask Spread Ranking Utility Functions
# -----------------------------------

def verify_price_consistency(ticker, polygon_price, schwab_client, max_diff=0.05):
    # Example of verifying price consistency with Schwab. Adjust as needed.
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

    # Separate tickers by variation threshold
    within_threshold = latest_data[latest_data['variation'] <= max_variation_threshold]
    skipped = latest_data[latest_data['variation'] > max_variation_threshold]

    # Sort within threshold by variation descending
    within_threshold.sort_values(by='variation', ascending=False, inplace=True)
    ranked_tickers = within_threshold['ticker'].tolist()
    skipped_tickers = skipped['ticker'].tolist()

    return ranked_tickers, skipped_tickers

# -----------------------------------
# Trading Logic with Future Prediction
# -----------------------------------
def process_ticker(data, positions, current_price, threshold, future_prices, skipped_tickers):
    idx, ticker, estimated_price = data

    # If ticker is in skipped_tickers, skip this iteration
    if ticker in skipped_tickers:
        print(f"Skipping {ticker} due to high variation this iteration.")
        return

    price = current_price.get(ticker, {}).get('price', None)
    if price is None or price <= 0:
        return
    position = positions[ticker]

    future_price = future_prices[idx]

    print(f"Ticker: {ticker}, Current Price: {price}, Estimated Price: {estimated_price}, Future Price(10m): {future_price}")

    # If future price is higher than current price, buy now (if no position)
    if future_price > price and position == 0:
        if verify_price_consistency(ticker, price, schwab_client):
            quantity = open_position(ticker)
            if isinstance(quantity, int) and quantity > 0:
                positions[ticker] += quantity
                buy_time[ticker] = datetime.now()
                print(f"Bought {quantity} shares of {ticker}")
        else:
            print(f"Price discrepancy too large for {ticker}, skipping buy.")
    # If we have a position for 10 minutes or more, sell it
    elif position > 0:
        # Check if 10 minutes have passed since the buy
        if ticker in buy_time and (datetime.now() - buy_time[ticker]) >= timedelta(minutes=10):
            quantity = position
            close_position(ticker, quantity)
            positions[ticker] = 0
            print(f"Sold {quantity} shares of {ticker} after 10 minutes hold.")

    print(f"Time: {datetime.now()}, Ticker: {ticker}, Price: {price}, Position: {positions[ticker]}")

# -----------------------------------
# Trading Session Setup
# -----------------------------------

min_price_threshold = 1.01
max_price_threshold = 50
buffer_amount = 3500
position_size_target = 50

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

try:
    tickers_df = pd.read_excel(excel_file_path)
    tickers = tickers_df['Ticker'].dropna().unique().tolist()
except Exception as e:
    print(f"Error reading Excel file: {e}")
    tickers = []

num_tickers = len(tickers)

loop = asyncio.get_event_loop()
quotes = loop.run_until_complete(get_quotes_async(tickers, polygon_api_key))

initial_prices = [quotes.get(t, {'price': 100.0})['price'] for t in tickers]

# Initialize EKF
ekf = ExtendedKalmanFilter(num_tickers=num_tickers, initial_states=initial_prices, alpha=0.01, mu=100.0)

positions = {ticker: 0 for ticker in tickers}
current_price = {ticker: {'price':0.0, 'bid':0.0, 'ask':0.0} for ticker in tickers}
time_interval = 5
threshold = 0.005

buy_time = {}
est = timezone('US/Eastern')
initial_capital = 1000.0
net_profit_loss = 0.0

# Initialize a DataFrame to store bid-ask data
bid_ask_data = pd.DataFrame(columns=['timestamp', 'ticker', 'bid', 'ask'])

try:
    while True:
        now_est = datetime.now(est)
        start_time = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
        end_time = now_est.replace(hour=15, minute=59, second=0, microsecond=0)
        BIG_RED_BUTTON_EndTime = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

        if start_time <= now_est <= end_time:
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
                ranked_tickers = tickers[:]
                skipped_tickers = []

            # Update EKF
            ekf.predict()
            z = [current_price.get(t, {'price':100.0})['price'] for t in tickers]
            ekf.update(z)
            estimated_prices = ekf.get_states()

            # Predict future (10 minutes ahead)
            future_prices = ekf.predict_future(steps=120)  # 120 steps * 5s = 600s (10 mins)

            ticker_data_list = [(idx, t, estimated_prices[idx]) for idx, t in enumerate(tickers)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(lambda data: process_ticker(data, positions, current_price, threshold, future_prices, skipped_tickers), ticker_data_list)

            time.sleep(time_interval)

        elif end_time < now_est <= BIG_RED_BUTTON_EndTime:
            BIG_RED_BUTTON()
        else:
            if now_est > end_time:
                # After hours, wait until next day
                next_day = now_est + timedelta(days=1)
                start_time = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
            elif now_est < start_time:
                # Before market open, wait until open
                start_time = start_time

            time_until_start = (start_time - now_est).total_seconds()
            while start_time.weekday() >= 5:
                start_time += timedelta(days=1)
                time_until_start = (start_time - now_est).total_seconds()

            print(f"Not trading hours. Sleeping for {time_until_start} seconds.")
            time.sleep(time_until_start)
            continue

except KeyboardInterrupt:
    print("Trading stopped by user.")

BIG_RED_BUTTON()
