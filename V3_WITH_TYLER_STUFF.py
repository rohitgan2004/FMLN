import json
import time
import base64
import requests
import threading
import webbrowser
import urllib.parse
#from .stream import Stream
from datetime import datetime
import pandas as pd
import yfinance as yf
from arch import arch_model
import numpy as np
import tkinter as tk
import math
import winsound #This only works on windows
#remove most of the winsound stuff and replace with print statements for use on mac/linux
app_key = "INSERT"
app_secret = "INSERT"

#Tyler Stuff
class Client:

    def __init__(self, app_key, app_secret, callback_url="https://127.0.0.1", tokens_file="tokens.json", timeout=5, verbose=False, update_tokens_auto=True):
        """
        Initialize a client to access the Schwab API.
        :param app_key: app key credentials
        :type app_key: str
        :param app_secret: app secret credentials
        :type app_secret: str
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

        if app_key is None:
            raise Exception("app_key cannot be None.")
        elif app_secret is None:
            raise Exception("app_secret cannot be None.")
        elif callback_url is None:
            raise Exception("callback_url cannot be None.")
        elif tokens_file is None:
            raise Exception("tokens_file cannot be None.")
        elif len(app_key) != 32 or len(app_secret) != 16:
            raise Exception("App key or app secret invalid length.")
        elif callback_url[0:5] != "https":
            raise Exception("callback_url must be https.")
        elif callback_url[-1] == "/":
            raise Exception("callback_url cannot be path (ends with \"/\").")
        elif tokens_file[-1] == '/':
            raise Exception("Tokens file cannot be path.")
        elif timeout <= 0:
            raise Exception("Timeout must be greater than 0 and is recomended to be 5 seconds or more.")

        self._app_key = app_key             # app key credential
        self._app_secret = app_secret       # app secret credential
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
        auth_url = f'https://api.schwabapi.com/v1/oauth/authorize?client_id={self._app_key}&redirect_uri={self._callback_url}'
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
            'Authorization': f'Basic {base64.b64encode(bytes(f"{self._app_key}:{self._app_secret}", "utf-8")).decode("utf-8")}',
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
    Accounts and Trading Production
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

    def account_orders(self, accountHash, fromEnteredTime, toEnteredTime, maxResults=None, status=None):
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

    def order_details(self, accountHash, orderId):
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

    def order_cancel(self, accountHash, orderId):
        """
        Cancel a specific order by its ID, for a specific account
        :param accountHash: account hash from account_linked()
        :type accountHash: str
        :param orderId: order id
        :type orderId: str|int
        :return: response code
        :rtype: request.Response
        """
        return requests.delete(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/orders/{orderId}',
                               headers={'Authorization': f'Bearer {self.access_token}'},
                               timeout=self.timeout)

    def order_replace(self, accountHash, orderId, order):
        """
        Replace an existing order for an account. The existing order will be replaced by the new order. Once replaced, the old order will be canceled and a new order will be created.
        :param accountHash: account hash from account_linked()
        :type accountHash: str
        :param orderId: order id
        :type orderId: str|int
        :param order: order dictionary, examples in Schwab docs
        :type order: dict
        :return: response code
        :rtype: request.Response
        """
        return requests.put(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/orders/{orderId}',
                            headers={"Accept": "application/json", 'Authorization': f'Bearer {self.access_token}',
                                     "Content-Type": "application/json"},
                            json=order,
                            timeout=self.timeout)

    def account_orders_all(self, fromEnteredTime, toEnteredTime, maxResults=None, status=None):
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

    """
    def order_preview(self, accountHash, orderObject):
        #COMING SOON (waiting on Schwab)
        return requests.post(f'{self._base_api_url}/trader/v1/accounts/{accountHash}/previewOrder',
                             headers={'Authorization': f'Bearer {self.access_token}',
                                      "Content-Type": "application.json"}, data=orderObject)
    """

    def transactions(self, accountHash, startDate, endDate, types, symbol=None):
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

    def transaction_details(self, accountHash, transactionId):
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

    def preferences(self):
        """
        Get user preference information for the logged in user.
        :return: User Preferences and Streaming Info
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/trader/v1/userPreference',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            timeout=self.timeout)

    """
    Market Data
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

    def movers(self, symbol, sort=None, frequency=None):
        """
        Get movers in a specific index and direction
        :param symbol: symbol ("$DJI"|"$COMPX"|"$SPX"|"NYSE"|"NASDAQ"|"OTCBB"|"INDEX_ALL"|"EQUITY_ALL"|"OPTION_ALL"|"OPTION_PUT"|"OPTION_CALL")
        :type symbol: str
        :param sort: sort ("VOLUME"|"TRADES"|"PERCENT_CHANGE_UP"|"PERCENT_CHANGE_DOWN")
        :type sort: str
        :param frequency: frequency (0|1|5|10|30|60)
        :type frequency: int
        :return: movers
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/marketdata/v1/movers/{symbol}',
                            headers={"accept": "application/json", 'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser({'sort': sort, 'frequency': frequency}),
                            timeout=self.timeout)

    def market_hours(self, symbols, date=None):
        """
        Get Market Hours for dates in the future across different markets.
        :param symbols: list of market symbols ("equity", "option", "bond", "future", "forex")
        :type symbols: list
        :param date: date
        :type date: datetime | str
        :return: market hours
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/marketdata/v1/markets',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser(
                                {'markets': symbols, #self._format_list(symbols),
                                 'date': self._time_convert(date, 'YYYY-MM-DD')}),
                            timeout=self.timeout)

    def market_hour(self, market_id, date=None):
        """
        Get Market Hours for dates in the future for a single market.
        :param market_id: market id ("equity"|"option"|"bond"|"future"|"forex")
        :type market_id: str
        :param date: date
        :type date: datetime | str
        :return: market hours
        :rtype: request.Response
        """
        return requests.get(f'{self._base_api_url}/marketdata/v1/markets/{market_id}',
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            params=self._params_parser({'date': self._time_convert(date, 'YYYY-MM-DD')}),
                            timeout=self.timeout)

#Our stuff
def initial_screening(file_path,sheet_name,min_price_limit,max_price_limit,volume_limit,write_file_path):
    df = pd.read_excel(file_path,sheet_name)
    filtered_df = df[(df['Price'] > min_price_limit) & (df['Price'] < max_price_limit) & (df['AverageVolume'] >= volume_limit)]
    filtered_df = filtered_df.sort_values(by='Randomizer', ascending=False)
    df_write = pd.read_excel(write_file_path)
    initial_list = filtered_df['Ticker'].tolist()
    if len(initial_list) < len(df_write):
        initial_list.extend([np.nan] * (len(df_write) - len(initial_list)))

    df_write['TickerAnalysis'] = initial_list

    # Write the updated bought ticker list (minus those it said to sell off) to excel
    df_write.to_excel(write_file_path, sheet_name=sheet_name, index=False)

def load_tickers_from_excel(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    tickers = df['TickerAnalysis'].tolist()  # Assuming the tickers are in a column named 'TickerAnalysis'
    tickers = [ticker for ticker in tickers if isinstance(ticker, str) and ticker.strip()]
    return tickers

def load_tickers_from_excel_sellside_initial(file_path, sheet_name):# Assuming the tickers are in a column named 'BoughtTickers'; takes the top 50 tickers and provides box with list of them. For use outside of while loop
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    tickers_initial_buys = df['BoughtTickers'].tolist()#.head(50).tolist() commented this out for use with top 100 droppers*************************
    tickers_initial_buys = [ticker for ticker in tickers_initial_buys if isinstance(ticker, str) and ticker.strip()]
    winsound.Beep(1000, 500)
    print("Buy top 50:")
    print(tickers_initial_buys)
    return tickers_initial_buys

def load_tickers_from_excel_sellside(file_path, sheet_name):# Assuming the tickers are in a column named 'BoughtTickers'; takes the top 50 tickers
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    tickers = df['BoughtTickers'].tolist()#.head(50).tolist() commented this out for use with top 100 droppers*************************
    tickers = [ticker for ticker in tickers if isinstance(ticker, str) and ticker.strip()]
    return tickers

def fetch_historical_data(ticker, period='5d'): #change to 1d for intraday for sell signals
    stock_data = yf.download(ticker, period=period, interval='1m')
    return stock_data['Close']  

def fetch_historical_data_sellside(ticker, period='5d'): #changed to 1d for intraday for sell signals
    stock_data = yf.download(ticker, period=period, interval='1m')
    return stock_data['Close'] 

def get_bid_ask_spread(ticker):
    stock = yf.Ticker(ticker)
    quote_info = stock.info

    bid = quote_info.get('bid', None)
    ask = quote_info.get('ask', None)

    if bid is not None and ask is not None:
        spread = ask - bid
        return bid, ask, spread
    else:
        print(f"No bid/ask data available for {ticker}")

def fetch_spread_data(ticker, period='1d');
    
    spreads = []
    for i in range(30):
        bid, ask, spread = get_bid_ask_spread(ticker)
        if spread is not None:
            spreads.append(spreads)
        else:
            print("unable to fetch data")
    return np.array(spreads)

def analyze_spread_volatility(spread_data):
    daily_volatility = np.std(spread_data) / np.mean(spread_data)
    annualized_volatility = daily_volatility * np.sqrt(252)
    return annualized_volatility

def check_spread_volatility(ticker, threshold = 0.2):
    spread_data = fetch_spread_data(ticker)
    if len(spread_data) == 0:
        print(f"No spread data available for {ticker}")
        return True
    
    annualize_vol = analyze_spread_volatility(spread_data)

    if annualize_vol > threshold:
        return True
    else:
        return False
    


def fit_garch_model(returns):

    try:
        model = arch_model(returns, vol='Garch', p=1, q=1)
        model_fit = model.fit(disp='off')
        return model_fit
    
    except Exception as e:
        print(f"GARCH model fitting failed: {e}")
        return None

def qgarch_adjustment(model_fit, returns, nu=0.1, eta=0.1, phi=0.1, theta=0.1):
    residuals = model_fit.resid
    n = len(returns)
    g = np.zeros(n)
    
    # QGARCH adjustments
    for t in range(1, n):
        g[t] = nu + eta * (residuals[t-1] - phi)**2 + theta * g[t-1]
    
    return g[-1]  

def analyze_volatility(data):
    returns = 1000 * data.pct_change().dropna()  
    model_fit = fit_garch_model(returns)  
    volatility = qgarch_adjustment(model_fit, returns) 
    return volatility

def generate_signal(volatility, threshold=1.2):
    if volatility > threshold:
        return "SELL"
    elif volatility < 0.5:
        return "BUY"
    else:
        return "HOLD"

def process_sing_ticker(ticker):
    print(f"Processing {ticker}...")
    data = fetch_historical_data(ticker)

    if check_spread_volatility(ticker):
        return "HOLD"
    
    try:
        if len(data) > 0:
            volatility = analyze_volatility(data)
            signal = generate_signal(volatility)
            signals[ticker] = signal

    except Exception as e:
        print(f"Skipping {ticker} due to an error during processing: {e}")

    return signal

def process_tickers_sellside():#Changed name, uses sellside load tickers, removed print components as to not use up terminal space (might want to put back the errors?)
    tickers = BoughtTickers["ticker"].tolist()
    signals_sellside = {}

    for ticker in tickers:
        #print(f"Processing {ticker}...")
        data = fetch_historical_data_sellside(ticker)
        
        try:
            if len(data) > 0:
                volatility = analyze_volatility(data)
                signal = generate_signal(volatility)
                signals_sellside[ticker] = signal

        except Exception as e:
           # print(f"Skipping {ticker} due to an error during processing: {e}")
            continue
    print(signals_sellside)
    return signals_sellside

def json_inator(value):#Converts to json if status == 200, else returns "error"
    if value.status_code == 200:
        return value.json()
    else:
        return "error"
    
def order_json_inator(value):#Returns "Success" if status == 201, else returns "error"
    if value.status_code == 201:
        return "Success"
    else:
        return "error"

def position_size_gen():
    account_info = json_inator(client.account_details(account_hash))
    total_cash = account_info['securitiesAccount']['currentBalances']['totalCash']
    unsettled_cash = account_info['securitiesAccount']['currentBalances']['unsettledCash']
    settled_cash = total_cash - unsettled_cash
    position_size_target = (settled_cash * 0.75)/position_num_target
    return position_size_target

def settled_cash_balance():
    account_info = json_inator(client.account_details(account_hash))
    cashAvailableForTrading = account_info['securitiesAccount']['currentBalances']['cashAvailableForTrading']
    unsettled_cash = account_info['securitiesAccount']['currentBalances']['unsettledCash']
    settled_cash = cashAvailableForTrading - unsettled_cash
    print(settled_cash)
    return settled_cash

def open_position(ticker):#Opens position with a ticker, determines quantity using ask price and rounds down to the max quantity within the acceptable range of the position $ target
    settled_cash = settled_cash_balance()  
    if settled_cash > position_size_target:
    
        quote = json_inator(client.quote(ticker))
        if quote == "error":
            print("quote_error")
            return "quote_error"
        else:
            askPrice = quote[ticker]["quote"]["askPrice"]
            buyQuantity = math.floor(position_size_target/askPrice)
            order = {"orderType": "MARKET", "session": "NORMAL", "duration": "DAY", "orderStrategyType": "SINGLE",
                        "orderLegCollection": [
                            {"instruction": "BUY", "quantity": buyQuantity, "instrument": {"symbol": ticker, "assetType": "EQUITY"}}]}
            client.order_place(account_hash, order)
            print("Placed order")
            return buyQuantity
    
    else:
        print("insufficient_funds")
        return "insufficient_funds"

def close_position(ticker):#Closes a position, accesses the quantity from the bought tickers df
    sellQuantity = BoughtTickers.loc[BoughtTickers["ticker"] == ticker, "quantity"].values[0]   
    order = {"orderType": "MARKET", "session": "NORMAL", "duration": "DAY", "orderStrategyType": "SINGLE",
                "orderLegCollection": [
                    {"instruction": "SELL", "quantity": sellQuantity, "instrument": {"symbol": ticker, "assetType": "EQUITY"}}]}

    resp = client.order_place(account_hash, order)  
    if order_json_inator(resp) != "error":       
        print("Success")
    else:
        print("SELL ORDER ERROR")
        winsound.Beep(1000,3000)
        return "sell_order_error"   

def BIG_RED_BUTTON():#Closes all positions within the bought positions df
    SELL_TICKERS = BoughtTickers["ticker"].tolist()
    for ticker in SELL_TICKERS:
        close_position(ticker)
    SELL_TICKERS.clear()

#Establishes Client ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
client = Client(app_key,app_secret)
hashes = json_inator(client.account_linked())
account_hash = hashes[0]['hashValue']

#List of Relevant Algo Information/Trading Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
read_file_path = "Tickers-V3-read.xlsx"
write_file_path = "Tickers-V3-write.xlsx"  
sheet_name = "Sheet1"
volume_threshold = 10000
min_price_threshold = 1.01
max_price_threshold = 50
position_num_target = 10
position_size_target = position_size_gen()

print(json_inator(client.account_details_all()))


#Beginning of acitve bit -- buy side ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Goes through tickers from read sheet, filters those that do not meet the price or volume threshold, and then prints the list of them to the write sheet
initial_screening(read_file_path,sheet_name,min_price_threshold,max_price_threshold,volume_threshold,write_file_path)

tickers = load_tickers_from_excel(write_file_path, sheet_name)
signals = {}
buy_count = 0
BoughtTickers = pd.DataFrame({
    "ticker": [],
    "quantity": []
})

for ticker in tickers:
    if buy_count<position_num_target:
        try:
            signal = process_sing_ticker(ticker)
            if signal == "BUY":
                quote = json_inator(client.quote(ticker))
                if quote == "error":
                    print("quote_error")
                else:
                    assetMainType = quote[ticker]["assetMainType"]
                    totalVolume = quote[ticker]["quote"]["totalVolume"]
                    askPrice = quote[ticker]["quote"]["askPrice"]
                    askSize = quote[ticker]["quote"]["askSize"]
                    askFlux = askPrice*askSize*100  #Take this with a grain of salt--should be good enought though for what it's being used for
                    print(assetMainType, totalVolume, askPrice, askSize, askFlux)
                    if (assetMainType == "EQUITY") and (totalVolume >= volume_threshold) and (min_price_threshold < askPrice < max_price_threshold) and (askPrice < position_size_target) and (askFlux>position_size_target):          
                        print(f"{ticker}: {signal}") 
                        buy_count = buy_count+1
                        buy_quant = open_position(ticker)
                        boughtTicker = { "ticker": ticker,"quantity": buy_quant}
                        print(boughtTicker)
                        BoughtTickers = pd.concat([BoughtTickers, pd.DataFrame([boughtTicker])], ignore_index=True)
                    else:   
                        print("Ticker failed criteria")

        except Exception as e:
        # print(f"Skipping {ticker} due to an error during processing: {e}")
            continue        
            
        else:
            None
    else:
        None
print(BoughtTickers)

#Beginning of acitve bit -- sell side ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

reapeat_list = []


#Initially loads and alerts about buy signals
#load_tickers_from_excel_sellside_initial(write_file_path,sheet_name)

while True:    
    signals_sellside = process_tickers_sellside()

    for ticker, signal in signals_sellside.items():
        try:
            if signal == "SELL":
                winsound.Beep(1000,500)
                close_position(ticker)
                BoughtTickers = BoughtTickers[BoughtTickers['ticker'] != ticker]
                print(f"{ticker}: {signal}")

            else:
                reapeat_list.append(ticker)
        except Exception as e:
            reapeat_list.append(ticker)  

    if len(reapeat_list) == 0:
        print("All done for the day!")
        break
    
    reapeat_list.clear()
