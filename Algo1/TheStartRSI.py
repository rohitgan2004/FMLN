import pandas as pd
import yfinance as yf
from arch import arch_model
import numpy as np

def load_tickers_from_excel(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    tickers = df['Tickers'].tolist()  # Assuming the tickers are in a column named 'Ticker'
    tickers = [ticker for ticker in tickers if isinstance(ticker, str) and ticker.strip()]
    return tickers


def fetch_historical_data(ticker, period='1d'): #change to 1d for intraday for sell signals
    stock_data = yf.download(ticker, period=period, interval='1m')
    return stock_data['Close']  

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

def generate_signal(volatility, threshold=1.5):
    if volatility > threshold:
        return "SELL"
    elif volatility < threshold / 2:
        return "BUY"
    else:
        return "HOLD"

def process_tickers(file_path, sheet_name='Sheet1'):
    tickers = load_tickers_from_excel(file_path, sheet_name)
    signals = {}

    for ticker in tickers:
        print(f"Processing {ticker}...")
        data = fetch_historical_data(ticker)
        
        try:
            if len(data) > 0:
                volatility = analyze_volatility(data)
                signal = generate_signal(volatility)
                signals[ticker] = signal

        except Exception as e:
            print(f"Skipping {ticker} due to an error during processing: {e}")
            continue

    return signals

# Example usage
if __name__ == "__main__":
    file_path = "Tickers-15Aug24.xlsx"  
    sheet_name = "Sheet1"  
    signals = process_tickers(file_path, sheet_name)

    for ticker, signal in signals.items():
        print(f"{ticker}: {signal}")




def calculate_rsi(data, period=14):
    
    delta = data.diff(1).dropna()  # Calculate the difference between consecutive prices
    gain = delta.where(delta > 0, 0)  # Positive gains
    loss = -delta.where(delta < 0, 0)  # Negative losses

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss  # Relative Strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula

    return rsi

def generate_signal_rsi(rsi_value, buy_threshold=30, sell_threshold=70):
    
    
    if rsi_value < buy_threshold:
        return "BUY"
    elif rsi_value > sell_threshold:
        return "SELL"
    else:
        return "HOLD"

def analyze_rsi(data, period=14):
    """
    Analyze RSI of stock price data and generate signal.
    :param data: pandas series of stock prices (closing prices)
    :param period: RSI period
    :return: Trading signal based on RSI
    """
    rsi_series = calculate_rsi(data, period)
    rsi_value = rsi_series.iloc[-1]  # Get the latest RSI value
    return generate_signal_rsi(rsi_value)

def process_single_ticker_rsi(ticker):
    """
    Process a single ticker, calculate RSI, and determine trading signal.
    :param ticker: The stock ticker
    :return: Trading signal ("BUY", "SELL", or "HOLD")
    """
    print(f"Processing {ticker} using RSI...")
    data = fetch_historical_data(ticker)  # Fetch historical data (closing prices)
    
    try:
        if len(data) > 0:
            signal = analyze_rsi(data)  # Generate signal using RSI
            signals[ticker] = signal
            return signal

    except Exception as e:
        print(f"Skipping {ticker} due to an error during processing: {e}")
        return "HOLD"
