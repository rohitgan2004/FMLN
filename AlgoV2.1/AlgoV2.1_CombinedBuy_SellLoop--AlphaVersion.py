import pandas as pd
import yfinance as yf
from arch import arch_model
import numpy as np

def load_tickers_from_excel(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    tickers = df['TickerAnalysis'].tolist()  # Assuming the tickers are in a column named 'TickerAnalysis'
    tickers = [ticker for ticker in tickers if isinstance(ticker, str) and ticker.strip()]
    return tickers

def load_tickers_from_excel_sellside(file_path, sheet_name):# Assuming the tickers are in a column named 'BoughtTickers'; takes the top 50 tickers
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    tickers = df['BoughtTickers'].head(50).tolist()  
    tickers = [ticker for ticker in tickers if isinstance(ticker, str) and ticker.strip()]
    return tickers

def fetch_historical_data(ticker, period='5d'): #change to 1d for intraday for sell signals
    stock_data = yf.download(ticker, period=period, interval='1m')
    return stock_data['Close']  

def fetch_historical_data_sellside(ticker, period='1d'): #changed to 1d for intraday for sell signals
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

def process_tickers_sellside(file_path, sheet_name='Sheet1'):#Changed name, uses sellside load tickers, removed print components as to not use up terminal space (might want to put back the errors?)
    tickers = load_tickers_from_excel_sellside(file_path, sheet_name)
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

    return signals_sellside

#Beginning of acitve bit -- buy side ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Defined list of bought tickers to be input into excel
BoughtTickers = []

# Example usage
if __name__ == "__main__":
    file_path = "Tickers-15Aug24V2_1.xlsx"  
    sheet_name = "Sheet1"  
    signals = process_tickers(file_path, sheet_name)

    for ticker, signal in signals.items():
        if signal == "BUY":
            print(f"{ticker}: {signal}")
            BoughtTickers.append(ticker)

        else:
            None 
# Load the existing Excel file into a DataFrame; extend the bought tickers string to match the length

df = pd.read_excel('Tickers-15Aug24V2_1.xlsx')
if len(BoughtTickers) < len(df):
    BoughtTickers.extend([np.nan] * (len(df) - len(BoughtTickers)))

# Add the list as a new column in the DataFrame
# Let's assume the column is named 'BoughtTickers'
df['BoughtTickers'] = BoughtTickers

# Write the updated DataFrame back to Excel
df.to_excel('Tickers-15Aug24V2_1.xlsx', sheet_name="Sheet1", index=False)

#Beginning of acitve bit -- sell side ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

file_path = "Tickers-15Aug24V2_1.xlsx"  
sheet_name = "Sheet1" 
reapeat_list = []

if __name__ == "__main__": 
    while True:    
        signals_sellside = process_tickers_sellside(file_path, sheet_name)

        for ticker, signal in signals_sellside.items():
            if signal == "SELL":
                print(f"{ticker}: {signal}")
                #Here, we will either need to make it pause and create a pop-up window, or we would need to have it just sell the security automatically

            else:
                reapeat_list.append(ticker)
        
        if len(reapeat_list) == 0:
            print("All done for the day!")
            break

        else:
            df = pd.read_excel('Tickers-15Aug24V2_1.xlsx')
            if len(reapeat_list) < len(df):
                reapeat_list.extend([np.nan] * (len(df) - len(reapeat_list)))

            df['BoughtTickers'] = reapeat_list

            # Write the updated bought ticker list (minus those it said to sell off) to excel
            df.to_excel('Tickers-15Aug24V2_1.xlsx', sheet_name="Sheet1", index=False)

            reapeat_list.clear()
            continue