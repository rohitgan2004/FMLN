import pandas as pd
import yfinance as yf
from arch import arch_model
import numpy as np

def load_tickers_from_excel_sellside(file_path, sheet_name):# Assuming the tickers are in a column named 'BoughtTickers'; takes the top 50 tickers
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    tickers = df['BoughtTickers'].head(50).tolist()  
    tickers = [ticker for ticker in tickers if isinstance(ticker, str) and ticker.strip()]
    return tickers

def fetch_historical_data_sellside(ticker, period='1d'): #changed to 1d for intraday for sell signals
    stock_data = yf.download(ticker, period=period, interval='1m')
    return stock_data['Close']  

def fit_garch_model(returns):#I don't see anything to change here

    try:
        model = arch_model(returns, vol='Garch', p=1, q=1)
        model_fit = model.fit(disp='off')
        return model_fit
    
    except Exception as e:
        print(f"GARCH model fitting failed: {e}")
        return None

def qgarch_adjustment(model_fit, returns, nu=0.1, eta=0.1, phi=0.1, theta=0.1):#I don't see anything to change here
    residuals = model_fit.resid
    n = len(returns)
    g = np.zeros(n)
    
    # QGARCH adjustments
    for t in range(1, n):
        g[t] = nu + eta * (residuals[t-1] - phi)**2 + theta * g[t-1]
    
    return g[-1]  

def analyze_volatility(data):#I don't see anything to change here
    returns = 1000 * data.pct_change().dropna()  
    model_fit = fit_garch_model(returns)  
    volatility = qgarch_adjustment(model_fit, returns) 
    return volatility

def generate_signal(volatility, threshold=1.5):#I don't see anything to change here
    if volatility > threshold:
        return "SELL"
    elif volatility < threshold / 2:
        return "BUY"
    else:
        return "HOLD"

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

#Beginning of active bit ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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








