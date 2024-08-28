import pandas as pd
import yfinance as yf
from arch import arch_model
import numpy as np
import tkinter as tk
import winsound #This only works on windows**********************************************************************************************************

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
    #Display message box showing bought tickers
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()

    
    messagebox.showinfo("Tickers to Buy",tickers_initial_buys)
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

def process_sing_ticker(ticker):
    print(f"Processing {ticker}...")
    data = fetch_historical_data(ticker)
    
    try:
        if len(data) > 0:
            volatility = analyze_volatility(data)
            signal = generate_signal(volatility)
            signals[ticker] = signal

    except Exception as e:
        print(f"Skipping {ticker} due to an error during processing: {e}")

    return signal

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

#List of Relevant Excel Information ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
file_path = "Tickers-V2.3.xlsx"  
sheet_name = "Sheet1"



#Beginning of acitve bit -- buy side ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

tickers = load_tickers_from_excel(file_path, sheet_name)
signals = {}
buy_count = 0
BoughtTickers = []

skip_buy = input("Skip buy side? y/n:\n")

if skip_buy == "n":
    if __name__ == "__main__":
        
        position_target = int(input("How many positions to open today?\n"))
        for ticker in tickers:
            if buy_count<position_target:
                try:
                    signal = process_sing_ticker(ticker)
                    if signal == "BUY":
                        buy_count = buy_count+1
                        print(f"{ticker}: {signal}")
                        BoughtTickers.append(ticker)
                except Exception as e:
                # print(f"Skipping {ticker} due to an error during processing: {e}")
                    continue        
                    
                else:
                    None
            else:
                None
       
        # Load the existing Excel file into a DataFrame; extend the bought tickers string to match the length

        df = pd.read_excel(file_path)
        if len(BoughtTickers) < len(df):
            BoughtTickers.extend([np.nan] * (len(df) - len(BoughtTickers)))
        else:
            None

        # Add the list as a new column in the DataFrame
        # Let's assume the column is named 'BoughtTickers'
        df['BoughtTickers'] = BoughtTickers

        # Write the updated DataFrame back to Excel
        df.to_excel(file_path, sheet_name=sheet_name, index=False)            
                        
    else:
        None                
else:
    None

#Beginning of acitve bit -- sell side ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

reapeat_list = []

if __name__ == "__main__": 
    #Initially loads and alerts about buy signals
    load_tickers_from_excel_sellside_initial(file_path,sheet_name)

    while True:    
        signals_sellside = process_tickers_sellside(file_path, sheet_name)

        for ticker, signal in signals_sellside.items():
            try:
                if signal == "SELL":
                    print(f"{ticker}: {signal}")
                    winsound.Beep(1000, 500)
                    #Display message box showing bought tickers
                    from tkinter import messagebox

                    root = tk.Tk()
                    root.withdraw()

                    
                    messagebox.showinfo("Ticker to Sell",ticker)
                    
                    #Here, we would need to have it just sell the security automatically

                else:
                    reapeat_list.append(ticker)
            except Exception as e:
              reapeat_list.append(ticker)  

        if len(reapeat_list) == 0:
            print("All done for the day!")
            break

        else:
            df = pd.read_excel(file_path)
            if len(reapeat_list) < len(df):
                reapeat_list.extend([np.nan] * (len(df) - len(reapeat_list)))

            df['BoughtTickers'] = reapeat_list

            # Write the updated bought ticker list (minus those it said to sell off) to excel
            df.to_excel(file_path, sheet_name=sheet_name, index=False)

            reapeat_list.clear()
            continue