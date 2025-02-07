import numpy as np
import pandas as pd
from ib_insync import IB, Stock, MarketOrder, util
from sklearn.decomposition import PCA
from pykalman import KalmanFilter
import datetime

def connect_to_ib(host='127.0.0.1', port=7497, clientId=1):
    """Connect to the Interactive Brokers Gateway/TWS."""
    ib = IB()
    ib.connect(host, port, clientId)
    return ib

def fetch_data(ib, symbols, duration='1 D', barSize='5 mins'):
    """
    Fetch historical data for a list of symbols.
    
    :param ib: The connected IB object.
    :param symbols: List of equity symbols (e.g., ['AAPL', 'MSFT']).
    :param duration: Duration string (e.g., '1 D' for one day).
    :param barSize: Bar size setting (e.g., '5 mins').
    :return: A dictionary mapping each symbol to its historical data (pandas DataFrame).
    """
    data = {}
    for symbol in symbols:
        contract = Stock(symbol, 'SMART', 'USD')
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=barSize,
            whatToShow='MIDPOINT',
            useRTH=True,
            formatDate=1
        )
        df = util.df(bars)
        df.set_index('date', inplace=True)
        data[symbol] = df
    return data

def choose_embedding_dimension(returns_df, variance_threshold=0.9):
    """
        Choose the number of principal components (embedding dimension) required to capture
        a specified threshold of the total variance in the returns data.

        :param returns_df: DataFrame of returns (rows: timestamps, columns: symbols).
        :param variance_threshold: Cumulative variance threshold (default 0.9 for 90%).
        :return: The chosen embedding dimension (integer).
        """
    pca_temp = PCA()
    pca_temp.fit(returns_df.values)
    cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
    embedding_dim = np.searchsorted(cumulative_variance, variance_threshold) + 1
    print(f"Chosen embedding dimension: {embedding_dim}")
    return embedding_dim
    
def preprocess_data(data_dict):
    """
    Combine data for multiple symbols and compute log returns.
    
    :param data_dict: Dictionary of DataFrames for each symbol.
    :return: A DataFrame of log returns.
    """
    # Combine 'close' prices from each symbol into one DataFrame.
    combined_df = pd.DataFrame()
    for symbol, df in data_dict.items():
        combined_df[symbol] = df['close']
    combined_df.dropna(inplace=True)
    
    # Compute log returns and drop the initial NA row.
    returns = np.log(combined_df).diff().dropna()
    return returns

def compute_pca(returns_df):
    """
    Compute the first principal component (PC1) from the returns data.
    
    :param returns_df: DataFrame of returns (rows: timestamps, columns: symbols).
    :return: A pandas Series of PC1 values and the fitted PCA model.
    """
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(returns_df.values)
    # Create a time series for PC1 using the returns DataFrame's index.
    pc1_series = pd.Series(pc1.flatten(), index=returns_df.index)
    return pc1_series, pca

def apply_kalman_filter(series):
    """
    Smooth the input series using a Kalman filter.
    
    :param series: A pandas Series (e.g., the PC1 time series).
    :return: A pandas Series containing the filtered (smoothed) state estimates.
    """
    # Setup a simple 1D Kalman Filter
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=series.iloc[0],
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01
    )
    state_means, _ = kf.filter(series.values)
    filtered_series = pd.Series(state_means.flatten(), index=series.index)
    return filtered_series

def detect_regime_changes(filtered_series, window=10, threshold=0.001):
    """
    Detect regime changes by comparing rolling means of the filtered series.
    
    A regime change is flagged when the difference between the current rolling
    mean and the previous value of the rolling mean exceeds the threshold.
    
    :param filtered_series: The Kalman-filtered time series.
    :param window: Window size (number of observations) for the rolling mean.
    :param threshold: The minimum change required to signal a regime change.
    :return: A list of tuples (timestamp, signal) where signal is 'BUY' or 'SELL'.
    """
    # Compute the rolling mean of the filtered series.
    rolling_mean = filtered_series.rolling(window=window).mean()
    
    signals = []  # List to store (timestamp, signal)
    # Iterate over the rolling mean series (skipping the first window where data are insufficient)
    for i in range(window, len(rolling_mean)):
        current = rolling_mean.iloc[i]
        previous = rolling_mean.iloc[i - 1]
        # Skip if any rolling value is NaN.
        if pd.isna(current) or pd.isna(previous):
            continue
        # If the rolling mean increases significantly, consider it an upward regime change.
        if (current - previous) > threshold:
            signals.append((rolling_mean.index[i], 'BUY'))
        # If it decreases significantly, consider it a downward regime change.
        elif (current - previous) < -threshold:
            signals.append((rolling_mean.index[i], 'SELL'))
    return signals

def place_order(ib, symbol, signal, quantity=100):
    """
    Place a market order on the specified symbol via IB.
    
    :param ib: The IB connection object.
    :param symbol: Equity symbol (e.g., 'AAPL').
    :param signal: 'BUY' or 'SELL'.
    :param quantity: Number of shares.
    :return: The trade object returned by IB.
    """
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    
    if signal == 'BUY':
        order = MarketOrder('BUY', quantity)
    elif signal == 'SELL':
        order = MarketOrder('SELL', quantity)
    else:
        return None
    
    trade = ib.placeOrder(contract, order)
    return trade

def main():
    # Define the symbols you want to monitor/trade.
    symbols = ['AAPL', 'MSFT', 'GOOG']
    
    # Connect to IB.
    ib = connect_to_ib()
    print("Connected to Interactive Brokers.")
    
    # Fetch historical data.
    data_dict = fetch_data(ib, symbols, duration='1 D', barSize='5 mins')
    print("Fetched historical data for symbols:", symbols)
    
    # Preprocess the data: combine prices and compute log returns.
    returns_df = preprocess_data(data_dict)
    print("Computed log returns. Data shape:", returns_df.shape)
    
    # Compute the first principal component (PC1) from the returns.
    pc1_series, pca_model = compute_pca(returns_df)
    print("Computed PCA. PC1 series obtained.")
    
    # Smooth the PC1 series using a Kalman filter.
    filtered_pc1 = apply_kalman_filter(pc1_series)
    print("Applied Kalman filter to PC1 series.")
    
    # Detect regime changes using rolling window analysis on the filtered series.
    regime_signals = detect_regime_changes(filtered_pc1, window=10, threshold=0.001)
    print("Detected regime changes (timestamp, signal):")
    for ts, sig in regime_signals:
        print(f"  {ts}: {sig}")
    
    # For demonstration, if a regime change signal exists, use the latest one as the trading signal.
    if regime_signals:
        signal_time, signal = regime_signals[-1]
        print(f"Using latest signal at {signal_time}: {signal}")
        trade = place_order(ib, 'AAPL', signal, quantity=100)
        print("Placed order. Trade details:")
        print(trade)
    else:
        print("No regime change signal generated at this time.")
    
    # Disconnect from IB.
    ib.disconnect()
    print("Disconnected from Interactive Brokers.")

if __name__ == "__main__":
    main()
