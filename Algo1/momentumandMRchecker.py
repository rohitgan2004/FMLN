import numpy as np

def mean_reversion_strategy(data, lookback_period, entry_threshold, exit_threshold):
    # Calculate the mean and standard deviation of the data
    mean = np.mean(data[-lookback_period:])
    std = np.std(data[-lookback_period:])
    
    # Calculate the entry and exit levels
    entry_level = mean + entry_threshold * std
    exit_level = mean + exit_threshold * std
    
    # Determine the trading signal based on the current price
    current_price = data[-1]
    if current_price > entry_level:
        signal = -1  # Short signal
    elif current_price < exit_level:
        signal = 1  # Long signal
    else:
        signal = 0  # No signal
    
    return signal

# Example usage
data = [100, 105, 98, 102, 99, 101, 97, 103, 100, 105]
lookback_period = 5
entry_threshold = 1.0
exit_threshold = 0.5

signal = mean_reversion_strategy(data, lookback_period, entry_threshold, exit_threshold)
print("Trading signal:", signal)