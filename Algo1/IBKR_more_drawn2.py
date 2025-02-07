import numpy as np
import pandas as pd
from ib_insync import IB, Stock, MarketOrder, util

class KalmanFilter1D:
    """
    A basic 1D Kalman Filter to track a single 'true price' state
    from noisy observations.
    """
    def __init__(self, process_variance=1e-5, measurement_variance=1e-3):
        """
        :param process_variance: Q, process noise variance
        :param measurement_variance: R, measurement noise variance
        """
        self.process_var = process_variance
        self.meas_var = measurement_variance

        # State: x (the true price)
        # Initially unknown, we set to None
        self.x = None

        # State covariance: P
        self.P = 1.0  # large initial uncertainty

    def update(self, measurement):
        """
        Incorporate a new measurement (the observed price) into the filter.
        Returns the updated Kalman estimate.
        """
        if self.x is None:
            # First measurement
            self.x = measurement
            self.P = 1.0
        else:
            # Prediction step (assume x doesn't change drastically between steps)
            # x_pred = x
            # P_pred = P + Q
            self.P += self.process_var

            # Update step
            # K = P_pred / (P_pred + R)
            K = self.P / (self.P + self.meas_var)
            # x_new = x_pred + K * (z - x_pred)
            self.x = self.x + K * (measurement - self.x)
            # P_new = (1 - K) * P_pred
            self.P = (1 - K) * self.P

        return self.x

##############################################################################
# 2. Delay Embedding
##############################################################################
def get_delay_embedding(series, d=4, tau=5):
    """
    Construct delay-embedded vectors from a 1D series.
    :param series: 1D array or list of values (length N).
    :param d: Embedding dimension.
    :param tau: Time delay (in number of steps).
    :return: List of dicts. Each dict has:
        {
          'index': (int) the starting index in the original series,
          'X':     (np.array) the embedded vector in R^d
        }
    """
    embedded = []
    N = len(series)
    # The largest offset in the embedding is (d-1)*tau
    max_offset = (d - 1) * tau
    for i in range(N - max_offset):
        coords = [series[i + j*tau] for j in range(d)]
        embedded.append({
            'index': i,
            'X': np.array(coords, dtype=float)
        })
    return embedded

##############################################################################
# 3. Nearest-Neighbor Forecast
##############################################################################
def nearest_neighbor_forecast(embedded_data, current_vector, k=5, horizon=1):
    """
    Finds k-nearest neighbors of current_vector in 'embedded_data',
    then returns the average of their future values (at index + horizon).
    
    :param embedded_data: list of dicts (as returned by get_delay_embedding).
                          Each dict has 'index' and 'X'.
    :param current_vector: 1D np.array, shape (d,) representing the current state.
    :param k: number of neighbors to use.
    :param horizon: how many steps ahead from the neighbor's index to look.
    :return: float, the forecasted price (mean of neighbor futures).
             Returns None if we cannot compute (e.g. horizon out of range).
    """
    if not embedded_data:
        return None

    # Compute distances
    distances = []
    for item in embedded_data:
        dist = np.linalg.norm(item['X'] - current_vector)
        distances.append((dist, item['index']))
    
    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Take the k nearest
    neighbors = distances[:k]

    # For each neighbor, find the future value in the original series
    # 'index' + horizon might be out of range if near the end
    # so filter those out.
    future_values = []
    for _, idx in neighbors:
        neighbor_future_idx = idx + horizon
        # If neighbor_future_idx is valid in embedded_data's series range, fetch it
        if neighbor_future_idx < len(original_filtered_prices):
            future_values.append(original_filtered_prices[neighbor_future_idx])
    
    if not future_values:
        return None
    
    return np.mean(future_values)

##############################################################################
# 4. Interactive Brokers / main logic
##############################################################################
from ib_insync import BarDataList

# Global constants
WINDOW_SIZE = 500  # how many bars to keep in rolling window
EMBED_DIM = 4
EMBED_TAU = 5
K_NEIGHBORS = 5
FORECAST_HORIZON = 1

# We'll store data in these structures
data_df = pd.DataFrame(columns=['time', 'price', 'filtered_price'])
original_filtered_prices = []  # just a Python list for the embedded forecast usage

# Kalman filter instance (shared)
kf = KalmanFilter1D(process_variance=1e-5, measurement_variance=1e-3)

# Initialize IB connection
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define a contract
contract = Stock('AAPL', 'SMART', 'USD')

def place_order(action, quantity):
    """
    Simple helper to place a MarketOrder. 
    `action` is 'BUY' or 'SELL', `quantity` is an integer.
    """
    # For safety, you'd want to check existing positions, margin, etc.
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(contract, order)
    print(f"Placed {action} order for {quantity} shares.")
    return trade

def onNewBar(bars: BarDataList, hasNewBar: bool):
    """
    This function is called each time a new bar arrives (keepUpToDate=True).
    """
    if not hasNewBar:
        return

    # The last bar is bars[-1]
    bar = bars[-1]
    new_time = bar.time
    new_price = bar.close

    # 1. Update Kalman filter
    filtered_price = kf.update(new_price)

    # 2. Append to data structures
    global data_df, original_filtered_prices
    data_df = data_df.append({
        'time': new_time,
        'price': new_price,
        'filtered_price': filtered_price
    }, ignore_index=True)
    original_filtered_prices.append(filtered_price)

    # 3. Maintain rolling window
    if len(data_df) > WINDOW_SIZE:
        data_df = data_df.iloc[-WINDOW_SIZE:]
        original_filtered_prices = original_filtered_prices[-WINDOW_SIZE:]

    # 4. If we have enough data for embedding + horizon, do a forecast
    if len(original_filtered_prices) > (EMBED_DIM - 1) * EMBED_TAU + FORECAST_HORIZON:
        # Build delay-embedded vectors from the filtered prices in the rolling window
        embedded_data = get_delay_embedding(original_filtered_prices, d=EMBED_DIM, tau=EMBED_TAU)
        # The current vector is the last embedded vector's X
        # But be careful: the "last" embedded item might not line up exactly
        # We'll do a simpler approach: we manually build the current vector from the last d points with step tau.
        current_vector_indices = []
        start_idx = len(original_filtered_prices) - 1 - (EMBED_DIM - 1)*EMBED_TAU
        coords = [original_filtered_prices[start_idx + i*EMBED_TAU] for i in range(EMBED_DIM)]
        current_vector = np.array(coords, dtype=float)

        # 5. Forecast
        forecast_price = nearest_neighbor_forecast(embedded_data, current_vector,
                                                   k=K_NEIGHBORS,
                                                   horizon=FORECAST_HORIZON)
        if forecast_price is not None:
            current_actual_price = new_price
            # Simple example trading logic
            # If forecast is more than 0.1% above current price => buy
            # If forecast is more than 0.1% below current price => sell
            if forecast_price > current_actual_price * 1.001:
                place_order('BUY', 1)
            elif forecast_price < current_actual_price * 0.999:
                place_order('SELL', 1)

    # Debug print
    print(f"Time={new_time}, Price={new_price:.2f}, Filtered={filtered_price:.2f}")

# Request live (or historical+live) bars
bars = ib.reqHistoricalData(
    contract=contract,
    endDateTime='',
    durationStr='1 D',
    barSizeSetting='1 min',    # 1-minute bars
    whatToShow='TRADES',
    useRTH=False,
    keepUpToDate=True
)

# Attach the event handler
bars.updateEvent += onNewBar

# Start the IB event loop
print("Starting IB event loop...")
ib.run()
