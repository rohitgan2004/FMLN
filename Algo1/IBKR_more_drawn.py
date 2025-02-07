from ib_insync import IB, Stock, MarketOrder
import pandas as pd
import numpy as np
import datetime

###############################
# Global Settings and Globals #
###############################

# Kalman filter globals for a simple constant-price model
kalman_x = None      # The estimated true price
kalman_P = None      # The estimated error covariance
Q = 0.001            # Process noise covariance (tune as needed)
R = 0.1              # Measurement noise covariance (tune as needed)

# Rolling window settings and embedding parameters
WINDOW_SIZE = 2000           # Maximum number of bars to keep
EMBED_DIM = 4                # Embedding dimension (d)
TIME_DELAY = 5               # Time delay (tau) in bars
NEAREST_NEIGHBORS = 5        # Number of nearest neighbors to use in forecast
FORECAST_HORIZON = 1         # How many bars ahead to forecast

# Global DataFrame to store the latest bars
data_df = pd.DataFrame(columns=['time', 'price', 'filtered_price'])

###################################
# Connect to IB and Define Contract
###################################

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Update host/port/clientId as needed

# Define the instrument (for example, Apple stock)
contract = Stock('AAPL', 'SMART', 'USD')

###################################
# Helper Functions
###################################

def kalman_filter_update(new_price):
    """
    Update a simple Kalman filter for a constant-price model.
    The filter assumes that the true price does not change much from one step to the next.
    """
    global kalman_x, kalman_P, Q, R
    if kalman_x is None:
        # First observation: initialize state and error covariance.
        kalman_x = new_price
        kalman_P = 1.0
        return new_price

    # Prediction step: for a constant model, the prediction is the previous estimate.
    x_pred = kalman_x
    P_pred = kalman_P + Q

    # Update step: incorporate the new measurement
    K = P_pred / (P_pred + R)  # Kalman gain
    kalman_x = x_pred + K * (new_price - x_pred)
    kalman_P = (1 - K) * P_pred

    return kalman_x

def get_delay_embedding(series, d=EMBED_DIM, tau=TIME_DELAY):
    """
    Construct delay embedding vectors from a 1D NumPy array.
    
    Each embedding vector is constructed as:
    
        X[i] = [ series[i], series[i-tau], series[i-2*tau], ..., series[i-(d-1)*tau] ]
    
    Returns a list of dictionaries, each with:
        - 'time_index': the index in the original series corresponding to the last element of the embedding.
        - 'X': the embedding vector (as a NumPy array).
    """
    embeddings = []
    N = len(series)
    # Need at least (d-1)*tau + 1 data points to form one embedding vector.
    for i in range((d - 1) * tau, N):
        vec = np.array([series[i - j * tau] for j in range(d)])
        embeddings.append({'time_index': i, 'X': vec})
    return embeddings

def nearest_neighbor_forecast(embedded_data, current_vector, k=NEAREST_NEIGHBORS,
                              horizon=FORECAST_HORIZON, original_series=None):
    """
    Forecast the future price by:
      1. Finding the k nearest embedded vectors (using Euclidean distance) to the current_vector.
      2. For each neighbor, taking the price that occurred 'horizon' bars after the neighbor.
      3. Averaging these outcomes to produce a forecast.
    
    Parameters:
      - embedded_data: list of dictionaries returned from get_delay_embedding.
      - current_vector: the latest embedding vector (a NumPy array).
      - k: number of neighbors to use.
      - horizon: how many bars ahead to forecast.
      - original_series: the original NumPy array of filtered prices.
    
    Returns:
      - forecast_price: the average future price from the neighbors (or None if not enough data).
    """
    if original_series is None:
        print("Error: original_series is required for forecasting.")
        return None

    distances = []
    for emb in embedded_data:
        # Ensure that each neighbor has a future price available
        if emb['time_index'] + horizon >= len(original_series):
            continue
        vec = emb['X']
        dist = np.linalg.norm(current_vector - vec)
        distances.append((dist, emb))

    # Sort the embeddings by distance (nearest first)
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    outcomes = []
    for _, emb in neighbors:
        outcome_index = emb['time_index'] + horizon
        outcomes.append(original_series[outcome_index])
    if outcomes:
        forecast_price = np.mean(outcomes)
        return forecast_price
    else:
        return None

def place_order(action, quantity):
    """
    Place a market order via the IB API.
    
    Parameters:
      - action: 'BUY' or 'SELL'
      - quantity: number of shares to trade
    """
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(contract, order)
    print(f"{datetime.datetime.now()}: Placed {action} order for {quantity} shares.")
    return trade

###################################
# Real-Time Bar Handler
###################################

def onNewBar(bar):
    """
    Callback to handle each new bar from IB.
    
    For each new bar:
      1. Update the rolling DataFrame.
      2. Filter the price using the Kalman filter.
      3. Build the delay embedding if enough data is available.
      4. Forecast the next price using nearest neighbors.
      5. Compare forecast with current price and place an order if conditions are met.
    """
    global data_df

    # Extract bar data (assuming bar has .time and .close attributes)
    new_time = bar.time
    new_price = bar.close
    filtered_price = kalman_filter_update(new_price)

    # Append the new data row to our DataFrame
    data_df = data_df.append({'time': new_time,
                              'price': new_price,
                              'filtered_price': filtered_price},
                             ignore_index=True)

    # Maintain the rolling window size
    if len(data_df) > WINDOW_SIZE:
        data_df = data_df.iloc[-WINDOW_SIZE:]

    # Check if we have enough data for delay embedding.
    # Need at least (EMBED_DIM - 1)*TIME_DELAY + 1 data points.
    if len(data_df) >= (EMBED_DIM - 1) * TIME_DELAY + 1:
        # Use the filtered prices for embedding
        filtered_series = data_df['filtered_price'].values.astype(float)
        embedded_data = get_delay_embedding(filtered_series, d=EMBED_DIM, tau=TIME_DELAY)

        # Get the current state as the last embedding vector
        current_embedding = embedded_data[-1]['X']

        # Forecast the future price using nearest neighbors.
        forecast_price = nearest_neighbor_forecast(
            embedded_data,
            current_embedding,
            k=NEAREST_NEIGHBORS,
            horizon=FORECAST_HORIZON,
            original_series=filtered_series
        )

        if forecast_price is not None:
            current_actual_price = new_price
            print(f"{datetime.datetime.now()}: Current price: {current_actual_price:.2f}, "
                  f"Forecast price: {forecast_price:.2f}")

            # Trading logic: if the forecast price is significantly higher than the current price, buy;
            # if significantly lower, sell. Adjust threshold and quantity as needed.
            threshold = 0.001  # 0.1%
            if forecast_price > current_actual_price * (1 + threshold):
                place_order('BUY', 10)
            elif forecast_price < current_actual_price * (1 - threshold):
                place_order('SELL', 10)
        else:
            print(f"{datetime.datetime.now()}: Forecast price unavailable (not enough neighbor outcomes).")

###################################
# Subscribe to Real-Time Data and Run
###################################

# Request historical data with the keepUpToDate flag set to True to stream new bars.
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='1 D',
    barSizeSetting='1 min',
    whatToShow='TRADES',
    useRTH=False,
    keepUpToDate=True
)

# Attach our onNewBar callback so it is called every time a new bar is available.
bars.updateEvent += onNewBar

# Start the IB event loop. This call is blocking.
ib.run()
