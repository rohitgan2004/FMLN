import requests
import pandas as pd
import numpy as np
import datetime
import time

###############################
# Global Settings and Globals #
###############################

# IBKR Client Portal API base URL (adjust host/port as needed)
BASE_URL = "http://localhost:4999/v1/api"

# Define the contract (example: Apple stock)
CONID    = 265598     # IBKR’s conid for AAPL (verify with IBKR)
SYMBOL   = "AAPL"
EXCHANGE = "SMART"
CURRENCY = "USD"

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

# To avoid processing the same bar more than once
last_bar_time = None

###################################
# Helper Functions
###################################

def kalman_filter_update(new_price):
    """
    Update a simple Kalman filter for a constant-price model.
    Assumes that the true price does not change much from one step to the next.
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
    Returns a list of dictionaries with:
        - 'time_index': index corresponding to the last element in the embedding.
        - 'X': the embedding vector (a NumPy array).
    """
    embeddings = []
    N = len(series)
    # Need at least (d-1)*tau + 1 data points to form one embedding.
    for i in range((d - 1) * tau, N):
        vec = np.array([series[i - j * tau] for j in range(d)])
        embeddings.append({'time_index': i, 'X': vec})
    return embeddings

def nearest_neighbor_forecast(embedded_data, current_vector, k=NEAREST_NEIGHBORS,
                              horizon=FORECAST_HORIZON, original_series=None):
    """
    Forecast the future price by:
      1. Finding the k nearest embedded vectors (using Euclidean distance) to current_vector.
      2. For each neighbor, take the price that occurred 'horizon' bars after the neighbor.
      3. Average these outcomes to produce a forecast.
    """
    if original_series is None:
        print("Error: original_series is required for forecasting.")
        return None

    distances = []
    for emb in embedded_data:
        # Ensure that the neighbor has a future price available
        if emb['time_index'] + horizon >= len(original_series):
            continue
        vec = emb['X']
        dist = np.linalg.norm(current_vector - vec)
        distances.append((dist, emb))
    
    if len(distances) < k:
        k = len(distances)

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
    Place a market order via the IBKR Web API.
    Constructs a JSON payload and sends it to the order endpoint.
    """
    order_data = {
        "conid": CONID,
        "secType": "STK",
        "symbol": SYMBOL,
        "exchange": EXCHANGE,
        "currency": CURRENCY,
        "action": action,
        "orderType": "MKT",
        "totalQuantity": quantity,
        "tif": "DAY"
    }
    url = f"{BASE_URL}/iserver/account/orders"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=order_data, headers=headers, verify=False)
    if response.status_code in (200, 201):
        print(f"{datetime.datetime.now()}: Placed {action} order for {quantity} shares.")
    else:
        print(f"{datetime.datetime.now()}: Order placement failed: {response.text}")
    return response


def get_latest_bar():
    url = f"{BASE_URL}/iserver/marketdata/snapshot"
    params = {
        "conid": CONID,
        "bar": "1",
        "duration": "1 D",
        "useRth": "0"
    }
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, verify=False, timeout=10)
            response.raise_for_status()
            data = response.json()
            bars = data.get("bars", [])
            if bars:
                latest_bar = bars[-1]
                try:
                    latest_bar['time'] = datetime.datetime.fromisoformat(latest_bar['time'])
                except Exception as e:
                    print("Error parsing bar time:", e)
                    latest_bar['time'] = datetime.datetime.now()
                return latest_bar
            else:
                return None
        except requests.exceptions.RequestException as e:
            print(f"{datetime.datetime.now()}: Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return None

def onNewBar(bar):
    """
    Process a new bar by updating the rolling DataFrame, filtering the price,
    building a delay embedding, forecasting a future price, and placing orders if needed.
    """
    global data_df

    new_time = bar['time']
    new_price = float(bar['close'])
    filtered_price = kalman_filter_update(new_price)

    # Append the new row to the DataFrame without using deprecated append
    new_row = pd.DataFrame([{
        'time': new_time,
        'price': new_price,
        'filtered_price': filtered_price
    }])
    data_df = pd.concat([data_df, new_row], ignore_index=True)

    # Keep only the most recent WINDOW_SIZE bars
    if len(data_df) > WINDOW_SIZE:
        data_df = data_df.iloc[-WINDOW_SIZE:]

    # Ensure enough data for delay embedding
    if len(data_df) >= (EMBED_DIM - 1) * TIME_DELAY + 1:
        filtered_series = data_df['filtered_price'].values.astype(float)
        embedded_data = get_delay_embedding(filtered_series, d=EMBED_DIM, tau=TIME_DELAY)
        current_embedding = embedded_data[-1]['X']
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

            # Trading logic: if forecasted price is higher than current by a threshold, buy;
            # if lower, sell. Adjust threshold and quantity as needed.
            threshold = 0.001  # 0.1%
            if forecast_price > current_actual_price * (1 + threshold):
                place_order('BUY', 10)
            elif forecast_price < current_actual_price * (1 - threshold):
                place_order('SELL', 10)
        else:
            print(f"{datetime.datetime.now()}: Forecast unavailable (insufficient neighbor outcomes).")

###################################
# Main Loop: Poll for New Bars
###################################

def main():
    global last_bar_time

    print("Starting IBKR Web API polling...")
    while True:
        bar = get_latest_bar()
        if bar:
            bar_time = datetime.datetime.fromisoformat(bar['time'])
            if last_bar_time is None or bar_time > last_bar_time:
                last_bar_time = bar_time
                onNewBar(bar)
            else:
                print(f"{datetime.datetime.now()}: No new bar yet (last bar time: {last_bar_time}).")
        else:
            print("No bar data received.")
        # Sleep until the next bar (adjust the interval if needed)
        time.sleep(60)

if __name__ == '__main__':
    main()
