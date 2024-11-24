import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error

# Fetch and Clean Data using Kalman Filter
# Define API key
API_KEY = ''

# Define the ticker symbol - will add ticker for loop
ticker = 'AAPL'  # Example ticker

# Define dates
today = datetime.today()
two_weeks_ago = today - timedelta(days=14)
one_week_ago = today - timedelta(days=7)

start_date = two_weeks_ago.strftime('%Y-%m-%d')
end_date = one_week_ago.strftime('%Y-%m-%d')

# Fetch data from Polygon.io API
def fetch_polygon_data(ticker, start_date, end_date, API_KEY):
    url = f'https://api.polygon.io/v2/ticks/stocks/trades/{ticker}/{start_date}?apiKey={API_KEY}'
    # Simulated data fetching for example purposes
    dates = pd.date_range(start=start_date, end=end_date, freq='1min')
    data = pd.DataFrame({
        'time': dates,
        'bid': np.random.uniform(100, 200, size=len(dates)),
        'ask': np.random.uniform(100, 200, size=len(dates)),
    })
    return data

# Fetch the data
data = fetch_polygon_data(ticker, start_date, end_date, API_KEY)

# Calculate the bid-ask spread
data['spread'] = data['ask'] - data['bid']

# Clean bid-ask spread data using Kalman Filter
spread = data['spread'].values
initial_state = spread[0]
transition_matrices = [1]
observation_matrices = [1]

kf = KalmanFilter(transition_matrices=transition_matrices,
                  observation_matrices=observation_matrices,
                  initial_state_mean=initial_state)

state_means, state_covariances = kf.filter(spread)
data['cleaned_spread'] = state_means

# Step 2: Develop KFPPM and Volatility Model
# Calculate mid-price
data['mid_price'] = (data['bid'] + data['ask']) / 2

# Implement KFPPM
prices = data['mid_price'].values
initial_price = prices[0]

kf_price = KalmanFilter(transition_matrices=transition_matrices,
                        observation_matrices=observation_matrices,
                        initial_state_mean=initial_price)

state_means_price, state_covariances_price = kf_price.filter(prices)
data['predicted_price'] = state_means_price

# Compute returns for volatility model
data['returns'] = data['mid_price'].pct_change().dropna()
data = data.dropna(subset=['returns'])

# Fit GARCH model
am = arch_model(data['returns']*100, vol='GARCH', p=1, q=1)
res = am.fit(update_freq=5)
data['cond_vol'] = res.conditional_volatility

# Test Volatility Model on New Data
# Fetch data from a week ago
start_date_next = one_week_ago.strftime('%Y-%m-%d')
end_date_next = today.strftime('%Y-%m-%d')

data_next = fetch_polygon_data(ticker, start_date_next, end_date_next, API_KEY)
data_next['mid_price'] = (data_next['bid'] + data_next['ask']) / 2
data_next['returns'] = data_next['mid_price'].pct_change().dropna()
data_next = data_next.dropna(subset=['returns'])

# Forecast volatility
forecasts = res.forecast(horizon=len(data_next))
data_next['cond_vol_forecast'] = forecasts.variance.values[-1, :]

# Update Covariance Matrix Based on MSE
prices_next = data_next['mid_price'].values
state_means_price_next, state_covariances_price_next = kf_price.filter(prices_next)
data_next['predicted_price'] = state_means_price_next

# Compute MSE and update Kalman Filter
mse = mean_squared_error(data_next['mid_price'], data_next['predicted_price'])
new_observation_covariance = kf_price.observation_covariance * mse
kf_price.observation_covariance = new_observation_covariance

# Re-run the filter on combined data
combined_prices = np.concatenate((prices, prices_next))
state_means_combined, state_covariances_combined = kf_price.filter(combined_prices)
data_combined = pd.concat([data, data_next], ignore_index=True)
data_combined['predicted_price'] = state_means_combined

# Compare Volatility Model with Phase Space Analysis
# Phase Space Reconstruction
def reconstruct_phase_space(data_series, delay, dimension):
    n = len(data_series)
    phase_space = np.zeros((n - (dimension - 1) * delay, dimension))
    for i in range(dimension):
        phase_space[:, i] = data_series[i * delay : n - (dimension - 1 - i) * delay]
    return phase_space

delay = 1
dimension = 2
returns_series = data['returns'].values
phase_space = reconstruct_phase_space(returns_series, delay, dimension)

# Use Lyapunov Exponents for Market Stability Metrics
def compute_lyapunov_exponent(data_series, delay, dimension, max_iter=1000):
    from sklearn.neighbors import NearestNeighbors
    phase_space = reconstruct_phase_space(data_series, delay, dimension)
    n = len(phase_space)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(phase_space)
    distances, indices = nbrs.kneighbors(phase_space)
    nearest_neighbors = indices[:,1]
    le = []
    for i in range(n - max_iter):
        j = nearest_neighbors[i]
        dist = np.linalg.norm(phase_space[i] - phase_space[j])
        if dist == 0:
            continue
        le.append(np.log(dist))
    le_mean = np.mean(le)
    return le_mean

lyapunov_exponent = compute_lyapunov_exponent(returns_series, delay=1, dimension=2)
lyapunov_threshold = 0.01

if lyapunov_exponent > lyapunov_threshold:
    params = res.params.copy()
    params['omega'] *= 1.5
    am2 = arch_model(data['returns']*100, vol='GARCH', p=1, q=1)
    res2 = am2.fit(starting_values=params.values, update_freq=5)
    data['cond_vol_adjusted'] = res2.conditional_volatility
else:
    data['cond_vol_adjusted'] = data['cond_vol']

# Step 7: Use Models for HFT and Phase Space Analysis for Trends
# Generate trading signals based on KFPPM
data['signal'] = np.where(data['predicted_price'] > data['mid_price'], 1,
                          np.where(data['predicted_price'] < data['mid_price'], -1, 0))

# Adjust position sizing based on volatility
data['position_size'] = 1 / data['cond_vol_adjusted']
data['position_size'] = data['position_size'].clip(0, 10)

# Calculate strategy returns
data['strategy_returns'] = data['returns'] * data['signal'].shift(1) * data['position_size'].shift(1)
data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()

# Output plots and summaries as needed
