import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Download stock data
ticker = 'TSLA'
start_date = '2018-01-01'
end_date = '2023-01-01'

data = yf.download(ticker, start=start_date, end=end_date)

# Feature Engineering: Create Technical Indicators
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()


# Add RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


data['RSI'] = calculate_rsi(data)

# Lag Features: Adding 1-day lag
data['Close_Lag_1'] = data['Close'].shift(1)

# Drop rows with NaN values (due to moving averages and lag features)
data.dropna(inplace=True)

# Define Features and Target
X = data[['SMA_50', 'SMA_200', 'RSI', 'Close_Lag_1']]  # Include lag feature
y = data['Close']

# Split into Train and Test sets (keep temporal order)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression Model with Hyperparameter Tuning
ridge = Ridge()

# Define a parameter grid for Ridge Regression
param_grid = {'alpha': np.logspace(-5, 5, 10)}  # Search for alpha between 10^-5 and 10^5

# Perform Grid Search for Best Alpha
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best Alpha from Grid Search
best_alpha = grid_search.best_params_['alpha']
print(f"Best hyperparameter: alpha = {best_alpha}")

# Train Ridge Regression with the Best Alpha
best_ridge = grid_search.best_estimator_
best_ridge.fit(X_train_scaled, y_train)

# Predictions on the Test Set
y_pred = best_ridge.predict(X_test_scaled)

# Evaluation Metrics
r2_train = best_ridge.score(X_train_scaled, y_train)
r2_test = r2_score(y_test, y_pred)

print(f'R-squared (Training Set): {r2_train:.4f}')
print(f'R-squared (Test Set): {r2_test:.4f}')

# Plot the Actual vs Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Prices', color='red')
plt.title(f'{ticker} Actual vs Predicted Closing Prices')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.show()

# Plot the Technical Indicators
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.7)
plt.plot(data.index, data['SMA_50'], label='50-Day SMA', color='orange', alpha=0.7)
plt.plot(data.index, data['SMA_200'], label='200-Day SMA', color='green', alpha=0.7)
plt.title(f'{ticker} Stock Price and Technical Indicators')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.show()

# Predict next day price (Optional)
last_row = data.iloc[-1:]
X_future = scaler.transform(last_row[['SMA_50', 'SMA_200', 'RSI', 'Close_Lag_1']].values.reshape(1, -1))
predicted_price = best_ridge.predict(X_future)
print(f'Predicted next day closing price for {ticker}: ${predicted_price[0]:.2f}')


# What it is: The RSI is a momentum oscillator that measures the speed and change of price movements.

# Close_Lag_1 refers to the closing price of the stock from the previous day (or a certain previous period), often used as a lagged feature in time-series analysis

# Ridge: A linear regression model with L2 regularization to reduce overfitting