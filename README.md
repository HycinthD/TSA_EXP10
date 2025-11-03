# Exp.no: 10 IMPLEMENTATION OF SARIMA MODEL  
# Date: 03/11/2025  

# AIM:  
To implement SARIMA model using Python on Amazon stock dataset.  

# ALGORITHM:  
1. Explore the dataset  
2. Check for stationarity of the time series  
3. Determine SARIMA model parameters (p, q, P, Q)  
4. Fit the SARIMA model  
5. Make time series predictions and auto-fit the SARIMA model  
6. Evaluate model predictions  

# PROGRAM:
```
#Name: Hycinth D
#Reg No: 212223240055
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load Amazon stock dataset
data = pd.read_csv('/content/Amazon.csv')

# Convert 'Date' column to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use 'Close' price for forecasting and resample monthly
data = data[['Close']].resample('M').mean()

# Display first few rows
print(data.head())

# Plot the monthly closing price
plt.figure(figsize=(10,5))
plt.plot(data.index, data['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Amazon Monthly Closing Price')
plt.show()

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Check stationarity of Close price
check_stationarity(data['Close'])

# Plot ACF and PACF
plot_acf(data['Close'])
plt.show()

plot_pacf(data['Close'])
plt.show()

# Train-test split
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

# Fit SARIMA model
sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_result = sarima_model.fit(disp=False)

# Forecast
predictions = sarima_result.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# RMSE calculation
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted values
plt.figure(figsize=(10,5))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('SARIMA Model Predictions on Amazon Stock Data')
plt.legend()
plt.show()

print("\nRESULT:")
print("Thus, the SARIMA model was successfully implemented using the Amazon stock dataset.")
```

# Output:
<img width="859" height="470" alt="download" src="https://github.com/user-attachments/assets/ea4701b8-cb5a-4fac-8633-b68e7749471c" />
<img width="568" height="435" alt="download" src="https://github.com/user-attachments/assets/fc5393fa-11c2-4c4d-be9f-cccab9c1a280" />
<img width="568" height="435" alt="download" src="https://github.com/user-attachments/assets/12174a12-2966-4b52-a0e9-b8658f398f7e" />
<img width="866" height="470" alt="download" src="https://github.com/user-attachments/assets/82fd8520-1303-4b5c-99c9-8d502be8fdad" />

# RESULT:
Thus, the SARIMA model was successfully implemented using the Amazon stock dataset.
