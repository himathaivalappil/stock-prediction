'''
Project: Predict stock prices
'''

import os
import numpy as np
import pandas as pd
from datetime import time
from matplotlib import pyplot as plt

np.random.seed(42)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential, Model
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate

import tensorflow as tf 
tf.random.set_random_seed(42)

import datetime as dt
import urllib.request, json

# Loading data from csv file
filename = "stock_market_data-AAPL.csv"
df = pd.read_csv(os.path.join(os.getcwd(), filename))
print(df.head())

# Sort dataframe based on date (Timeseries data)
stockprices = df.sort_values('Date')
print(df.head())

#### Define helper functions to calculate the metrics RMSE and MAPE ####
def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)  
    """
    rmse = np.sqrt(np.mean((y_true-y_pred)**2))                   
    return rmse

### The effectiveness of prediction method is measured in terms of the Mean Absolute Percentage Error (MAPE) and RMSE
def calculate_mape(y_true, y_pred): 
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)    
    mape = np.mean(np.abs((y_true-y_pred) / y_true))*100    
    return mape

## Split the time-series data into training seq X and output value Y
def extract_seqX_outcomeY(data, N, offset):
    """
    Split time-series into training sequence X and outcome value Y
    Args:
        data - dataset 
        N - window size, e.g., 60 for 60 days 
        offset - position to start the split
    """
    X, y = [], []
    
    for i in range(offset, len(data)):
        X.append(data[i-N:i])
        y.append(data[i])
    
    return np.array(X), np.array(y)
#### Train-Test split for time-series ####
test_ratio = 0.2
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len(stockprices))
test_size = int(test_ratio * len(stockprices))

print("train_size: " + str(train_size))
print("test_size: " + str(test_size))

train = stockprices[:train_size][['Date', 'Close']]
test = stockprices[train_size:][['Date', 'Close']]



###================= simple MA
stockprices = stockprices.set_index('Date')

### For meduim-term trading 
def plot_stock_trend(var, cur_title, stockprices=stockprices, logNeptune=True, logmodelName='Simple MA'):
    ax = stockprices[['Close', var,'200day']].plot(figsize=(20, 10))
    plt.grid(False)
    plt.title(cur_title)
    plt.axis('tight')
    plt.ylabel('Stock Price ($)')
        
def calculate_perf_metrics(var, logNeptune=True, logmodelName='Simple MA'):
    ### RMSE 
    rmse = calculate_rmse(np.array(stockprices[train_size:]['Close']), np.array(stockprices[train_size:][var]))
    ### MAPE 
    mape = calculate_mape(np.array(stockprices[train_size:]['Close']), np.array(stockprices[train_size:][var]))
   
    return rmse, mape

# 20 days to represent the 22 trading days in a month
window_size = 50
CURRENT_MODEL = 'LSTM'

if CURRENT_MODEL == 'SMA':  
    window_var = str(window_size) + 'day'
    
    stockprices[window_var] = stockprices['Close'].rolling(window_size).mean()
    ### Include a 200-day SMA for reference 
    stockprices['200day'] = stockprices['Close'].rolling(200).mean()
    
    ### Plot and performance metrics for SMA model
    plot_stock_trend(var=window_var, cur_title='Simple Moving Averages', logmodelName='Simple MA')
    rmse_sma, mape_sma = calculate_perf_metrics(var=window_var, logmodelName='Simple MA')
    print(rmse_sma, mape_sma)
    
elif CURRENT_MODEL == 'EMA':  
    ###### Exponential MA
    window_ema_var = window_var+'_EMA'
    # Calculate the N-day exponentially weighted moving average
    stockprices[window_ema_var] = stockprices['Close'].ewm(span=window_size, adjust=False).mean()
    stockprices['200day'] = stockprices['Close'].rolling(200).mean()
    
    ### Plot and performance metrics for EMA model
    plot_stock_trend(var=window_ema_var, cur_title='Exponential Moving Averages', logmodelName='Exp MA')
    rmse_ema, mape_ema = calculate_perf_metrics(var=window_ema_var, logmodelName='Exp MA')

elif CURRENT_MODEL == 'LSTM':
    
    layer_units, optimizer = 50, 'adam' 
    cur_epochs = 15
    cur_batch_size = 20
    
    cur_LSTM_pars = {'units': layer_units, 
                     'optimizer': optimizer, 
                     'batch_size': cur_batch_size, 
                     'epochs': cur_epochs
                     }
        
    ## use the past N stock prices for training to predict the N+1th closing price
   
    # scale 
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(stockprices[['Close']])
    scaled_data_train = scaled_data[:train.shape[0]]
    
    X_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)
    
    ### Build a LSTM model and log model summary to Neptune ###    
    def Run_LSTM(X_train, layer_units=50, logNeptune=True, NeptuneProject=None):     
        inp = Input(shape=(X_train.shape[1], 1))
        
        x = LSTM(units=layer_units, return_sequences=True)(inp)
        x = LSTM(units=layer_units)(x)
        out = Dense(1, activation='linear')(x)
        model = Model(inp, out)
        
        # Compile the LSTM neural net
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        return model   
    
    model = Run_LSTM(X_train, layer_units=layer_units, logNeptune=True)
    
    history = model.fit(X_train, y_train, epochs=cur_epochs, batch_size=cur_batch_size, 
                        verbose=1, validation_split=0.1, shuffle=True)
    
    # predict stock prices using past window_size stock prices
    def preprocess_testdat(data=stockprices, scaler=scaler, window_size=window_size, test=test):    
        raw = data['Close'][len(data) - len(test) - window_size:].values
        raw = raw.reshape(-1,1)
        raw = scaler.transform(raw)
        
        X_test = []
        for i in range(window_size, raw.shape[0]):
            X_test.append(raw[i-window_size:i, 0])
            
        X_test = np.array(X_test)
        
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        return X_test
    
    X_test = preprocess_testdat()
    
    predicted_price_ = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price_)
    
    # Plot predicted price vs actual closing price 
    test['Predictions_lstm'] = predicted_price   
    
    # Evaluate performance
    rmse_lstm = calculate_rmse(np.array(test['Close']), np.array(test['Predictions_lstm']))
    mape_lstm = calculate_mape(np.array(test['Close']), np.array(test['Predictions_lstm']))
    

    print(rmse_lstm, mape_lstm)
    ### Plot prediction and true trends and log to Neptune         
    def plot_stock_trend_lstm(train, test, logNeptune=True):        
        fig = plt.figure(figsize = (20,10))
        plt.plot(train['Date'], train['Close'], label = 'Train Closing Price')
        plt.plot(test['Date'], test['Close'], label = 'Test Closing Price')
        plt.plot(test['Date'], test['Predictions_lstm'], label = 'Predicted Closing Price')
        plt.title('LSTM Model')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.legend(loc="upper left")

    plot_stock_trend_lstm(train, test)