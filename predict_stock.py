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

import helper

# Loading data from csv file
filename = "stock_market_data-AAPL.csv"
df = pd.read_csv(os.path.join(os.getcwd(), filename))
print(df.head())

# Sort dataframe based on date (Timeseries data)
stockprices = df.sort_values('Date')
print(df.head())

#### Train-Test split for time-series ####
test_ratio = 0.2
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len(stockprices))
test_size = int(test_ratio * len(stockprices))

print("train_size: " + str(train_size))
print("test_size: " + str(test_size))

train = stockprices[:train_size][['Date', 'Close']]
test = stockprices[train_size:][['Date', 'Close']]