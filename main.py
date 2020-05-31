import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import requests
import time
import os
from subprocess import call
from datetime import date




def stock_predict(sym):
    end_date=date.today()
    
    try:
      df_after2008 = web.DataReader(sym, data_source='yahoo', start='2009-08-01', end=end_date)
      df_before2007=web.DataReader(sym, data_source='yahoo', start='2002-01-01', end='2007-06-30')
      df=pd.concat([df_after2008,df_before2007])

    except Exceptiom as e:
      df=web.DataReader(sym, data_source='yahoo', start='-01-08-19', end=end_date)

    
    #Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])#Converting the dataframe to a numpy array
    dataset = data.values#Get /Compute the number of rows to train the model on
    training_data_len = math.ceil( len(dataset) *.6)
    
    
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)
    
    #Create the scaled training data set 
    train_data = scaled_data[0:training_data_len  , : ]#Split the data into x_train and y_train data sets
    x_train=[]
    y_train = []
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
    
    #Build the LSTM network model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=10)
    
    model.save(os.path.join("results", sym) + ".h5")
    
    
if not os.path.isdir("results"):
    os.mkdir("results")

company=pd.read_csv('company.csv')
symbol=company['Symbol']


for sym in range(150,300):
  stock_predict(symbol[sym])


	








    
