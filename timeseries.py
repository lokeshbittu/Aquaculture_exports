# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:59:02 2020

@author: medha
"""
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import TimerseriesGenerator
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.layers import Dropout




df = pd.read_csv('https://query.data.world/s/wzcpuy7osv6fmaejltglwqrylqnkwr')

df1 = df[df["ATTRIBUTE_DESC"] == "Farm Price"]

df1 = df1[["YEAR_ID","TIMEPERIOD_ID","AMOUNT"]]

df1['DATE'] = pd.to_datetime(df1.YEAR_ID.astype(str) + '/' + df1.TIMEPERIOD_ID.astype(str) + '/01').dt.date

df1 = df1[["DATE","AMOUNT"]]

df1 = df1.set_index("DATE")
#writing the timeseries algorithm for the data

#train_test_Series 
train, test = df1[:-163],df1[-163:]

Scaler = MinMaxScaler()
Scaler.fit(train)
train = Scaler.transform(train)
test = Scaler.transform(test)

n_input =12
n_features = 1
generator  = TimeseriesGenerator(train,test, length = n_input,batch_size=6)

model = Sequential()

model.add(LSTM(200,activation='relu',input_shape=(n_input,n_features)))

model.add(Dropout(0.15))

model.add(Dense(1))

model.compile(optimizer = "adam",loss = "mse")

model.fit_generator(generator,epochs = 180)

pred_list = []

batch = train[-n_input:].reshape((1,n_input,n_features))

for i in range(n_input):
    pred_list.append(model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
    
df_predict = pd.DataFrame(Scaler.inverse_transform(pred_list),index = df1[-n_input:].index,columns =["predictions"])

df_test = pd.concat([df1,df_predict],axis =1)

df_test.tail(12)

plt.figure(figsize = (20,5))
plt.plot(df_test.index,df_test['AMOUNT'])
plt.plot(df_test.index,df_test['predictions'],color = "r")
plt.show()




