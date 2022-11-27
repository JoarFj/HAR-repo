#from socket import NI_NUMERICHOST
#from symbol import yield_arg
#from uuid import NAMESPACE_X500
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import random
import keras
import tensorflow as tf




#copy path of your csv file
df = pd.read_csv(r'C:\Skola\harren\joardata.csv', delimiter=',', header=None)
#data 6 measurements + labelled output (activity)
#adding titles:
df.columns =['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz','Activity']

#All activities: for now we have 3 activities: sitting, standing, walking
print(df.Activity.unique())
#window size and how big step we move our window
n_time_steps = 50 
step = 10 
segments = []
labels = []
n_features=6

#reshaping our data to arrays for model
for i in range(0,  df.shape[0]- n_time_steps, step):  

    xs = df['Ax'].values[i: i + 50]
    ys = df['Ay'].values[i: i + 50]
    zs = df['Az'].values[i: i + 50]
    gx = df['Gx'].values[i: i + 50]
    gy = df['Gy'].values[i: i + 50]
    gz = df['Gz'].values[i: i + 50]

    label = stats.mode(df['Activity'][i: i + 50])[0][0]

    segments.append([xs, ys, zs, gx, gy, gz])

    labels.append(label)
#reshape the segments which is (list of arrays) to a list
reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps, n_features)

labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

print(reshaped_segments.shape)

#split to train and test data, 80% train 20% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size = 0.2, random_state = 42)


print(X_train.shape)

#model:
'''''
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout

model = Sequential()
# RNN layer
model.add(LSTM(units = 128, input_shape = (X_train.shape[1], X_train.shape[2])))
# Dropout layer
model.add(Dropout(0.5)) 
# Dense layer with ReLu
model.add(Dense(units = 64, activation='relu'))
# Softmax layer
model.add(Dense(y_train.shape[1], activation = 'softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

#training on data:
n_epochs=50
batch_size=1024
history = model.fit(X_train, y_train, epochs = n_epochs, validation_split = 0.20, batch_size = batch_size, verbose = 1)


loss, accuracy = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
print("Test Accuracy :", accuracy)
print("Test Loss :", loss)

'''''