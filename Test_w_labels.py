import numpy as np
import tensorflow as tf

print("hello ")
from socket import NI_NUMERICHOST
from symbol import yield_arg
from uuid import NAMESPACE_X500
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import random
import keras
from tensorflow import lite
import tensorflow as tf

import glob
import pandas as pd
import os
# Get CSV files list from a folder

pathname= 'C:\\SKOLA\\harren\\useData'
#filenames should be named "RAW_xxxxxxxx"
print(glob.glob('useData/RAW*.csv'))
# Concatenate all DataFrames

alldf= []
for one_filename in glob.glob('useData/RAW*.csv'):
    print(f'loading {one_filename}')
    new_df= pd.read_csv(one_filename, header=None)
    alldf.append(new_df)

df= pd.concat(alldf)

df.columns =['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz','Activity']

print(df[0:].shape)


print('remove gyroscope data (y/n)?')
x = input()
#print(x)

if x == 'y':
    print('confirmed')
    del df['Gx']
    del df['Gy']
    del df['Gz']
    n_features=3
else:
    n_features=6

#if gyro is included we have 7 features, else we have 4
print(df[0:].shape, n_features)

#All unique activities: 
print(df.Activity.unique())
#window size and how big step we move our window
n_time_steps =250
step = 20
segments = []
labels = []


#reshaping our data to arrays for model
for i in range(0,  df.shape[0]- n_time_steps, step):  

    xs = df['Ax'].values[i: i + n_time_steps]
    ys = df['Ay'].values[i: i + n_time_steps]
    zs = df['Az'].values[i: i + n_time_steps]
    
    if n_features==6:

        gx = df['Gx'].values[i: i + n_time_steps]
        gy = df['Gy'].values[i: i + n_time_steps]
        gz = df['Gz'].values[i: i + n_time_steps]

    label = stats.mode(df['Activity'][i: i + n_time_steps])[0][0]

    if n_features == 6:
        segments.append([xs, ys, zs, gx, gy, gz])
    elif n_features == 3:
        segments.append([xs, ys, zs])

    labels.append(label)

#reshape the segments which is (list of arrays) to a list

reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps, n_features)
print(reshaped_segments.shape)

labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)
#labels = np.asarray(labels)
print(labels.shape)


#split to train and test data, 80% train 20% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size = 0.2, random_state = 42)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
#print(X_train.shape)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D


#CNN model
model = Sequential()
model.add(Conv1D(filters = 120, kernel_size=3 ,activation='relu',input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(Conv1D(filters=60, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.4))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
# Softmax layer
model.add(Dense(y_train.shape[1], activation = 'softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

'''
##LSTM
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout

model = Sequential()
# RNN layer
model.add(LSTM(units = 64, input_shape = (X_train.shape[1], X_train.shape[2])))
# Dropout layer
model.add(Dropout(0.5)) 
# Dense layer with ReLu
model.add(Dense(units = 128, activation='relu'))
# Softmax layer
model.add(Dense(y_train.shape[1], activation = 'softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
'''

model.fit(X_train, y_train, epochs = 15, validation_split = 0.10, batch_size = 1024, verbose = 1)

#print(X_train.shape)
#print(y_train.shape)
#print("xsharone")
#print(X_test[1:2].shape)
print(model.evaluate(X_test, y_test, verbose=0))
run_model = tf.function(lambda x: model(x))
# This is important, let's fix the input size.
BATCH_SIZE = 1
STEPS = n_time_steps
INPUT_SIZE = n_features
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))

# model directory.
MODEL_DIR = "cnn_lstm"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)


#converting shiit
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()

with open('CNNconvertedwithgyro.tflite', 'wb') as f:
    f.write(tflite_model)
# Run the model with TensorFlow to get expected results.
TEST_CASES = 10

# Run the model with TensorFlow Lite
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for i in range(TEST_CASES):
  expected = model.predict(X_test[i:i+1])
  interpreter.set_tensor(input_details[0]["index"], X_test[i:i+1, :, :])
  interpreter.invoke()
  result = interpreter.get_tensor(output_details[0]["index"])

  # Assert if the result of TFLite model is consistent with the TF model.
  np.testing.assert_almost_equal(expected, result, decimal=5)
  print("Done. The result of TensorFlow matches the result of TensorFlow Lite.")

  # Please note: TfLite fused Lstm kernel is stateful, so we need to reset
  # the states.
  # Clean up internal states.
  interpreter.reset_all_variables()
