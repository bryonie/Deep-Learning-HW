import os
import glob
import pickle as pkl
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D, TimeDistributed, LSTM
from keras.models import Model, load_model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

warnings.filterwarnings("ignore")

raw_signal_dir = 'input/Data_Raw_signals.pkl'
spectrogram_dir = 'input/Data_Spectrograms.pkl'

with open(raw_signal_dir, 'rb') as raw:
    raw_signal_data = pkl.load(raw)

with open(spectrogram_dir, 'rb') as specs:
    spectrogram_data = pkl.load(specs)


print(type(raw_signal_data))
print(len(raw_signal_data))
# print(raw_signal_data.shape)
print(type(raw_signal_data[0]))
print(type(raw_signal_data[1]))
print(raw_signal_data[1].shape)
print(len(raw_signal_data[1]))
print(raw_signal_data[0].shape)

for i in range (125,130):
    print(raw_signal_data[1][i])
    print(raw_signal_data[0][i].shape)

classes = raw_signal_data[1]
raw_signals = raw_signal_data[0]

le = LabelEncoder()

labels = le.fit_transform(classes)
labels = np_utils.to_categorical(labels, 
num_classes=len(classes))

# Splitting data from train into train and test data
x_train, x_val, y_train, y_val = train_test_split(
    raw_signals,
labels,stratify = labels, train_size = 0.7, test_size = 0.3,
random_state=777,shuffle=True)

print(x_train.shape)
print(x_val.shape)

# Building 1D Convolution model
shape = x_train.shape
inputs = Input(shape=(shape[1], shape[2]))

##
# 1D Convolution + LSTM Modelmodel working with data 80:20, batch=32
model = Sequential()
model.add(Conv1D(16,1, activation='relu', strides=1, padding='same', input_shape=(shape[1], shape[2])))
model.add(Conv1D(32,3, activation='relu', strides=1, padding='same'))
model.add(Conv1D(64,3, activation='relu', strides=1, padding='same'))
model.add(MaxPooling1D(3))
model.add(LSTM(128, return_sequences=True, input_shape=(shape[1], shape[2])))
model.add(LSTM(65, return_sequences=True))
model.add(Dropout(0.05))
model.add(Flatten())
model.add(Dense(35, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#Setting the model name
hdf_count = len(glob.glob1('/models',"*.hdf5"))
model_name = "model(" + str(hdf_count + 1) + ").hdf5"
print("Model: {}".format(model_name))

# Setting up easy stoping and model checkpoints
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
 patience=10, min_delta=0.0001) 
mc = ModelCheckpoint(model_name, monitor='val_acc', 
verbose=1, save_best_only=True, mode='max')

# Fit the model to train

history=model.fit(x_train, y_train ,epochs=100, callbacks=[es,mc], 
batch_size=32, validation_data=(x_val,y_val), verbose=1)