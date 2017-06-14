#!/usr/bin/env python
import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.utils import shuffle


np.random.seed(1979)

from keras.models import Sequential
from keras.layers import Input, Lambda
from keras.layers import Cropping2D, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping,TensorBoard

from preprocess import cBatchGenerator



def split_train_test(driving_log):
   """
   Split the driving log file to train and validate
   """
   data = pd.read_csv(driving_log)
   data = shuffle(data,random_state=1979)
   df_train, df_valid = model_selection.train_test_split(data, test_size=.2,random_state=1979)
   return (df_train, df_valid)


pather = "/Users/jagan/Desktop/SelfDrivingData/Behavioural_Cloning_data/Lap8/"
data = split_train_test(pather + "driving_log.csv")


def fit_model(data,model_name):
    df_train, df_valid = data

    # use keras to implement NVIDIA architecture,  5 CNN layers, dropout and 4 dense layer

    es = EarlyStopping(monitor='val_loss', min_delta=0.0,patience=1,verbose=1, mode='auto')

    model = Sequential()
    model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20),(0,0)))) # (top,bottom),(left,right)
    model.add(Conv2D(24,(5,5),strides = (2,2), activation = "relu"))
    model.add(Conv2D(36,(5,5),strides = (2,2), activation = "relu"))
    model.add(Conv2D(48,(5,5),activation = "relu"))
    model.add(Conv2D(64,(3,3),activation = "relu"))
    model.add(Conv2D(64,(3,3),activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer = "adam")

    mhist = model.fit_generator(cBatchGenerator(df_train, pather),\
    steps_per_epoch=df_train.shape[0]/128,epochs=80,\
    validation_data=cBatchGenerator(df_valid, pather),\
    callbacks=[es],validation_steps=int(df_valid.shape[0]/128),workers=25,verbose=1)
    #validation_steps=int(df_valid.shape[0]/128),workers=25,verbose=1)
    #callbacks=[es],validation_steps=int(df_valid.shape[0]/128),workers=15,verbose=1)
    print(mhist.history)
    model.save(model_name + '.h5')  # save model to .h5 file, including architechture, weights, loss, optimizer

if __name__ == "__main__":
    pather = "/Users/jagan/Desktop/SelfDrivingData/Behavioural_Cloning_data/Lap6/"
    data = split_train_test(pather + "driving_log.csv")
    fit_model(data,"Lap6_ep4_model")
