# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:54:21 2023

@author: Drew Watson
"""

import sys
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Conv1D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def assembleData(trainSplit=0.7,validSplit=0.2,look_back=10, predict_len=3):
    flareSummary = pd.read_csv("../data/xrsSummary.csv",header=0)
    date_time = pd.to_datetime(flareSummary.pop('Date_Time'), format='%Y-%m-%d %H:%M:%S')

    # Convert timestamps to seconds
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    
    # Splitting Data
    
    N = len(flareSummary)
    testSplit = 1.0 - trainSplit - validSplit
    
    trainData = flareSummary[0:int(N*trainSplit)].get("XRS_B_Flux")
    validData = flareSummary[int(N*trainSplit):int(N*(trainSplit+validSplit))].get("XRS_B_Flux")
    testData = flareSummary[int(N*(trainSplit+validSplit)):].get("XRS_B_Flux")
    
    

    num_features = flareSummary.shape[1]
    
    # Normalize the data
    
    train_mean = trainData.mean()
    train_std = trainData.std()
    
    trainData = (trainData - train_mean) / train_std
    validData = (validData - train_mean) / train_std
    testData = (testData - train_mean) / train_std
    
    # Window the data (Split into input and output)    
    trainX, trainY = formatData(look_back,predict_len,trainData)
    validX, validY = formatData(look_back,predict_len,validData)
    testX, testY = formatData(look_back,predict_len,testData)

    return trainX, trainY, validX, validY, testX, testY

    
# From a set of data, create inputs and outputs to be supplied to RNN
def formatData(look_back,predict_len,data):
    X,y = [], []
    for i in range(len(data) - look_back - predict_len):
        d = i + look_back
        X.append([data[i:d]])
        y.append(np.array(data[d:d+predict_len]))
    X = np.array(X)
    y = np.array(y)

    return X, y

# Define the RNN Model to train on
def createModel(look_back,pred_len):
    model = Sequential()
    model.add(SimpleRNN(units=64,input_shape=(1,look_back), activation="relu", return_sequences=True))
    # model.add(Conv1D(filters=10, kernel_size=2, activation="relu"))
    model.add(SimpleRNN(32))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(300, activation="relu"))
    model.add(Dense(300,activation="relu"))
    model.add(Dense(150, activation="relu"))
    model.add(Dense(pred_len))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy','mse'])
    return model

def getInput():
    persist = True
    validInput = False
    while not validInput:
        response = input("Do you want to persist the RNN? (y/n)")
        if response == "y" or response == "Y":
            persist = True
            validInput = True
        elif response == "N" or response == "n":
            persist = False
            validInput = True
        else:
            print("Response not recognized, please try again.")
    validInput = False
    dirName = ""
    while not validInput:
        dirName = input("Please enter the name of the directory you want to persist the RNN to: ")
        if dirName != "":
            validInput = True
        else:
            print("Response not recognized, please try again") 
    return persist, dirName   

if __name__=="__main__":
    LOOK_BACK = 10
    PRED_LEN = 5
    TRAIN_SPLIT = 0.7
    VALID_SPLIT = 0.2
    EPOCHS = 1000
    if len(sys.argv) >= 2:
        PERSIST = True
        dirName = sys.argv[1]
    else:
        PERSIST = False
        dirName = ""
    # PERSIST, dirName = getInput()

    trainX, trainY, validX, validY, testX, testY = assembleData(TRAIN_SPLIT,VALID_SPLIT,LOOK_BACK,PRED_LEN)
    model = createModel(LOOK_BACK,PRED_LEN)
    model.fit(trainX, 
              trainY, 
              epochs=EPOCHS, 
              verbose=1,
              validation_data=(validX,validY),
              shuffle=False,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

    if PERSIST:
        model.save(f"../models/{dirName}")
    
    print(" ==== Evaluating Model ====")
    results = model.evaluate(testX,testY,batch_size=128)
    print(f"Test Loss = {results[0]}, Test Accuracy = {results[1]}")
    

    predictions = model.predict(testX)

    print("===== Generating Plot =====")

    plotPredictions = []
    plotTestY = []
    timeSteps = np.arange(len(predictions))
    for i in range(len(predictions)):
        plotTestY.append(testY[i][0])
        plotPredictions.append(predictions[i][0])
    plt.plot(timeSteps,plotPredictions,'.r',label="Predicted Flux")
    plt.plot(timeSteps,plotTestY,'.b',label="Actual Flux")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized XRS_B Flux")
    plt.legend(loc="upper right")
    plt.semilogy()
    if PERSIST:
        plt.savefig(f"../images/{dirName}.png")
    plt.show()
    