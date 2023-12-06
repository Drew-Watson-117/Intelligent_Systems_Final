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
import tensorflow as tf
import fullEvaluation
from models import getModel

def assembleData(trainSplit=0.7,validSplit=0.2,look_back=10, predict_len=3):
    flareSummary = pd.read_csv("../data/xrsSummary.csv",header=0)

    # Splitting Data
    
    N = len(flareSummary)
    
    trainData = flareSummary[0:int(N*trainSplit)].get("XRS_B_Flux")
    validData = flareSummary[int(N*trainSplit):int(N*(trainSplit+validSplit))].get("XRS_B_Flux")
    testData = flareSummary[int(N*(trainSplit+validSplit)):].get("XRS_B_Flux")
    
    
    
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
    modelName = ""
    while not validInput:
        modelName = input("Please enter the name of the directory you want to persist the RNN to: ")
        if modelName != "":
            validInput = True
        else:
            print("Response not recognized, please try again") 
    return persist, modelName   

if __name__=="__main__":
    LOOK_BACK = 10
    PRED_LEN = 5
    TRAIN_SPLIT = 0.7
    VALID_SPLIT = 0.2
    EPOCHS = 1000

    if len(sys.argv) >= 2:
        PERSIST = True
        modelName = sys.argv[1]
    else:
        PERSIST, modelName = getInput()

    trainX, trainY, validX, validY, testX, testY = assembleData(TRAIN_SPLIT,VALID_SPLIT,LOOK_BACK,PRED_LEN)
    model = getModel(LOOK_BACK,PRED_LEN)
    model.fit(trainX, 
              trainY, 
              epochs=EPOCHS, 
              verbose=1,
              validation_data=(validX,validY),
              shuffle=False,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

    if PERSIST:
        model.save(f"../models/{modelName}.keras")
    
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
        plt.savefig(f"../images/training/{modelName}.png")
    else:
        plt.show()
    plt.clf()

    fullEvaluation.main(LOOK_BACK,PRED_LEN,modelName)    