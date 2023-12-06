import sys
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def assembleData(look_back, pred_len):
    flareSummary = pd.read_csv("../data/xrsSummary.csv",header=0)
    
    data = flareSummary.get("XRS_B_Flux")
    
    # Normalize the data
    
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std

    X, y = formatData(look_back, pred_len, data)
    
    return X, y

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
    validInput = False
    while not validInput:
        response = input("Enter the name of the model to load: ")
        if response != "":
            modelName = response
            validInput = True
        else:
            print("Response not recognized, please try again.") 
    return modelName

def main(look_back, pred_len, modelName):

    model = load_model(f"../models/{modelName}.keras")
    
    X, y = assembleData(look_back, pred_len)

    print(" ==== Evaluating Model on ALL DATA ====")
    results = model.evaluate(X,y,batch_size=128)
    print(f"Loss = {results[0]}, Accuracy = {results[1]}")

    yHat = model.predict(X)
    
    plotPredictions = []
    plotTestY = []
    timeSteps = np.arange(len(yHat))
    for i in range(len(yHat)):
        plotTestY.append(y[i][0])
        plotPredictions.append(yHat[i][0])
    plt.plot(timeSteps,plotPredictions,'.r',label="Predicted Flux")
    plt.plot(timeSteps,plotTestY,'.b',label="Actual Flux")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized XRS_B Flux")
    plt.legend(loc="upper right")
    plt.semilogy()
    plt.savefig(f"../images/fullEvaluation/{modelName}.png")

if __name__=="__main__":
    if len(sys.argv) > 1:
        main(10,5,sys.argv[1])
