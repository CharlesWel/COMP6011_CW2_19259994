
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

cvsData = pd.read_csv("heart_failure_clinical_records_dataset.csv")

inputCols = ["age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction","high_blood_pressure","platelets","serum_creatinine","serum_sodium","sex","smoking","time"]
answerCol = "DEATH_EVENT"

xData = cvsData[inputCols].values
yData = cvsData[answerCol].values

mins = xData.min(axis=0)
maxs = xData.max(axis=0)
diff = maxs - mins
diff[diff == 0] = 1

xData = (xData - mins) / diff

xTrain, xTemp, yTrain, yTemp = train_test_split(xData, yData, test_size=0.30, random_state=42)
xVal, xTest, yVal, yTest = train_test_split(xTemp, yTemp, test_size=0.50, random_state=42)

XTrain = xTrain.tolist()
YTrain = yTrain.tolist()

XVal = xVal.tolist()
YVal = yVal.tolist()

XTest = xTest.tolist()
YTest = yTest.tolist()

inputSize = len(inputCols)

def sigmoid(z):
  if z >= 0:
    return 1 / (1 + math.exp(-z))
  else:
    answer = math.exp(z)
    temp = answer / (1 + answer)
    return temp

def sigmoidDe(z):
  return z * (1 - z)

def buildLayers(numHiddenLayers, hiddenUni):
  sizes = [inputSize]
  for i in range(numHiddenLayers):
    sizes.append(hiddenUni)
  sizes.append(1)
  return sizes

def inNet(layerSizes, seed=42):
  random.seed(seed)

  weights = []
  bias = []

  for i in range(1, len(layerSizes)):
    currSize = layerSizes[i]
    pastSize = layerSizes[i - 1]

    layerA = []
    layerB = []

    for j in range(currSize):
      row = []
      for l in range(pastSize):
        row.append((random.random() - 0.5) * 0.2) 
      layerA.append(row)
      layerB.append((random.random() - 0.5) * 0.2)

    weights.append(layerA)
    bias.append(layerB)

  return weights, bias

def forwardPass(features, weights, bias):
  activations = [features]
  preActivations = []

  for i in range(len(weights)):
    prevA = activations[-1]
    layerA = []
    layerB = []

    for j in range(len(weights[i])):  
      z = bias[i][j]
      for l in range(len(prevA)):
        z += prevA[l] * weights[i][j][l]
      temp = sigmoid(z)
      layerA.append(z)
      layerB.append(temp)

    preActivations.append(layerA)
    activations.append(layerB)

  return activations, preActivations
