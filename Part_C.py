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

def predictProb(features, weights, bias):
  activations, preActivations = forwardPass(features, weights, bias)
  return activations[-1][0]

def predictLable(features, weights, bias):
  if predictProb(features, weights, bias) >= 0.5:
    return 1
  else:
    return 0

def accuracy(weights, bias, features, lables):
  correct = 0
  sizeA = len(features)

  for i in range(sizeA):
    if predictLable(features[i], weights, bias) == lables[i]:
      correct += 1

  return correct / sizeA

def meanSquareError(features, weights, bias, lables):
  totalSE = 0
  temp = len(features)

  for i in range(temp):
    prob = predictProb(features[i], weights, bias)
    error = prob - lables[i]
    totalSE += error * error

  return totalSE / temp

def gradientDescentDeep(features, lables, learningRate, passes, Xval, Yval, graph, numHiddenLayers, hiddenUni):
  layerSizes = buildLayers(numHiddenLayers, hiddenUni)
  weights, bias = inNet(layerSizes, seed=42)

  trainError = []
  valError = []

  for i in range(passes):
    gradA = []
    gradB = []
    for j in range(len(weights)):
      layerGradA = []
      for l in range(len(weights[j])):
        layerGradA.append([0] * len(weights[j][l]))
      gradA.append(layerGradA)
      gradB.append([0] * len(bias[j]))

    for i in range(len(features)):
      x = features[i]
      y = lables[i]

      activations, preActivations = forwardPass(x, weights, bias)

      delt = [None] * len(weights)

      avtivationsOut = activations[-1][0]
      deltaOut = (avtivationsOut - y) * sigmoidDe(avtivationsOut)
      delt[-1] = [deltaOut]


      for j in range(len(weights) - 2, -1, -1):
        currSize = len(weights[j])
        nextSize = len(weights[j + 1])

        currDelta = [0] * currSize

        for l in range(currSize):
          s = 0
          for k in range(nextSize):
            s += delt[j + 1][k] * weights[j + 1][k][l]

          a = activations[j + 1][l]  
          currDelta[l] = s * sigmoidDe(a)

        delt[j] = currDelta

      for r in range(len(weights)):
        prevA = activations[r]      
        currDelta = delt[r]      

        for a in range(len(weights[r])):  
          gradB[r][a] += currDelta[a]
          for b in range(len(weights[r][a])):  
            gradA[r][a][b] += currDelta[a] * prevA[b]

    meanGrad = 1 / len(features)

    for c in range(len(weights)):
      for d in range(len(weights[c])):
        bias[c][d] -= learningRate * (gradB[c][d] * meanGrad)
        for e in range(len(weights[c][d])):
          weights[c][d][e] -= learningRate * (gradA[c][d][e] * meanGrad)

    if graph == True:
      trainError.append(meanSquareError(features, weights, bias, lables))

      if Xval != None and Yval != None:
        valError.append(meanSquareError(Xval, weights, bias, Yval))

  if graph == False:
    return weights, bias
  else:
    return weights, bias, trainError, valError

def displayLearningCurve(trainError, valError, title):
  epochs = range(1, len(trainError) + 1)

  plt.plot(epochs, trainError, label="Training Error")

  if valError != None and len(valError) > 0:
    plt.plot(epochs, valError, label="Validation Error")

  plt.xlabel("Epochs")
  plt.ylabel("Mean Squared Error")
  plt.title(title)
  plt.legend()
  plt.grid(True)
  plt.show()

print("Enter 1 to run networks (2, 4, 6 layers)")
print("Enter 2 to run networks and display learning curves (2, 4, 6 layers)")
print("Enter 3 to find the best hyperparamites (for ONE chosen depth)")
print()
#choice = int(input(""))
choice = 3

if choice == 1 or choice == 2:
  learningRate = 0.25
  passes = 350
  hiddenUni = 16

  layerOptions = [2, 4, 6] 

  for numHiddenLayers in layerOptions:
    weights, bias, trainError, valError = gradientDescentDeep(XTrain, YTrain, learningRate, passes, XVal, YVal, True,numHiddenLayers, hiddenUni)

    trainingAccuracy = accuracy(weights, bias, XTrain, YTrain)
    validationAccuracy = accuracy(weights, bias, XVal, YVal)

    print("Hidden layers:", numHiddenLayers, " Hidden units per layer:", hiddenUni)
    print("Training classification rate:", trainingAccuracy)
    print("Validation classification rate:", validationAccuracy)

    if choice == 2:
      displayLearningCurve(trainError, valError,"Learning Curves (" + str(numHiddenLayers) + " Hidden Layers)")

else:
  learningRateOptions = [0.001, 0.005, 0.02, 0.025, 0.01, 0.05, 0.1]
  passesOptions = [1, 5, 10, 15, 20, 25, 30, 50]
  hiddenUnitsOptions = [2, 4, 6]

  numHiddenLayers = 4

  testNum = 1
  best = -1
  bestValues = [0, 0]
  bestDiff = -1
  bestParams = [0, 0, 0]

  for i in range(len(hiddenUnitsOptions)):
    for j in range(len(learningRateOptions)):
      for l in range(len(passesOptions)):
        weights, bias = gradientDescentDeep(
          XTrain, YTrain, learningRateOptions[j], passesOptions[l],
          XVal, YVal, False, numHiddenLayers, hiddenUnitsOptions[i]
        )

        trainingAccuracy = accuracy(weights, bias, XTrain, YTrain)
        validationAccuracy = accuracy(weights, bias, XVal, YVal)

        print("Test", testNum)
        print("hidden units:", hiddenUnitsOptions[i])
        print("learning rate:", learningRateOptions[j])
        print("number of passes: ", passesOptions[l])
        print("Training classification rate:", trainingAccuracy)
        print("Validation classification rate: ", validationAccuracy)
        print()

        if (validationAccuracy > bestValues[1]) or (validationAccuracy == bestValues[1] and (trainingAccuracy - validationAccuracy) < bestDiff):
          best = testNum
          bestValues[0] = trainingAccuracy
          bestValues[1] = validationAccuracy
          bestDiff = (trainingAccuracy - validationAccuracy)
          bestParams = [hiddenUnitsOptions[i], learningRateOptions[j], passesOptions[l]]

        testNum += 1

  print("The best test was test", best, "for", numHiddenLayers, "hidden layers")
  print("Which has the results:")
  print("Training classification rate:", bestValues[0])
  print("Validation classification rate: ", bestValues[1])
  print("Best params: hidden unit: ", bestParams[0], " learningRate: ", bestParams[1], " passes: ", bestParams[2])

