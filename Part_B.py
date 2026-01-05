import math 
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

XTest =  xTest.tolist()
YTest = yTest.tolist()

def sigmoid(z):
  temp = 1 / (1 + math.exp(-z))
  
  return temp

def predictProb(featrues, bias, weightVec):
  temp = bias
  
  for i in range(len(featrues)):
    temp += featrues[i] * weightVec[i]
    
  return sigmoid(temp)

def predictLable(featrues, bias, weightVec):
  if predictProb(featrues, bias, weightVec) >= 0.5:
    return 1
    
  else:
    return 0

def accuracy(weightVec, bias, featrues, lables):
    correct = 0
    sizeA = len(featrues)

    for i in range(sizeA):
        if predictLable(featrues[i], bias, weightVec) == lables[i]:
            correct += 1

    return correct / sizeA

def meanSquareError(featrues, bias, weightVec, lables):
  totalSE = 0
  temp = len(featrues)
  
  for i in range(temp):
    prob = predictProb(featrues[i], bias, weightVec)
    error = prob - lables[i]

    totalSE += error * error
  return totalSE / temp

def gradientDescent(featrues, lables, learningRate, passes, Xval, Yval, graph):
  weightVector = [0] * 12
  bias = 0
  trainError = []
  valError = []
  
  for i in range(passes):
    gradientLoss = [0] * 12
    gradientLossBias = 0
    
    for j in range(len(featrues)):
      prob = predictProb(featrues[j], bias, weightVector)
      error = prob - lables[j]
      
      for l in range(12):
        gradientLoss[l] += error * featrues[j][l]
        
      gradientLossBias += error
      
    meanGrad = 1 / len(featrues)
    
    for k in range(12):
      weightVector[k] -= learningRate * (gradientLoss[k] * meanGrad)
      
    bias -= learningRate * (gradientLossBias * meanGrad)

    if graph == True:
      trainError.append(meanSquareError(featrues, bias, weightVector, lables))

      if Xval != None and Yval != None:
        valError.append(meanSquareError(XVal, bias, weightVector, YVal))
  
  if graph == False:
    return weightVector, bias 
  else:
    return weightVector, bias, trainError, valError

def displayLearningCurve(trainError, valError):
  epochs = range(1, len(trainError) + 1)

  plt.plot(epochs, trainError, label="Training Error")

  if valError != None and len(valError) > 0:
    plt.plot(epochs, valError, label="Validation Error")

  plt.xlabel("Epochs")
  plt.ylabel("Mean Squared Error")
  plt.title("Learning Curves")
  plt.legend()
  plt.grid(True)
  plt.show()

print("Enter 1 to run network")
print("Enter 2 to run network and display learning curve")
print("Enter 3 to find the best hyperparamites")
print()
#choice = int(input(""))
choice = 3

if choice == 1 or choice == 2:
  weightvector, bias, trainError, valError = gradientDescent(XTrain, YTrain, 0.5, 200, XVal, YVal, True)

  trainingAccuracy = accuracy(weightvector, bias, XTrain,  YTrain)
  validationAccuracy = accuracy(weightvector, bias, XVal, YVal)

  print("Training classification rate:", trainingAccuracy)
  print("Validation classification rate", validationAccuracy)

  if choice == 2:
    displayLearningCurve(trainError, valError)

else:
  learningRateOptions = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
  passesOptions = [20, 50, 100, 200, 250, 300, 350, 400, 450, 500] 
  testNum = 1
  best = -1
  bestValues = [0,0]
  bestDiff = -1

  for i in range(len(learningRateOptions)):
    for j in range(len(passesOptions)):
      weightvector, bias = gradientDescent(XTrain, YTrain, learningRateOptions[i], passesOptions[j], XVal, YVal, False)

      trainingAccuracy = accuracy(weightvector, bias, XTrain,  YTrain)
      validationAccuracy = accuracy(weightvector, bias, XVal, YVal)

      print("Test", testNum," learning rate:", learningRateOptions[i], " number of passes", passesOptions[j])
      print("Training classification rate:", trainingAccuracy)
      print("Validation classification rate", validationAccuracy)
      print("Bias rate:", bias)
      print("Weight vector:", weightvector)
      print()

      if (validationAccuracy > bestValues[1]) or (validationAccuracy == bestValues[1] and (trainingAccuracy - validationAccuracy) < best):
        best = testNum
        bestValues[0] = trainingAccuracy
        bestValues[1] = validationAccuracy
        bestDiff = (trainingAccuracy - validationAccuracy)
      
      testNum += 1
      
  
  print("The best test was test ",best)
  print("Which has the results:")
  print("Training classification rate:", bestValues[0])
  print("Validation classification rate", bestValues[1])


