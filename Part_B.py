import math 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

cvsData = pd.read_csv("heart_failure_clinical_records_dataset.csv")

inputCols = ["age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction","high_blood_pressure","platelets","serum_creatinine","serum_sodium","sex","smoking","time"] 
answerCol = "DEATH_EVENT"

xData = cvsData[inputCols].values
yData = cvsData[answerCol].values

mins = xData.min()
maxs = xData.max()
xData = (xData - mins) / (maxs - mins)

xTrain, xTemp, yTrain, yTemp = train_test_split(xData, yData, test_size=0.30, random_state=42)
xVal, xTest, yVal, yTest = train_test_split(xTemp, yTemp, test_size=0.50, random_state=42)

XTrain = xTrain.tolist()
YTrain = yTrain.tolist()

XVal = xVal.tolist()
YVal = yVal.tolist()

XTest =  xTest.tolist()
YTest = yTest.tolist()

def sigmoid(x):
  temp = 1 / (1 + math.exp(-x))
  
  return temp

def predictProb(x, y, a):
  temp = y
  
  for i in range(len(x)):
    temp += x[i] * a[i]
    
  return sigmoid(temp)

def predictLable(x, y, a):
  if predictProb(x, y, a) >= 0.5:
    return 1
    
  else:
    return 0


def accuracy(x, y, a, b):
  correct = 0
  sizeA = len(a)
  
  for i in range(sizeA):
    if predictLable(x, y, a[i]) == b[i]:
      correct += 1
      
  temp = correct / sizeA
  
  return temp

def meanSquareError(x, y, a, b):
  totalSE = 0
  temp = len(a)
  
  for i in range(temp):
    prob = predictProb(x, y, a[i])
    error = prob - b[i]
    totalSE += error * error
  return totalSE / temp

def gradientDescent(a, b, learningRate, passes):
  weightVector = [0] * 12
  bias = 0
  
  for i in range(passes):
    gradientLoss = [0] * 12
    gradientLossBias = 0
    
    for j in range(len(a)):
      prob = predictProb(weightVector, bias, a[j])
      error = prob - b[j]
      
      for l in range(12):
        gradientLoss[l] += error * a[j][l]
        
      gradientLossBias += error
      
    meanGrad = 1 / len(a)
    
    for k in range(12):
      weightVector[k] -= learningRate * (gradientLoss[k] * meanGrad)
      
    bias -= learningRate * (gradientLossBias * meanGrad)
  
  return weightVector, bias 

print("Enter 1 to")
print("Enter 2 to")
#choice = int(input(""))
choice = 1

if choice == 1:
  weightvector, bias = gradientDescent(XTrain, YTrain, 0.1, 100)

  trainingAccuracy = accuracy(weightvector, bias, XTrain,  YTrain)
  validationAccuracy = accuracy(weightvector, bias, XVal, YVal)

  print("Training classification rate:", trainingAccuracy)
  print("Validation classification rate", validationAccuracy)
