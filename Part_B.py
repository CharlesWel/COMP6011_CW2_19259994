import maths 
import pandas as pd
import matplotlib.pyplot as plt 

cvsData = pd.read_cvs("heart_failure_clinical_records_dataset.csv")

inputCols = ["age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction","high_blood_pressure","platelets","serum_creatinine","serum_sodium","sex","smoking","time"] 
answerCol = "DEATH_EVENT"

cvsData = cvsData[inputCols + [answerCol]].copy()
cvsData = cvsData.dropna(subset = inputCls + [answerCol])

xData = cvsData[inputCols].values
yData = cvsData[answerCol].values

xTrain, xTemp, yTrain, yTemp = train_test_split(xData, yData, test_size=0.30, random_state=42)
xVal, xTest, yVal, yTest = train_test_split(xTemp, yTemp, test_size=0.50, random_state=42)

XTrain = xTrain.tolist()
YTrain = yTrain.tolist()

XVal = xVal.tolist()
YVal = yVal.tolist()

XTest =  xTest.tolist()
YTest = yTest.tolist()
