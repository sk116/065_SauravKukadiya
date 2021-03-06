import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np 
import pandas as pd 
import io
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount("/content/drive")

from sklearn import datasets, preprocessing

main_data = pd.read_csv("/content/drive/MyDrive/Sem7_ML_Data/LAB06/BuyComputer.csv")

main_data.drop(columns = ['User ID',], axis = 1,inplace = True)
main_data.head()

#Declare label as last column in the source file

Y = main_data.iloc[:,-1].values

print(Y)

#Declaring X as all columns excluding last

X = main_data.iloc[:, :-1].values

X

#Splitting data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 54)

# Scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Variabes to calculate sigmoid function

y_pred = []
x_length = len(X_train[0])
w = []
b = 0.2
print(x_length)

entries = len(X_train[:,0])
entries

for weights in range(x_length):
    w.append(0)
w

# Sigmoid function
def sigmoid(z):
  return 1/(1 + np.exp(-z))

def predict(inputs):
    return sigmoid(np.dot(w, inputs) + b)

#Loss function
def loss_func(y, a):
    J = -(y * np.log(a) + (1 - y) * np.log(1 - a))
    return J

dw = []
db = 0
J = 0
alpha = 0.1
for x in range(x_length):
    dw.append(0)

#Repeating the process 3000 times
for iterations in range(3500):
    for i in range(entries):
        localx = X_train[i]
        a = predict(localx)
        dz = a - y_train[i]
        J += loss_func(y_train[i], a)
        for j in range(x_length):
            dw[j] = dw[j]+(localx[j]*dz)
        db += dz
    J = J/entries
    db = db/entries
    for x in range(x_length):
        dw[x]=dw[x]/entries
    for x in range(x_length):
        w[x] = w[x]-(alpha*dw[x])
    b = b-(alpha*db)         
    J=0

#print bias
print(b)

#predicting the label
for x in range(len(y_test)):
    y_pred.append(predict(X_test[x]))

#print actual and predicted values in a table
for x in range(len(y_pred)):
    if y_pred[x] >= 0.5:
        y_pred[x] = 1
    else:
        y_pred[x] = 0

# Calculating accuracy of prediction
count = 0
for x in range(len(y_pred)):
    if(y_pred[x] == y_test[x]):
        count = count + 1
print('Accuracy:',(count/(len(y_pred))) * 100)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state = 0)

#Fit
LR.fit(X_train, y_train)

#predicting the test label with LR. Predict always takes X as input
y_predLR=LR.predict(X_test)
#y_predLR
prediction_count = 0
for x in range(len(y_pred)):
    if(y_pred[x]==y_test[x]):
        prediction_count=prediction_count+1
print('Accuracy:',(prediction_count/(len(y_pred)))*100)

