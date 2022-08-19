import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
dataset=pd.read_csv('D:/Project/Bank_nifty_2022.csv')
dataset.head()
dataset['Date']=pd.to_datetime(dataset.Date)
dataset.shape
dataset.drop('Turnover (Rs. Cr)',axis=1, inplace=True)
dataset.head()
dataset.isnull().sum()
dataset.isna().any()
dataset.info()
dataset.describe()
print(len(dataset))
dataset['Open'].plot(figsize=(16,6))
x=dataset[['Open','High','Low','Shares Traded']]
y=dataset['Close']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
x_train.shape
y_train.shape
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
regressor=LogisticRegression()
regressor.fit(x_train, y_train)
#print(regressor.coef_)
#print(regressor.intercept_)
predicted=regressor.predict(x_test)
print(x_test)
predicted.shape
dframe=pd.DataFrame(y_test,predicted)
print(dframe)
dfr=pd.DataFrame({'Actual Data':y_test,'Predicted Data':predicted})
print(dfr)
from sklearn.metrics import confusion_matrix, accuracy_score
score = regressor.score(x_test,y_test)
print("Accuracy Score = ", score)
import math
print('Mean_absolute_Error: ', metrics.mean_absolute_error(y_test,predicted))
print('Mean Squared Error: ',metrics.mean_squared_error(y_test,predicted))
print('Root mean squared Error: ', math.sqrt(metrics.mean_squared_error(y_test,predicted)))
graph=dfr.head(30)
graph.plot(kind='bar')
