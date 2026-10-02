import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.datasets import load_boston
import os 
# bostonData = load_boston()
# bostonData = pd.read_csv('data.csv')

print(os.path.abspath('data.csv'))

bostonData = pd.read_csv('bostonHousing.csv')
print(bostonData.head())


#  predictor (feature) and the target variable
#average number of rooms per dwelling
X = bostonData[['rm','lstat']]
# add rmse after predict value  
# 'medv' is the median value of owner-occupied homes (target)
y = bostonData['medv']  

#Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

predictResult = model.predict(X_test)
# RMSE = np.sqrt(meansquareerror(ytest,ypredict))
# print(predictResult,y_test)
accuracy = accuracy_score(y_test,predictResult)
# print(accuracy)

