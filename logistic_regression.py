import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

data = pd.read_csv('titanic.csv')
print(data.head())

a = 'PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked'

print(a.split(','))

X = data[['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']]
print(X)
print(data.info())

Y = data['Survived']

X['Age'].fillna(data['Age'].median(skipna = True),inplace= True) 

X['Embarked'].fillna(data['Embarked'].value_counts().idxmax(),inplace= True)

X.drop('Cabin', axis= 1,inplace = True)
X.drop('Ticket',axis = 1,inplace = True)
print(data)