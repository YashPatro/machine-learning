import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn import preprocessing
data = pd.read_csv('titanic.csv')
print(data.head())

a = 'PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked'

print(a.split(','))

X = data[['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']]

print(data.info())

Y = data['Survived']

data['Age'].fillna(data['Age'].median(skipna = True),inplace= True) 

data['Embarked'].fillna(data['Embarked'].value_counts().idxmax(),inplace= True)

data.drop('Cabin', axis= 1,inplace = True)
data.drop('Ticket',axis = 1,inplace = True)
print(data)

#new column  combing parch&sibl

data['TravelAlone'] = np.where((data['SibSp']+data['Parch'])>0,0,1)

data.drop('SibSp',axis= 1, inplace = True)
data.drop('Parch',axis = 1,inplace = True)

labelEncoder = preprocessing.LabelEncoder()

data['Sex'] = labelEncoder.fit_transform(data['Sex'])

data['Embarked']  = labelEncoder.fit_transform(data['Embarked'])

print(data.head())