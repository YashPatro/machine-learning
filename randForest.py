from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


#reading file
data = pd.read_csv('student-mat.csv')

print(data.head())
print(data.info())
print(data.describe())

# label endocing + collumns
obj = preprocessing.LabelEncoder()
c = 'school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences'

collumnData = c.split(',')
# data[school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3
# ]
print(collumnData)

for i in collumnData:
    data[i] = obj.fit_transform(data[i])

print(data['school'])

data.drop('G1',axis = 1,inplace = True)

data.drop('G2',axis = 1,inplace = True)
X = data[collumnData]
Y = data['G3']

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.2,random_state  = 5)

clf = RandomForestClassifier(n_estimators= 88)

clf.fit(Xtrain,Ytrain)

ypred = clf.predict(Xtest)

matrix = confusion_matrix(Xtest,ypred)

acc = classification_report(Ytest,ypred)

sns.heatmap(matrix,annot = True,fmt = 'd')

plt.title('Predicted vs actual')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()