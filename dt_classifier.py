from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


data = pd.read_csv('car.csv')

data.columns = ('sales','maintenance','doors','persons','bootspace','safety','class')
print(data.head())
print(data.describe())

obj = preprocessing.LabelEncoder()

data['sales'] = obj.fit_transform(data['sales'])

data['maintenance'] = obj.fit_transform(data['maintenance'])

data['doors'] = obj.fit_transform(data['doors'])

data['persons'] = obj.fit_transform(data['persons'])
data['bootspace'] = obj.fit_transform(data['bootspace'])
data['safety'] = obj.fit_transform(data['safety'])

data['class'] = obj.fit_transform(data['class'])


X = data[['sales','maintenance','doors','persons','bootspace','safety']]
Y = data['class']

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size= 0.2,random_state= 40)

obj = DecisionTreeClassifier(criterion= 'entropy',random_state = 1)

obj.fit(x_train,y_train)

pre = obj.predict(x_test)

matrix = confusion_matrix(y_test,pre)

sns.heatmap(matrix,annot = True,fmt = 'd')

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()
print(classification_report(y_test,pre))
