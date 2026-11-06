import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix



data = pd.read_csv('iris.csv')
print(data.head())

data = data.ffill().bfill()

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = preprocessing.MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xtrain, ytrain)

predictions = knn.predict(xtest)
matrix = confusion_matrix(ytest,predictions)
print(confusion_matrix(ytest, predictions))
print(classification_report(ytest, predictions))



sns.heatmap(matrix,annot= True,fmt = 'd')

plt.title('confusion_matrix')

plt.xlabel('predicted')
plt.ylabel('actual')

plt.show()