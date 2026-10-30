import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
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

# data['Fare'].fillna(data['Fare'].mean(skipna=True),inplace=True)
# print(data.isnull().sum())
data.drop('Fare',axis = 1,inplace = True)
print(data.head())
X = data[['Pclass','Sex','Age','Embarked','TravelAlone']]
Y = data['Survived']
print(X.info())

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size= 0.2,random_state= True)
lr = LogisticRegression()

print(Xtest,Xtrain)

lr.fit(Xtrain,Ytrain)

Ypred = lr.predict(Xtest)

ma = confusion_matrix(Ytest,Ypred)

sns.heatmap(ma,annot= True,fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.show()
print(classification_report(Ytest,Ypred))


# #hw
# import numpy as np
# import pandas as pd
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score

# column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
# data = pd.read_csv('car.csv', names=column_names)

# print("--- Initial Data Snapshot ---")
# print(data.head())
# print("\n--- Data Information ---")
# print(data.info())

# label_encoder = preprocessing.LabelEncoder()

# for col in column_names:
#     data[col] = label_encoder.fit_transform(data[col])

# print("\n--- Data Post Label Encoding ---")
# print(data.head())

# X = data[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']]
# Y = data['class']

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# model = LogisticRegression()
# model.fit(X_train, Y_train)

# Y_pred = model.predict(X_test)

# print("\n--- Model Evaluation ---")
# print(f"Accuracy Score: {accuracy_score(Y_test, Y_pred):.4f}")
# print("\nClassification Report:")
# print(classification_report(Y_test, Y_pred))
