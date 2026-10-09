'''# coloumns: Field
Meaning
crim
Crime rate per person in the area
zn
Proportion of residential land zoned for large houses
indus
Proportion of land used for non-retail business
chas
Whether the area is near the Charles River: 1 = yes, 0 = no
nox
Nitric oxide pollution level
rm
Average number of rooms per house
age
Percentage of houses built before 1940
dis
Distance from major employment centres
rad
Accessibility to major highways
tax
Property tax rate
ptratio
Pupil-to-teacher ratio in schools
b
Historical demographic-derived variable in the original dataset
lstat
Percentage of lower-status population
medv
Median house value — this is usually the target/output'''

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.datasets import load_boston
import os 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score
# bostonData = load_boston()
# bostonData = pd.read_csv('data.csv')

print(os.path.abspath('data.csv'))

bostonData = pd.read_csv('bostonHousing.csv')
print(bostonData.head())

X = bostonData[['lstat']]

y = bostonData['medv']
print(X,y)