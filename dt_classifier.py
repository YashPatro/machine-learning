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