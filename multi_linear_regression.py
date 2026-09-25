import numpy as np
import pandas as pd
import seaborn as sns
# from sklearn.datasets import load_boston


# bostonData = load_boston()
# bostonData = pd.read_csv('data.csv')

bostonData = pd.read_csv('https://github.com/selva86/datasets/blob/master/BostonHousing.csv')
print(bostonData.head())

bostonData['"medv"']

