# import numpy
# import matplotlib as plt
# from sklearn.linear_model import LinearRegression

# #manual calcutations 

# x = [1,2,3,5,7]
# y= [1,3,5,2,9]


# def mean_finder(list1):
#     return sum(list1)/len(list1)

# mean_x = mean_finder(x)
# mean_y = mean_finder(y)

# numerator = 0
# denomenator = 0

# for i in range(len(x)):
#     numerator+= (x[i]-mean_x)*(y[i]-mean_y)
#     denomenator += pow((x[i]-mean_x),2)        

# m = numerator/denomenator
# print(m)
# c = round(mean_y-m*mean_x,1)
# print(c)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#hw 
#Load data from a CSV file 
df = pd.read_csv('data.csv')

#2D array
# X = df[['x']] 
# y = df['y']


X = [1,2,3,5,7]
y= [1,3,5,2,9]


# Linear Regression model
model = LinearRegression()
model.fit(X, y)


# coeffi/gradient
m = model.coef_[0]
c = model.intercept_

print(f"Calculated Slope (m): {m}")
print(f"Calculated Intercept (c): {c}")



#plot line
plt.scatter(X, y, color='blue', label='Actual Data from CSV')
plt.plot(X, model.predict(X), color='red', label='Best-Fit Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression from CSV Data')

plt.legend()
plt.show()