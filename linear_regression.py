import numpy
import matplotlib as plt
from sklearn.linear_model import LinearRegression

#manual calcutations 

x = [1,2,3,5,7]
y= [1,3,5,2,9]


def mean_finder(list1):
    return sum(list1)/len(list1)

mean_x = mean_finder(x)
mean_y = mean_finder(y)

numerator = 0
denomenator = 0

for i in range(len(x)):
    numerator+= (x[i]-mean_x)*(y[i]-mean_y)
    denomenator += pow((x[i]-mean_x),2)        

m = numerator/denomenator
print(m)
c = round(mean_y-m*mean_x,1)
print(c)
