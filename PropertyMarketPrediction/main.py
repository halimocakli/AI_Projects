import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

dataFrame = pd.read_csv("multilinearregression.csv", sep=";")

print(dataFrame)
print("----------------------------------------------")
print(dataFrame[["alan", "odasayisi", "binayasi"]])
print("----------------------------------------------")
print(dataFrame["fiyat"])

x = dataFrame[['alan', 'odasayisi', 'binayasi']]
y = dataFrame['fiyat']

regression = linear_model.LinearRegression()
regression.fit(x.values, y.values)

reg1 = regression.predict([[230, 4, 10]])
reg2 = regression.predict([[230, 6, 0]])
reg3 = regression.predict([[355, 3, 20]])
reg4 = regression.predict([[100, 2, 10]])
reg5 = regression.predict([[230, 4, 10], [230, 6, 0], [355, 3, 20], [100, 2, 10]])

print("\n\n")

print(reg1)
print(reg2)
print(reg3)
print(reg4)
print(reg5)

print("\n")

print(regression.coef_)
print(regression.intercept_)

print("\n")

a = regression.intercept_
b1 = regression.coef_[0]
b2 = regression.coef_[1]
b3 = regression.coef_[2]

x1 = 230
x2 = 4
x3 = 10

y = a + b1*x1 + b2*x2 + b3*x3
print(y)

