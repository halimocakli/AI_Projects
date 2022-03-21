import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataFrame = pd.read_csv("polynomial.csv", sep=";")
print(dataFrame)

plt.scatter(dataFrame["deneyim"], dataFrame["maas"])
plt.xlabel("Experience(Year)")
plt.ylabel("Salary")
plt.savefig("1.png", dpi=300)
plt.show()

experience = dataFrame[["deneyim"]]
salary = dataFrame["maas"]

regression = LinearRegression()
regression.fit(experience, salary)

plt.xlabel("Experience(Year)")
plt.ylabel("Salary")

plt.scatter(dataFrame["deneyim"], dataFrame["maas"])

xAxis = dataFrame["deneyim"]
yAxis = regression.predict(dataFrame[["deneyim"]])

plt.plot(xAxis, yAxis, color="green", label="Linear Regression")
plt.legend()
plt.show()

polynomialRegression = PolynomialFeatures(degree=4)
xPolynomial = polynomialRegression.fit_transform(experience)

regression = LinearRegression()
regression.fit(xPolynomial, salary)

yHead = regression.predict(xPolynomial)

plt.plot(dataFrame["deneyim"], yHead, color="red", label="Polynomial Regression")
plt.legend()

plt.scatter(dataFrame["deneyim"], dataFrame["maas"])
plt.show()

k = [[4.5]]

xPolynomial = polynomialRegression.fit_transform(k)
regression.predict(xPolynomial)
