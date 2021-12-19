import numpy
import pandas as pd
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"

dataFrame = pd.read_csv("Avm_Musterileri.csv")
print(dataFrame.head())

dataFrame.rename(columns={"Annual Income (k$)": "income"}, inplace=True)
dataFrame.rename(columns={"Spending Score (1-100)": "score"}, inplace=True)

plt.scatter(dataFrame["income"], dataFrame["score"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

scaler = MinMaxScaler()
scaler.fit(dataFrame[["income"]])
dataFrame["income"] = scaler.transform(dataFrame[["income"]])

scaler.fit(dataFrame[["score"]])
dataFrame["score"] = scaler.transform(dataFrame[["score"]])

print(dataFrame.head())
print(dataFrame.tail())

kRange = range(1, 11)
listDist = []

for k in kRange:
    myKmeansModel = KMeans(n_clusters=k)
    myKmeansModel.fit(dataFrame[["income", "score"]])
    listDist.append(myKmeansModel.inertia_)

plt.xlabel("K")
plt.ylabel("Distortion Value (inertia)")
plt.plot(kRange, listDist)
plt.show()

myKmeansModel = KMeans(n_clusters=5)
yPredicted = myKmeansModel.fit_predict(dataFrame[["income", "score"]])
print(yPredicted)

dataFrame["cluster"] = yPredicted
print(dataFrame.head())

print(myKmeansModel.cluster_centers_)

dataFrame1 = dataFrame[dataFrame.cluster == 0]
dataFrame2 = dataFrame[dataFrame.cluster == 1]
dataFrame3 = dataFrame[dataFrame.cluster == 2]
dataFrame4 = dataFrame[dataFrame.cluster == 3]
dataFrame5 = dataFrame[dataFrame.cluster == 4]

plt.xlabel('Income')
plt.ylabel('Score')

plt.scatter(dataFrame1['income'], dataFrame1['score'], color='green')
plt.scatter(dataFrame2['income'], dataFrame2['score'], color='red')
plt.scatter(dataFrame3['income'], dataFrame3['score'], color='black')
plt.scatter(dataFrame4['income'], dataFrame4['score'], color='orange')
plt.scatter(dataFrame5['income'], dataFrame5['score'], color='purple')

plt.scatter(myKmeansModel.cluster_centers_[:, 0], myKmeansModel.cluster_centers_[:, 1], color="blue", marker="X",
            label="centroid")
plt.legend()
plt.show()

