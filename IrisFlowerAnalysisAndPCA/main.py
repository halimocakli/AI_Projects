import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataURL = "pca_iris.data"
dataFrame = pd.read_csv(dataURL, names=["sepal length", "sepal width", "petal length", "petal width", "target"])
print(dataFrame)

features = ['sepal length', 'sepal width', 'petal length', 'petal width']

xAxis = dataFrame[features]
yAxis = dataFrame[["target"]]

xAxis = StandardScaler().fit_transform(xAxis)
print(xAxis)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(xAxis)
principalDataFrame = pd.DataFrame(data=principalComponents, columns=["principal component 1", "principal component 2"])

print(principalDataFrame)

finalDataFrame = pd.concat([principalDataFrame, dataFrame[["target"]]], axis=1)
finalDataFrame.head()

dataFrameSetosa = finalDataFrame[dataFrame.target == "Iris-setosa"]
dataFrameVirginica = finalDataFrame[dataFrame.target == "Iris-virginica"]
dataFrameVersicolor = finalDataFrame[dataFrame.target == "Iris-versicolor"]

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.scatter(dataFrameSetosa["principal component 1"], dataFrameSetosa["principal component 2"], color="green")
plt.scatter(dataFrameVirginica["principal component 1"], dataFrameVirginica["principal component 2"], color="red")
plt.scatter(dataFrameVersicolor["principal component 1"], dataFrameVersicolor["principal component 2"], color="blue")

targets = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
colors = ["g", "b", "r"]

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

for target, col in zip(targets, colors):
    dataFrameTemp = finalDataFrame[dataFrame.target==target]
    plt.scatter(dataFrameTemp["principal component 1"], dataFrameTemp["principal component 2"], color=col)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

plt.show()