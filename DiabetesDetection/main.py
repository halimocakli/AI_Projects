import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("diabetes.csv")
dataset.head()

# Outcome = 1 : Diabetic Patient
# Outcome = 0 : Healthy Person
diabeticPatients = dataset[dataset.Outcome == 1]
healthyPeople = dataset[dataset.Outcome == 0]

# Şimdilik sadece gloucose'a bakarak örnek bir çizim yapalım:
# Programımızın sonunda makine öğrenme modelimiz sadece glikoza değil tüm diğer verilere bakarak bir tahmin yapacaktır..
plt.scatter(healthyPeople.Age, healthyPeople.Glucose, color="green", label="sağlıklı", alpha=0.4)
plt.scatter(diabeticPatients.Age, diabeticPatients.Glucose, color="red", label="diabet hastası", alpha=0.4)

plt.xlabel("Age")
plt.ylabel("Glucose")

plt.legend()
plt.show()

# x ve y eksenlerini belirleyelim
yAxis = dataset.Outcome.values
xRowData = dataset.drop(["Outcome"], axis=1)
# Outcome sütununu(dependent variable) çıkarıp sadece independent variables bırakıyoruz
# Çüknü KNN algoritması x değerleri içerisinde gruplandırma yapacak..

# normalization yapıyoruz - x_ham_veri içerisindeki değerleri sadece 0 ve 1 arasında olacak şekilde hepsini güncelliyoruz
# Eğer bu şekilde normalization yapmazsak yüksek rakamlar küçük rakamları ezer ve KNN algoritmasını yanıltabilir!
xAxis = (xRowData - np.min(xRowData)) / (np.max(xRowData) - np.min(xRowData))

# önce
print("Row datas before normalization")
print("------------------------------")
print(xRowData.head())

# sonra
print("\nDatas that we are going to educate AI with after normalization")
print("----------------------------------------------------------------")
print(xAxis.head())

# train datamız ile test datamızı ayırıyoruz
# train datamız sistemin sağlıklı insan ile hasta insanı ayırt etmesini öğrenmek için kullanılacak
# test datamız ise bakalım makine öğrenme modelimiz doğru bir şekilde hasta ve sağlıklı insanları ayırt edebiliyor mu diye
# test etmek için kullanılacak...
xTrain, xTest, yTrain, yTest = train_test_split(xAxis, yAxis, test_size=0.1, random_state=1)

# knn modelimizi oluşturuyoruz.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xTrain, yTrain)
prediction = knn.predict(xTest)

print("\nConfirmation test results of our test data for K = 3:", knn.score(xTest, yTest))
print("-----------------------------------------------------------------------")

counter = 1

for k in range(1, 11):
    newKnn = KNeighborsClassifier(n_neighbors=k)
    newKnn.fit(xTrain, yTrain)
    print(counter, "\t", "Accuracy rate: %", newKnn.score(xTest, yTest) * 100)
    counter = counter + 1
