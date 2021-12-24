"""
    Bu projede görsel sınıflandırma(image processing) işlemleri için Transfer Learning algoritması kullanacağız.
    Üzerinde çalışacağımız verilerin tamamını kendimiz hazırladık ve "test" ile "training" olmak üzere iki farklı kategoriye ayırdık.
    Projemizde VGG-16 adlı pre-trained derin öğrenme modelini kullanmaktayız. VGG-16 modeli sayesinde eğiteceğimiz yeni modelin kısa
    zamanda ortaya çıkabilmesini sağlıyoruz. Bununla birlikte VGG-16 modelinin bir CNN örneği olduğunu da ekleyelim.
    MTARSI(Multi-type Aircraft of Remote Sensing Images)'dan alınan fotoğraflar veri seti olarak kullanıldı. Bu fotoğraflar genel
    olarak uydu görüntülerinden elde edilmektedir.

"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, \
    load_img  # Fotoğraflar üzerinde çalışmamızı sağlar
from tensorflow.keras.models import Sequential  # Yeni bir model yapılandırabilmemizi sağlar
from tensorflow.keras.layers import Dense  # Yeni bir model yapılandırabilmemizi sağlar
from tensorflow.keras.applications.vgg16 import VGG16  # Projede kullanılar Transfer Learning modeli
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from PIL import Image  # Fotoğrafları açmamızı, yeniden botlandırmamızı vs. sağlayan python görüntüleme kütüphanesi

# Öncelikle train ve test görüntülemizin bulunduğu dosyaların dizin adreslerini değişken olarak tanımlıyoruz
trainingFilesPath = "aircraftDataset/Train/"
testFilesPath = "aircraftDataset/Test/"

# Veri setimizde bulunan herhangi bir fotoğrafı img değişkenine atayalım
img = load_img(testFilesPath + "B-52/3-1.jpg")
# Aşağıdaki işlemle biraz önce img değişkenine atadığımız fotoğrafın boyutunu görüntülüyoruz
print(img_to_array(img).shape)
print("\n")

# Fotoğrafı görüntüleyelim ve düzgünce görüp göremediğimizi kontrol edelim
plt.imshow(img)
plt.show()

# Her şey iyi durumda ise test ve training veri setimizi yapılandıralım.
# Fotoğrafları 224x224 olacak şekilde boyutlandırmamızın sebebi VGG-16'nın bu boyuttaki fotoğrafları kabul etmesidir.
trainingData = ImageDataGenerator().flow_from_directory(trainingFilesPath, target_size=(224, 224))
testData = ImageDataGenerator().flow_from_directory(testFilesPath, target_size=(224, 224))

# Veri setimizde 5 tip uçak bulunduğu için değişkenimize 5 atadık. Tip sayısı arttıkça değiştirebiliriz.
numberOfAircraftTypes = 5

"""
Bu aşama itibariyle modelimizi yapılandırmaya başlıyoruz. Daha önce de belirttiğimiz gibi Transfer Learning özelinde VGG16
modelini kullanacağız. Kendi VGG16 modelimizde kendi hazırladığımız test ve trainin verilerimizi modeli eğitmek ve test etmek
için girdi olarak kullanmaktayız. İhtiyaca göre girdilerimizi değiştirebiliriz.
Orijinal VGG-16 modeli ise içerisinde 15.000.000 etiketli ve yüksek kaliteli, 22.000 kategori üzerinden derlenmiş görsel barındıran
veri seti kullanılarak tasarlandı. Bunun sonucunda 1000 kategoriye ayrılmış, kullanılabilir görüntü verilerinden oluşan bir veri seti
oluşturuldu. Bu veri seti VGG-16'nın son katmanı olan output katmanında bulunmaktadır ve içerindeki görseller tamamen uçaklardan oluşmamaktadır. 
bu nedenle biz, savaş uçağı görsellerini kullandığımız yeni bir model üretiyor ve üretim için daha önceden eğitilmiş ve başarı oranı yüksek 
VGG-16 katmanlarını kullanıyoruz.
"""

# Model nesnemizi tanımlayalım
vgg = VGG16()
vggLayers = vgg.layers
print("VGG Layers are Printing...")
print(vggLayers)
print("\n")

"""
Bu aşamada ise yeni bir sequential(ardışık) model üreteceğiz ve ürettiğimiz yeni modele VGG-16 modeli içerisindeki tüm katmanları,
son katman olan output katmanını hariç tutarak, ekleceyeğiz. Son katmanı kendi modelimiz için ayırdık. Böylece input katmanı olacak
şekilde, savaş uçaklarına ait görseller üzerinden eğitilmiş kendi modelimizi VGG-16 modeline son katman olarak ekliyoruz.
"""

vggModelLayerSizeToBeUsed = len(vggLayers) - 1 # "-1" kendi modelimiz için ayırdığımız son output katmanı
model = Sequential()

for i in range(vggModelLayerSizeToBeUsed):
    model.add(vggLayers[i])

# VGG-16 modeli içerisindeki 16 katmanın tamamını yeniden eğitmek istemediğimiz için katmanların "trainable" özelliğini kapatıyoruz.
for layers in model.layers:
    layers.trainable = False

# Orijinal output katmanını atladık ve kendi output katmanımızı yeni modelimize ekliyoruz.
model.add(Dense(numberOfAircraftTypes, activation="softmax"))
print(model.summary())

# Model tasarımımızı tamamladıktan sonra derleme işlemine geçebiliriz
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

batchSize = 4

"""
epochs değerini bilgisayarımızın performansına göre ayarlayabiliriz. Aşağıdaki model eğitimimizin sonunda val_accuracy değerini
%88 olarak görüyoruz. Bu iyi bir sonuç. accuracy ifadesi bir grup eğitilmiş verinin doğruluğu olarak kabul edilirken
val_accuracy ise bir grup test edilmiş verinin doğruluğunu belirtir. Yani bizim test edilmiş verilerimizin doğruluk değeri %88.
"""
model.fit_generator(trainingData, steps_per_epoch=400 // batchSize, epochs=50, validation_data=testData,
                    validation_steps=200 // batchSize)

# Şimdi de eğittimiz modeli kullanarak bir savaş uçağı modelini tespit edip edemediğini kontrol edelim
# Bir savaş uçağı fotoğrafını değişkenimize atıyor ve aynı anda çözünürlüğünü 224x224 değerine ayarlıyoruz
img = Image.open("f22.jpg").resize((224, 224))
# prediction işlemi yapabilmek için img değişkeninde bulunan fotoğrafı array formatına dönüştürüyoruz
img = np.array(img)

# Fotoğrafımızın şekil/çözünürlük ve boyut bilgilerini görüntüleyelim
print(img.shape)
print(img.ndim)

# VGG-16 modelinde tahmin yapabilmek için verimiz 4 boyutlu olmak durumunda. Bu sebeple "-1" ile verimize boyut ekliyoruz
img = img.reshape(-1, 224, 224, 3)

# Fotoğrafımızın şekil/çözünürlük ve boyut bilgilerini tekrar görüntüleyelim
print(img.shape)
print(img.ndim)

# Girdi piksellerini -1 ile 1 arasında scale ediyoruz çünkü VGG-16 modeli bunu gerektiriyor.
img = preprocess_input(img)

# Şimdi prediction aşamasına geçebiliriz
imgForDisplay = load_img("f22.jpg")
plt.imshow(imgForDisplay)
plt.show()

# Şimdi prediction aşamasına geçebiliriz
predictions = model.predict(img)
print(predictions)

"""
Yukarıdaki satırlarda "model.add(Dense(numberOfAircraftTypes, activation="softmax"))" şeklinde bir kod yazmıştık. Bu ifadedeki
"softmax" bir aktivasyon katmanıdır. Buna göre softmax katmanı, önceki katmandan gelen değerleri alarak sınıflandırma işlemi 
içerisinde olasılıksal değer üretimi gerçekleştirir. Sınıflandırma yaparken hangi sınıfa daha yakın olduğuna dair değer üretir.
Ayrıca softmax katmanının bir output katmanı olduğunu da ekleyelim. Örneğin 3 farklı uçağa ait sınıfların 0 1 2 sırasıyla 
etiketlendiğini farz edelim. O halde şifrelenmiş vektörler:

    Class 0: [1, 0, 0]
    Class 1: [0, 1, 0]
    Class 2: [0, 0, 1]

şeklinde olur. Buna "one-hot encoding" denir.

Bizim çalışmamızda:

    A-10  Thunderbolt:  [1,0,0,0,0]
    Boeing B-52:        [0,1,0,0,0]
    Boeing E-3 Sentry:  [0,0,1,0,0]   
    F-22 Raptor:        [0,0,0,1,0]  
    KC-10 Extender:     [0,0,0,0,1]  
    
şeklindedir.

Örneğin sonuç olarak encoded integer class 1'i bekliyorsak hedef vektör:

    [0,1,1] 
    
olmalı.

softmax katmanının çıktısı, predictions değeri en fazla olan veri için 1 olacak şekildedir. Aşağıda imageClass değişkenine
uçaklarımızın isimlerini liste olarak atadık ve herbirinin bir index değeri oldu. Buna göre prediction işlemi için örnek 
veri olarak F-22 Raptor verdiğimizde softmax katmanının çıktısında örnek verimizin indexi ile aynı indexte 1'e en yakın
değer bulunurken diğerlerinde daha küçük değerler bulunur. Buna göre çıktımız: 

    [[4.1648740e-04 1.3187087e-04 1.2833414e-05 9.9896097e-01 4.7782363e-04]]
    
olur.

Buna göre Class 3 olan F-22 Raptop tahmini doğrudur. imageClasses listemizin içerisinde F-22'yi bulmak için ise armax
fonksiyonunu kullanırız.
"""

imageClasses = ["A-10 Thunderbolt", "Boeing B-52", "Boeing E-3 Sentry", "F-22 Raptor", "KC-10 Extender"]

result = np.argmax(predictions[0])
print(imageClasses[result])