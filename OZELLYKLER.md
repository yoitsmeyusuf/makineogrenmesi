# Makine Öğrenmesi Algoritmaları - Proje Özellikleri

Bu belge, projede nelerin yapılabileceğini ve hangi algoritmaların kullanılabileceğini detaylı bir şekilde açıklar.

## 🎯 Proje Ne Yapabilir?

### 1. Denetimli Öğrenme (Supervised Learning)

Bu proje, etiketli verilerle çalışarak tahmin modelleri oluşturmanıza olanak sağlar:

#### 📊 Sınıflandırma Algoritmaları
- **Logistic Regression**: İkili ve çok sınıflı sınıflandırma için temel algoritma
- **Decision Tree**: Karar ağaçları ile kolay yorumlanabilir modeller
- **Random Forest**: Birden çok karar ağacının kombinasyonu, yüksek doğruluk
- **K-Nearest Neighbors (KNN)**: Komşuluk tabanlı sınıflandırma
- **Support Vector Classifier (SVC)**: Destek vektör makineleri
- **XGBoost**: Gradient boosting ile güçlü performans
- **LightGBM**: Hızlı ve etkili gradient boosting
- **CatBoost**: Kategorik veriler için optimize edilmiş boosting
- **Multi-Layer Perceptron (MLP)**: Yapay sinir ağları

#### 📈 Regresyon Algoritmaları
- **Linear Regression**: Doğrusal ilişkileri modellemek için

### 2. Denetimsiz Öğrenme (Unsupervised Learning)

#### 🔍 Kümeleme Algoritmaları
- **K-Means**: Merkez tabanlı kümeleme
- **Hierarchical Clustering**: Hiyerarşik kümeleme
- **DBSCAN**: Yoğunluk tabanlı kümeleme
- **Gaussian Mixture Models (GMM)**: Olasılık tabanlı kümeleme

#### 📐 Boyut İndirgeme
- **Principal Component Analysis (PCA)**: Ana bileşen analizi
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: Görselleştirme için
- **Linear Discriminant Analysis (LDA)**: Sınıf ayırıcı analiz

#### 🛒 Birliktelik Kuralları
- **Apriori Algorithm**: Market sepeti analizi
- **FP-Growth**: Hızlı sık öğe kümesi bulma
- **ECLAT**: Dikey format ile birliktelik kuralları

## 🗂️ Kullanılan Veri Setleri

### Hazır Veri Setleri
1. **Breast Cancer Dataset**: İkili sınıflandırma (569 örnek, 30 özellik)
   - Meme kanseri teşhisi: iyi huylu (benign) vs kötü huylu (malignant)
   - Tümörün çeşitli fiziksel özelliklerini içerir

2. **Iris Dataset**: Çok sınıflı sınıflandırma (150 örnek, 4 özellik)
   - 3 iris çiçeği türü: Setosa, Versicolor, Virginica
   - Çanak yaprağı ve taç yaprağı ölçümleri

## 📈 Performans Ölçümü

### Değerlendirme Metrikleri
- **Accuracy (Doğruluk)**: Doğru tahmin edilen örneklerin oranı
- **F1-Score**: Precision ve Recall'un harmonik ortalaması
- **Model Karşılaştırması**: Tüm algoritmaların performans tabloları

### Veri Ön İşleme
- **StandardScaler**: Özelliklerin standartlaştırılması (ortalama=0, varyans=1)
- **Train-Test Split**: Verinin eğitim ve test setlerine ayrılması

## 🛠️ Teknik Özellikler

### Kullanılan Kütüphaneler
```python
- scikit-learn: Temel makine öğrenmesi algoritmaları
- pandas: Veri manipülasyonu
- numpy: Sayısal hesaplamalar  
- matplotlib: Veri görselleştirme
- xgboost: Gelişmiş gradient boosting
- lightgbm: Microsoft'un hızlı boosting algoritması
- catboost: Yandex'in kategorik veri odaklı algoritması
- jupyter: İnteraktif notebook ortamı
```

## 🎓 Eğitim Materyali

### Jupyter Notebook'lar
1. **SupervisedLearning.ipynb**: 
   - Kapsamlı denetimli öğrenme örnek uygulamaları
   - Her algoritma için detaylı kod ve açıklama
   - Performans karşılaştırmaları ve sonuç analizi

2. **UnsupervisedLearning.ipynb**:
   - Denetimsiz öğrenme algoritmaları
   - Kümeleme, boyut indirgeme ve birliktelik kuralları
   - Teorik açıklamalar ve pratik örnekler

## 🚀 Nasıl Kullanılır?

### 1. Kurulum
```bash
pip install -r requirements.txt
```

### 2. Jupyter Notebook Başlatma
```bash
jupyter notebook
```

### 3. Notebook'ları Çalıştırma
- `algorithms/SupervisedLearning.ipynb` dosyasını açın
- Hücreleri sırayla çalıştırın
- Sonuçları inceleyin ve algoritmaları karşılaştırın

## 📊 Örnek Sonuçlar

### Breast Cancer Dataset Performansı
- **Random Forest**: ~94% doğruluk
- **Logistic Regression**: ~96% doğruluk  
- **XGBoost**: ~95% doğruluk
- **Neural Network (MLP)**: ~94% doğruluk

### Iris Dataset Performansı  
- **Logistic Regression**: ~97% doğruluk
- **Random Forest**: ~96% doğruluk
- **Decision Tree**: ~97% doğruluk
- **SVC**: ~93% doğruluk

## 🎯 Kimler Kullanabilir?

### Hedef Kitle
- **Öğrenciler**: Makine öğrenmesi konseptlerini öğrenmek isteyenler
- **Araştırmacılar**: Hızlı prototipleme ve algoritma karşılaştırması
- **Veri Bilimciler**: Farklı algoritmaları test etmek isteyenler
- **Eğitmenler**: Ders materyali olarak kullanmak isteyenler

### Seviye Gereksinimleri
- **Başlangıç**: Python temel bilgisi yeterli
- **Orta**: Makine öğrenmesi kavramları faydalı
- **İleri**: Algoritma parametrelerini optimize etmek isteyenler

## 🔄 Gelecek Geliştirmeler

### Planlanabilecek Özellikler
- Derin öğrenme algoritmaları (TensorFlow/PyTorch)
- Zaman serisi analizi
- Doğal dil işleme örnekleri
- Computer vision uygulamaları
- Model deployment örnekleri
- Hiperparametre optimizasyonu
- Cross-validation teknikleri

---

**Bu proje, makine öğrenmesi algoritmalarını pratik olarak öğrenmek ve karşılaştırmak için kapsamlı bir kaynak sağlar.**