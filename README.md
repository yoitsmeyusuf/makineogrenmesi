# Makine Öğrenmesi Algoritmaları

Bu proje, makine öğrenmesi algoritmalarının kapsamlı bir şekilde Python ile uygulandığı, öğrenme amaçlı bir kaynaktır. **9 farklı denetimli öğrenme** ve **6+ denetimsiz öğrenme** algoritması ile birlikte pratik örnekler sunar.

## 🚀 Hızlı Başlangıç

### Demo Script ile Hızlı Test
```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Tüm algoritmaları hızlıca test et
python demo.py
```

### Jupyter Notebook ile Detaylı İnceleme
```bash
jupyter notebook
# Daha sonra algorithms/ klasöründeki notebook'ları açın
```

## 📚 Proje İçeriği

### 🎯 Denetimli Öğrenme Algoritmaları
| Algoritma | Tür | Performans (Breast Cancer) |
|-----------|-----|----------------------------|
| **Logistic Regression** | Sınıflandırma | ~98.3% |
| **Decision Tree** | Sınıflandırma | ~94.2% |
| **Random Forest** | Sınıflandırma | ~97.1% |
| **K-Nearest Neighbors** | Sınıflandırma | ~95.9% |
| **Support Vector Classifier** | Sınıflandırma | ~97.7% |
| **XGBoost** | Sınıflandırma | ~96.5% |
| **LightGBM** | Sınıflandırma | ~95.9% |
| **CatBoost** | Sınıflandırma | ~97.7% |
| **Multi-Layer Perceptron** | Sınıflandırma | ~97.7% |

### 🔍 Denetimsiz Öğrenme Algoritmaları
- **Kümeleme**: K-Means, Hierarchical Clustering, DBSCAN, Gaussian Mixture Models
- **Boyut İndirgeme**: PCA, t-SNE, Linear Discriminant Analysis
- **Birliktelik Kuralları**: Apriori, FP-Growth, ECLAT

## 📊 Kullanılan Veri Setleri

### 1. Breast Cancer Dataset
- **Amaç**: İkili sınıflandırma (meme kanseri teşhisi)
- **Örnekler**: 569 hasta
- **Özellikler**: 30 nümerik özellik
- **Sınıflar**: İyi huylu (benign) vs Kötü huylu (malignant)

### 2. Iris Dataset  
- **Amaç**: Çok sınıflı sınıflandırma (çiçek türü tanıma)
- **Örnekler**: 150 çiçek
- **Özellikler**: 4 fiziksel ölçüm
- **Sınıflar**: Setosa, Versicolor, Virginica

## 🗂️ Dosya Yapısı

```
makineogrenmesi/
├── README.md                           # Bu dosya
├── OZELLYKLER.md                       # Detaylı özellik listesi
├── demo.py                             # Hızlı demo scripti
├── requirements.txt                    # Bağımlılık listesi
└── algorithms/
    ├── SupervisedLearning.ipynb        # Denetimli öğrenme örnekleri
    └── UnsupervisedLearning.ipynb      # Denetimsiz öğrenme örnekleri
```

## 🛠️ Kurulum

### Gereksinimler
- Python 3.8+
- pip veya conda

### Adım 1: Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### Adım 2: Doğrula
```bash
python demo.py
```
Bu komut tüm algoritmaları test eder ve sonuçları gösterir.

## 📖 Kullanım Örnekleri

### 1. Hızlı Test (Demo Script)
```bash
python demo.py
```
- Tüm algoritmaları otomatik test eder
- Performans sonuçlarını gösterir
- ~2 dakikada tamamlanır

### 2. Detaylı İnceleme (Jupyter Notebook)
```bash
jupyter notebook
```
- `algorithms/SupervisedLearning.ipynb`: Adım adım açıklamalarla denetimli öğrenme
- `algorithms/UnsupervisedLearning.ipynb`: Kümeleme ve boyut indirgeme örnekleri

### 3. Özelleştirmiş Kullanım
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Veri yükle ve hazırla
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

# Veriyi standartlaştır
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Tahmin yap
predictions = model.predict(X_test_scaled)
```

## 🎯 Hedef Kitle

- **Öğrenciler**: Makine öğrenmesi kavramlarını pratik olarak öğrenmek isteyenler
- **Araştırmacılar**: Hızlı prototipleme ve algoritma karşılaştırması yapmak isteyenler
- **Eğitmenler**: Ders materyali arayan öğretim görevlileri
- **Geliştiriciler**: ML algoritmaları hakkında tecrübe kazanmak isteyenler

## 📈 Öne Çıkan Özellikler

- ✅ **9 Denetimli Öğrenme** algoritması (Klasik + Modern)
- ✅ **6+ Denetimsiz Öğrenme** algoritması 
- ✅ **Performans karşılaştırmaları** ve detaylı açıklamalar
- ✅ **Türkçe dokümantasyon** ve yorumlar
- ✅ **Hızlı demo scripti** ile anında test
- ✅ **Jupyter Notebook** desteği ile interaktif öğrenme
- ✅ **Gerçek veri setleri** ile pratik örnekler

## 🤝 Katkıda Bulunma

Bu proje açık kaynak kodludur. Katkılarınızı bekliyoruz:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-algoritma`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni algoritma eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-algoritma`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje eğitim amaçlı olarak hazırlanmıştır ve özgürce kullanılabilir.

---

**💡 İpucu**: Daha detaylı bilgi için `OZELLYKLER.md` dosyasını inceleyebilir, hızlı test için `python demo.py` komutunu çalıştırabilirsiniz.
