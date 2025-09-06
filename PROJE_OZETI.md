# Proje Özeti: "Neler Yapabildiğini" Anlama Rehberi

Bu belge, **makineogrenmesi** projesinin tam olarak neleri yapabildiğini açıklar.

## 🎯 Projenin Ana Yetenekleri

### 1. **Denetimli Öğrenme (Supervised Learning)**
Projede **9 farklı algoritma** bulunuyor:

| Algoritma | Tür | Breast Cancer Dataset Performansı |
|-----------|-----|-----------------------------------|
| Logistic Regression | Sınıflandırma | **%98.3** (En iyi) |
| Random Forest | Sınıflandırma | %97.1% |
| Support Vector Classifier | Sınıflandırma | %97.7% |
| Neural Network (MLP) | Sınıflandırma | %97.7% |
| CatBoost | Sınıflandırma | %97.7% |
| XGBoost | Sınıflandırma | %96.5% |
| K-Nearest Neighbors | Sınıflandırma | %95.9% |
| LightGBM | Sınıflandırma | %95.9% |
| Decision Tree | Sınıflandırma | %94.2% |

### 2. **Denetimsiz Öğrenme (Unsupervised Learning)**
- **Kümeleme**: K-Means, DBSCAN, Hierarchical, GMM
- **Boyut İndirgeme**: PCA, t-SNE, LDA  
- **Birliktelik Kuralları**: Apriori, FP-Growth, ECLAT

### 3. **Veri Setleri**
- **Breast Cancer**: 569 hasta, 30 özellik (kanser teşhisi)
- **Iris**: 150 çiçek, 4 özellik (tür sınıflandırma)

## 🚀 Nasıl Kullanılır?

### **Hızlı Test (2 dakikada tüm özellikler)**
```bash
python demo.py
```
Bu komut:
- Tüm 15+ algoritmayı test eder
- Performans sonuçlarını gösterir
- Hangi algoritmanın en iyi çalıştığını söyler

### **Detaylı Öğrenme (Jupyter Notebook)**
```bash
jupyter notebook
# Sonra algorithms/ klasöründeki dosyaları açın
```

### **Basit Örnekler (Yeni başlayanlar için)**
```bash
cd examples
python basit_siniflandirma.py    # Sınıflandırma örneği
python basit_kumeleme.py         # Kümeleme örneği
```

## 📊 Test Sonuçları (Gerçek Performans)

### Denetimli Öğrenme Sonuçları:
```
Logistic Regression  | Accuracy: 0.9825 | F1-Score: 0.9860 🏆
Random Forest        | Accuracy: 0.9708 | F1-Score: 0.9772
Support Vector       | Accuracy: 0.9766 | F1-Score: 0.9815
Neural Network       | Accuracy: 0.9766 | F1-Score: 0.9815
CatBoost            | Accuracy: 0.9766 | F1-Score: 0.9817
```

### Denetimsiz Öğrenme Sonuçları:
```
K-Means Kümeleme     | ARI Score: 0.6201 (gerçek sınıflarla %62 uyum)
DBSCAN Kümeleme      | 2 küme buldu, 34 gürültü noktası
PCA Boyut İndirgeme  | %95.8 varyans korundu (4 boyuttan 2'ye)
t-SNE               | KL divergence: 0.173 (iyi görselleştirme)
```

## 📚 Hangi Dosya Ne İşe Yarar?

| Dosya | Açıklama | Kimler Kullanmalı |
|-------|----------|-------------------|
| `demo.py` | Tüm özellikleri hızlı test | Herkese önerilen ilk adım |
| `algorithms/SupervisedLearning.ipynb` | Detaylı denetimli öğrenme | Öğrenciler, öğretmenler |
| `algorithms/UnsupervisedLearning.ipynb` | Kümeleme ve boyut indirgeme | İleri seviye öğrenimi |
| `examples/basit_siniflandirma.py` | Başlangıç sınıflandırma örneği | Yeni başlayanlar |
| `examples/basit_kumeleme.py` | Başlangıç kümeleme örneği | Yeni başlayanlar |
| `OZELLYKLER.md` | Detaylı özellik listesi | Tüm bilgilere ihtiyaç duyanlar |

## 🎓 Kimler Kullanabilir?

### **Öğrenciler**
- Makine öğrenmesi dersi alan öğrenciler
- Bilgisayar mühendisliği öğrencileri  
- Veri bilimi öğrenenler

### **Eğitmenler**
- Makine öğrenmesi dersi veren öğretim görevlileri
- Workshop/eğitim düzenleyenler
- Online kurs hazırlayanlar

### **Profesyoneller**
- Veri bilimciler (hızlı prototipleme için)
- Yazılım geliştiriciler (ML öğrenmek için)
- Araştırmacılar (algoritma karşılaştırması için)

## 💡 Neleri ÖĞRETİR?

1. **Pratik Uygulama**: Teoriden çok pratik yaparak öğrenme
2. **Algoritma Karşılaştırması**: Hangi algoritma ne zaman kullanılır?
3. **Gerçek Veri**: Hazır veri setleriyle gerçek senaryolar
4. **Performans Analizi**: Model başarısını nasıl ölçeriz?
5. **Veri Ön İşleme**: Veriyi ML için nasıl hazırlarız?

## 🚀 Gelecek Özellikler (Eklenebilecekler)

- Derin öğrenme algoritmaları (TensorFlow/PyTorch)
- Zaman serisi analizi
- Doğal dil işleme örnekleri
- Model deployment örnekleri
- Hiperparametre optimizasyonu
- Cross-validation teknikleri

## 📈 Özet: Bu Proje Size Ne Sağlar?

✅ **15+ Algoritma** tek yerde
✅ **Türkçe açıklamalar** ile kolay öğrenme  
✅ **Gerçek performans sonuçları** ile karşılaştırma
✅ **3 farklı kullanım seviyesi** (demo, notebook, örnek)
✅ **Anında test** imkanı (`python demo.py`)
✅ **Başlangıçtan ileri seviyeye** uygun içerik

---

**🎯 Sonuç**: Bu proje, makine öğrenmesi algoritmalarını **pratik olarak öğrenmek ve karşılaştırmak** için kapsamlı bir kaynak. Hem yeni başlayanlar hem de deneyimli kullanıcılar için uygun.