# Örnek Kullanımlar

Bu klasörde, makine öğrenmesi algoritmalarının basit ve anlaşılır örnekleri bulunmaktadır.

## 📁 Mevcut Örnekler

### 1. `basit_siniflandirma.py` - Denetimli Öğrenme Örneği
- **Algoritma**: Logistic Regression
- **Veri Seti**: Breast Cancer Dataset
- **Amaç**: İkili sınıflandırma (kanser teşhisi)
- **Özellikler**:
  - Veri ön işleme (StandardScaler)
  - Model eğitimi ve değerlendirme
  - Örnek tahminler
  - En önemli özelliklerin gösterimi

**Çalıştırma:**
```bash
cd examples
python basit_siniflandirma.py
```

### 2. `basit_kumeleme.py` - Denetimsiz Öğrenme Örneği
- **Algoritma**: K-Means Clustering
- **Veri Seti**: Iris Dataset
- **Amaç**: Veriyi 3 kümeye ayırma
- **Özellikler**:
  - Kümeleme kalitesi analizi (Silhouette Score)
  - Gerçek sınıflarla karşılaştırma (ARI)
  - Optimum küme sayısı analizi (Elbow Method)
  - Küme merkezleri ve dağılım analizi

**Çalıştırma:**
```bash
cd examples
python basit_kumeleme.py
```

## 🎯 Bu Örneklerin Amacı

Bu örnekler şu amaçlarla hazırlanmıştır:

1. **Başlangıç Seviyesi**: ML algoritmalarına yeni başlayanlar için
2. **Adım Adım Açıklama**: Her işlem açıklanmış durumda
3. **Pratik Uygulama**: Gerçek veri setleri ile çalışma
4. **Sonuç Analizi**: Model performansını anlama
5. **Geliştirme Önerileri**: İleri seviye için yol haritası

## 🚀 Nasıl Kullanılır?

### Tüm örnekleri çalıştırma:
```bash
# Ana dizinden
cd examples
python basit_siniflandirma.py
python basit_kumeleme.py
```

### Kodu inceleme:
Her dosya iyi dokümante edilmiş ve açıklayıcı yorumlar içerir. Kodu okuyarak ML sürecini adım adım öğrenebilirsiniz.

### Özelleştirme:
- Farklı algoritmalar deneyebilirsiniz
- Parametreleri değiştirebilirsiniz
- Kendi veri setinizi kullanabilirsiniz

## 🔄 Sonraki Adımlar

Bu temel örnekleri çalıştırdıktan sonra:

1. **Jupyter Notebook'ları** inceleyın (`algorithms/` klasörü)
2. **Demo scripti** çalıştırın (`python demo.py`)
3. **Farklı veri setleri** ile deneyin
4. **Hiperparametre optimizasyonu** öğrenin
5. **Model karşılaştırması** yapın

## 💡 İpuçları

- Her örnek ~1-2 dakikada çalışır
- Çıktıları dikkatle okuyun
- Farklı random_state değerleri deneyin
- Sonuçları yorumlamaya çalışın