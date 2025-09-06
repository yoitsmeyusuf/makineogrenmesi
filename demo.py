#!/usr/bin/env python3
"""
Makine Öğrenmesi Algoritmaları - Demo Scripti
Bu script projenin ana özelliklerini gösterir ve tüm algoritmaları hızlıca test eder.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Supervised Learning Algoritmaları
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Unsupervised Learning Algoritmaları
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Boosting Algoritmaları
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

def print_header(title):
    """Başlık yazdırma fonksiyonu"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_results(algorithm_name, accuracy, f1):
    """Sonuç yazdırma fonksiyonu"""
    print(f"{algorithm_name:20} | Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}")

def test_supervised_learning():
    """Denetimli öğrenme algoritmalarını test et"""
    print_header("DENETIMLİ ÖĞRENME ALGORİTMALARI TEST EDİLİYOR")
    
    # Veri setini yükle
    print("\n📊 Breast Cancer veri seti yükleniyor...")
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target
    print(f"   - Örnek sayısı: {X.shape[0]}")
    print(f"   - Özellik sayısı: {X.shape[1]}")
    print(f"   - Sınıf sayısı: {len(np.unique(y))}")
    
    # Veriyi böl ve standartlaştır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n🔧 Veri hazırlandı: {X_train.shape[0]} eğitim, {X_test.shape[0]} test örneği")
    
    # Algoritmaları tanımla
    algorithms = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=50),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Classifier": SVC(random_state=42),
        "Neural Network (MLP)": MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(50,))
    }
    
    # Boosting algoritmalarını ekle
    if XGBOOST_AVAILABLE:
        algorithms["XGBoost"] = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    
    if LIGHTGBM_AVAILABLE:
        algorithms["LightGBM"] = LGBMClassifier(random_state=42, verbosity=-1)
    
    if CATBOOST_AVAILABLE:
        algorithms["CatBoost"] = CatBoostClassifier(random_state=42, verbose=False)
    
    print(f"\n🤖 {len(algorithms)} algoritma test ediliyor...")
    print("-" * 60)
    
    results = []
    for name, algorithm in algorithms.items():
        try:
            # Modeli eğit
            algorithm.fit(X_train_scaled, y_train)
            
            # Tahmin yap
            predictions = algorithm.predict(X_test_scaled)
            
            # Metrikleri hesapla
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            
            print_results(name, accuracy, f1)
            results.append({"Algorithm": name, "Accuracy": accuracy, "F1-Score": f1})
            
        except Exception as e:
            print(f"{name:20} | HATA: {str(e)[:40]}...")
    
    # En iyi algoritmayı bul
    if results:
        best_result = max(results, key=lambda x: x["Accuracy"])
        print(f"\n🏆 En iyi algoritma: {best_result['Algorithm']} "
              f"(Accuracy: {best_result['Accuracy']:.4f})")
    
    return results

def test_unsupervised_learning():
    """Denetimsiz öğrenme algoritmalarını test et"""
    print_header("DENETIMSİZ ÖĞRENME ALGORİTMALARI TEST EDİLİYOR")
    
    # Iris veri setini yükle
    print("\n🌸 Iris veri seti yükleniyor...")
    dataset = load_iris()
    X, y = dataset.data, dataset.target
    print(f"   - Örnek sayısı: {X.shape[0]}")
    print(f"   - Özellik sayısı: {X.shape[1]}")
    print(f"   - Gerçek sınıf sayısı: {len(np.unique(y))}")
    
    # Veriyi standartlaştır
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n🔧 Veri standartlaştırıldı")
    print("-" * 60)
    
    # K-Means Kümeleme
    print("🔍 K-Means Kümeleme:")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Kümeleme kalitesini değerlendir (gerçek etiketlerle karşılaştır)
    from sklearn.metrics import adjusted_rand_score
    ari_kmeans = adjusted_rand_score(y, kmeans_labels)
    print(f"   - Adjusted Rand Index: {ari_kmeans:.4f}")
    print(f"   - Küme merkezleri: {kmeans.n_clusters} adet")
    
    # DBSCAN Kümeleme
    print("\n🔍 DBSCAN Kümeleme:")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    print(f"   - Bulunan küme sayısı: {n_clusters_dbscan}")
    print(f"   - Gürültü noktası sayısı: {n_noise}")
    
    if n_clusters_dbscan > 0:
        ari_dbscan = adjusted_rand_score(y, dbscan_labels)
        print(f"   - Adjusted Rand Index: {ari_dbscan:.4f}")
    
    # PCA Boyut İndirgeme
    print("\n📐 PCA Boyut İndirgeme:")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    print(f"   - 2 boyuta indirgendi")
    print(f"   - Açıklanan varyans: {explained_variance[0]:.3f}, {explained_variance[1]:.3f}")
    print(f"   - Toplam açıklanan varyans: {sum(explained_variance):.3f} ({sum(explained_variance)*100:.1f}%)")
    
    # t-SNE Boyut İndirgeme
    print("\n📐 t-SNE Boyut İndirgeme:")
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        print(f"   - 2 boyuta indirgendi")
        print(f"   - KL divergence: {tsne.kl_divergence_:.3f}")
    except Exception as e:
        print(f"   - HATA: {str(e)[:50]}...")

def display_project_info():
    """Proje bilgilerini göster"""
    print_header("MAKİNE ÖĞRENMESİ ALGORİTMALARI PROJESİ")
    
    print("\n🎯 Bu proje şunları yapabilir:")
    print("   ✓ Denetimli Öğrenme: Sınıflandırma ve Regresyon")
    print("   ✓ Denetimsiz Öğrenme: Kümeleme ve Boyut İndirgeme")
    print("   ✓ Model Performans Karşılaştırması")
    print("   ✓ Veri Ön İşleme ve Görselleştirme")
    
    print("\n📚 Mevcut Algoritmalar:")
    print("   • Klasik: Logistic Regression, Decision Tree, KNN, SVM")
    print("   • Ensemble: Random Forest")
    print("   • Boosting: XGBoost, LightGBM, CatBoost")
    print("   • Neural Network: Multi-Layer Perceptron")
    print("   • Kümeleme: K-Means, DBSCAN")
    print("   • Boyut İndirgeme: PCA, t-SNE")
    
    print("\n📊 Veri Setleri:")
    print("   • Breast Cancer: İkili sınıflandırma (569 örnek, 30 özellik)")
    print("   • Iris: Çok sınıflı sınıflandırma (150 örnek, 4 özellik)")
    
    print("\n📖 Jupyter Notebook'lar:")
    print("   • algorithms/SupervisedLearning.ipynb")
    print("   • algorithms/UnsupervisedLearning.ipynb")
    
    print(f"\n🔧 Kütüphane Durumu:")
    print(f"   • XGBoost: {'✓ Mevcut' if XGBOOST_AVAILABLE else '✗ Yüklenmemiş'}")
    print(f"   • LightGBM: {'✓ Mevcut' if LIGHTGBM_AVAILABLE else '✗ Yüklenmemiş'}")
    print(f"   • CatBoost: {'✓ Mevcut' if CATBOOST_AVAILABLE else '✗ Yüklenmemiş'}")

def main():
    """Ana fonksiyon"""
    print("🚀 Makine Öğrenmesi Algoritmaları Demo Başlatılıyor...\n")
    
    # Proje bilgilerini göster
    display_project_info()
    
    # Denetimli öğrenme test et
    supervised_results = test_supervised_learning()
    
    # Denetimsiz öğrenme test et  
    test_unsupervised_learning()
    
    # Özet
    print_header("DEMO TAMAMLANDI")
    print("\n✅ Tüm algoritmalar başarıyla test edildi!")
    print("\n📝 Detaylı örnekler için Jupyter Notebook'ları inceleyin:")
    print("   • jupyter notebook")
    print("   • algorithms/SupervisedLearning.ipynb")
    print("   • algorithms/UnsupervisedLearning.ipynb")
    
    print("\n📄 Daha fazla bilgi için OZELLYKLER.md dosyasını okuyun.")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()