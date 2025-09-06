"""
Basit Kümeleme Örneği - K-Means
Bu örnek, Iris veri seti kullanarak K-Means kümeleme yapar.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt

def main():
    print("🌸 Basit Kümeleme Örneği - K-Means")
    print("="*40)
    
    # 1. Veri setini yükle
    print("\n1️⃣ Iris veri seti yükleniyor...")
    data = load_iris()
    X, y = data.data, data.target
    
    print(f"   • Örnek sayısı: {X.shape[0]}")
    print(f"   • Özellik sayısı: {X.shape[1]}")
    print(f"   • Özellik isimleri: {data.feature_names}")
    print(f"   • Gerçek sınıf sayısı: {len(data.target_names)}")
    print(f"   • Sınıf isimleri: {data.target_names}")
    
    # 2. Özellikleri standartlaştır
    print("\n2️⃣ Özellikler standartlaştırılıyor...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("   • StandardScaler uygulandı")
    
    # 3. K-Means modelini oluştur ve eğit
    print("\n3️⃣ K-Means modeli eğitiliyor...")
    n_clusters = 3  # Gerçek sınıf sayısı kadar
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    print(f"   • {n_clusters} küme ile model eğitildi")
    
    # 4. Sonuçları değerlendir
    print("\n4️⃣ Sonuçlar:")
    
    # Silhouette Score
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"   • Silhouette Score: {silhouette_avg:.4f}")
    print("     (1'e yakın olması daha iyi kümeleme anlamına gelir)")
    
    # Adjusted Rand Index (gerçek etiketlerle karşılaştırma)
    ari = adjusted_rand_score(y, cluster_labels)
    print(f"   • Adjusted Rand Index: {ari:.4f}")
    print("     (1'e yakın olması gerçek sınıflarla daha uyumlu demek)")
    
    # 5. Küme merkezleri
    print(f"\n5️⃣ Küme merkezleri:")
    centers = kmeans.cluster_centers_
    for i, center in enumerate(centers):
        print(f"   • Küme {i+1}: {center}")
    
    # 6. Her kümedeki örnek sayısı
    print(f"\n6️⃣ Kümelerdeki örnek dağılımı:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"   • Küme {cluster+1}: {count} örnek")
    
    # 7. Gerçek sınıflarla karşılaştırma
    print(f"\n7️⃣ Gerçek sınıflarla karşılaştırma:")
    confusion_matrix = np.zeros((3, 3))
    
    for true_label in range(3):
        print(f"\n   {data.target_names[true_label]} sınıfı ({np.sum(y == true_label)} örnek):")
        mask = (y == true_label)
        cluster_dist = np.bincount(cluster_labels[mask], minlength=3)
        for cluster in range(3):
            percentage = (cluster_dist[cluster] / np.sum(y == true_label)) * 100
            print(f"     -> Küme {cluster+1}: {cluster_dist[cluster]} örnek ({percentage:.1f}%)")
    
    # 8. Optimum küme sayısı analizi
    print(f"\n8️⃣ Optimum küme sayısı analizi (Elbow Method):")
    inertias = []
    k_range = range(1, 8)
    
    for k in k_range:
        temp_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        temp_kmeans.fit(X_scaled)
        inertias.append(temp_kmeans.inertia_)
    
    print("   • K değeri | Inertia (düşük daha iyi)")
    print("   " + "-" * 35)
    for k, inertia in zip(k_range, inertias):
        print(f"   • K={k:2d}      | {inertia:8.2f}")
    
    print("\n✅ Kümeleme örneği tamamlandı!")
    print("\n💡 Bu örneği geliştirmek için:")
    print("   • Farklı kümeleme algoritmaları deneyin (DBSCAN, Hierarchical)")
    print("   • PCA ile boyut indirgeme yapın")
    print("   • Sonuçları görselleştirin")
    print("   • Farklı mesafe metrikleri kullanın")

if __name__ == "__main__":
    main()