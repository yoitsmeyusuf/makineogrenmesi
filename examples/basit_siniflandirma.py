"""
Basit Sınıflandırma Örneği - Logistic Regression
Bu örnek, meme kanseri veri seti kullanarak logistic regression ile sınıflandırma yapar.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def main():
    print("🔬 Basit Sınıflandırma Örneği - Logistic Regression")
    print("="*50)
    
    # 1. Veri setini yükle
    print("\n1️⃣ Veri seti yükleniyor...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print(f"   • Örnek sayısı: {X.shape[0]}")
    print(f"   • Özellik sayısı: {X.shape[1]}")
    print(f"   • Sınıf isimleri: {data.target_names}")
    
    # 2. Veriyi eğitim ve test olarak böl
    print("\n2️⃣ Veri train/test olarak bölünüyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   • Eğitim seti: {X_train.shape[0]} örnek")
    print(f"   • Test seti: {X_test.shape[0]} örnek")
    
    # 3. Özellikleri standartlaştır
    print("\n3️⃣ Özellikler standartlaştırılıyor...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   • StandardScaler uygulandı")
    
    # 4. Modeli oluştur ve eğit
    print("\n4️⃣ Logistic Regression modeli eğitiliyor...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    print("   • Model eğitimi tamamlandı")
    
    # 5. Tahminleri yap
    print("\n5️⃣ Test seti üzerinde tahminler yapılıyor...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # 6. Sonuçları değerlendir
    print("\n6️⃣ Sonuçlar:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   • Doğruluk (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detaylı rapor
    print("\n📊 Detaylı Classification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # 7. Örnek tahminler göster
    print("\n🔍 Örnek tahminler (ilk 5 test örneği):")
    print("-" * 60)
    print(f"{'Gerçek':>10} {'Tahmin':>10} {'Güven':>10} {'Sonuç':>15}")
    print("-" * 60)
    
    for i in range(min(5, len(y_test))):
        gercek = data.target_names[y_test[i]]
        tahmin = data.target_names[y_pred[i]]
        guven = np.max(y_pred_proba[i])
        sonuc = "✓ Doğru" if y_test[i] == y_pred[i] else "✗ Yanlış"
        
        print(f"{gercek:>10} {tahmin:>10} {guven:>9.3f} {sonuc:>15}")
    
    # 8. Model katsayıları
    print(f"\n🧮 Model parametreleri:")
    print(f"   • En önemli 3 özellik:")
    feature_importance = np.abs(model.coef_[0])
    top_features = np.argsort(feature_importance)[-3:][::-1]
    
    for i, feat_idx in enumerate(top_features):
        print(f"     {i+1}. {data.feature_names[feat_idx]}: {model.coef_[0][feat_idx]:.3f}")
    
    print("\n✅ Örnek tamamlandı!")
    print("\n💡 Bu örneği geliştirmek için:")
    print("   • Farklı algoritmalar deneyin (Random Forest, SVM, etc.)")
    print("   • Hiperparametre optimizasyonu yapın")
    print("   • Cross-validation kullanın")
    print("   • Özellik seçimi uygulayın")

if __name__ == "__main__":
    main()