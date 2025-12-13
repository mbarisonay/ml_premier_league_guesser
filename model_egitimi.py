import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------------------------------------------------------
# 1. VERİYİ YÜKLEME VE DÜZENLEME
# ---------------------------------------------------------
# Dosya adını kendine göre değiştirmeyi unutma
df = pd.read_csv('fbref_premier_league_stats_2014-2025_COMPLETE.csv') 

print(f"Veri seti yüklendi. Toplam maç sayısı: {len(df)}")

# Sütun isimlerindeki olası boşlukları temizle
df.columns = df.columns.str.strip()

# Hedef Değişkeni (Maç Sonucu) Oluştur: 
# 2: Ev Sahibi, 1: Beraberlik, 0: Deplasman
def get_result(row):
    if row['FTHG'] > row['FTAG']:
        return 2
    elif row['FTHG'] == row['FTAG']:
        return 1
    else:
        return 0

df['MatchResult'] = df.apply(get_result, axis=1)

# ---------------------------------------------------------
# 2. TAKIM DNA'LARINI (PROFİLLERİNİ) ÇIKARMA
# ---------------------------------------------------------
# Hangi istatistikleri kullanarak takım gücü belirleyeceğiz?
# Veri setindeki sütunlara göre en kritikleri seçtim:
feature_cols_base = [
    'Shots', 'SOT', 'Corners', 'Fouls', 
    'Possession', 'PassesCompleted', 'PassesAttempted', 
    'Touches', 'Tackles', 'Interceptions', 
    'AerialsWon', 'Clearances', 'YellowCards', 'RedCards'
]

# Hem Home hem Away sütunlarını bu isimlerle eşleştirelim
home_cols = ['Home' + col for col in feature_cols_base]
away_cols = ['Away' + col for col in feature_cols_base]

# Ev sahibi performanslarını çek
home_stats = df[['HomeTeam'] + home_cols].copy()
home_stats.columns = ['Team'] + feature_cols_base # İsimleri standartlaştır

# Deplasman performanslarını çek
away_stats = df[['AwayTeam'] + away_cols].copy()
away_stats.columns = ['Team'] + feature_cols_base # İsimleri standartlaştır

# Hepsini alt alta birleştir (Tüm maç istatistikleri havuzu)
all_performances = pd.concat([home_stats, away_stats], ignore_index=True)

# Takımların ortalamasını al (İşte bu TAKIM DNA'sıdır)
team_profiles = all_performances.groupby('Team')[feature_cols_base].mean()

# ---------------------------------------------------------
# 3. BRIGHTON YAMASI (Eksik Takım Uydurma)
# ---------------------------------------------------------
# Eğer Brighton listede yoksa veya verisi çok azsa, lig ortalamasını basıyoruz.
target_missing_team = "Brighton"

if target_missing_team not in team_profiles.index:
    print(f"{target_missing_team} bulunamadı, lig ortalamasıyla oluşturuluyor...")
    # Tüm ligin ortalamasını al
    league_average = team_profiles.mean()
    # Brighton ismiyle yeni bir satır ekle
    team_profiles.loc[target_missing_team] = league_average
else:
    print(f"{target_missing_team} verisi mevcut, yamaya gerek kalmadı.")

print("\n--- Takım Profilleri Örneği (İlk 5 Takım) ---")
print(team_profiles.head())

# ---------------------------------------------------------
# 4. MODEL EĞİTİM VERİSİNİ HAZIRLAMA
# ---------------------------------------------------------
# Geçmiş maçların sonuçlarını, takımların GENEL PROFİLLERİNE göre öğreteceğiz.
# "Güçlü Hücum vs Zayıf Defans gelince ne oluyor?" mantığı.

X = []
y = []

for index, row in df.iterrows():
    home = row['HomeTeam']
    away = row['AwayTeam']
    
    # Eğer takımlar profil listemizde varsa (ki Brighton'ı ekledik, hepsi olmalı)
    if home in team_profiles.index and away in team_profiles.index:
        
        # O takımın genel DNA'sını çek
        h_dna = team_profiles.loc[home].values
        a_dna = team_profiles.loc[away].values
        
        # İki takımın özelliklerini yan yana koy (Input)
        match_features = np.concatenate([h_dna, a_dna])
        
        X.append(match_features)
        y.append(row['MatchResult'])

X = np.array(X)
y = np.array(y)

# ---------------------------------------------------------
# 5. MAKİNE ÖĞRENMESİ (Random Forest)
# ---------------------------------------------------------
print("\nModel eğitiliyor... (Bu işlem veri boyutuna göre biraz sürebilir)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForest: Futbol verisi için en sağlam, overfitting riski düşük modeldir.
model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# Test Sonuçları
y_pred = model.predict(X_test)
print(f"\nModel Doğruluğu: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))

# ---------------------------------------------------------
# 6. KAYIT (Web App İçin Paketleme)
# ---------------------------------------------------------
export_package = {
    'model': model,
    'team_profiles': team_profiles,
    'feature_names': feature_cols_base
}

joblib.dump(export_package, 'premier_league_sim_data.pkl')
print("\nBAŞARILI! 'premier_league_sim_data.pkl' dosyası oluşturuldu.")
print("Şimdi Web App aşamasına geçebiliriz.")