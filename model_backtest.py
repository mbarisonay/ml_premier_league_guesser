import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. VERİYİ YÜKLE
df = pd.read_csv('fbref_premier_league_stats_2014-2025_COMPLETE.csv') # Dosya adını düzelt

# Tarih formatını ayarla
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df.columns = df.columns.str.strip() # Boşluk temizliği

# Hedef Değişken (0: Away, 1: Draw, 2: Home)
def get_result(row):
    if row['FTHG'] > row['FTAG']: return 2
    elif row['FTHG'] == row['FTAG']: return 1
    else: return 0
df['MatchResult'] = df.apply(get_result, axis=1)

# ---------------------------------------------------------
# 2. ZAMANSAL BÖLME (KRİTİK ADIM)
# ---------------------------------------------------------
# 2023-2024 Sezonu yaklaşık Ağustos 2023'te başladı.
split_date = pd.to_datetime("2023-08-01")

# Eğitim Seti: 2023 öncesi maçlar
train_df = df[df['Date'] < split_date].copy()

# Gerçek Test Seti: Sadece 2023-2024 sezonu maçları
real_23_24_df = df[df['Date'] >= split_date].copy()

print(f"Eğitim Verisi (Eski Sezonlar): {len(train_df)} maç")
print(f"Test Verisi (23-24 Sezonu): {len(real_23_24_df)} maç")

# ---------------------------------------------------------
# 3. PROFİL (DNA) ÇIKARMA (Sadece Eski Maçlardan!)
# ---------------------------------------------------------
feature_cols_base = [
    'Shots', 'SOT', 'Corners', 'Fouls', 'Possession', 
    'PassesCompleted', 'PassesAttempted', 'Touches', 
    'Tackles', 'Interceptions', 'AerialsWon', 'Clearances', 
    'YellowCards', 'RedCards'
]
home_cols = ['Home' + col for col in feature_cols_base]
away_cols = ['Away' + col for col in feature_cols_base]

h_stats = train_df[['HomeTeam'] + home_cols].copy()
h_stats.columns = ['Team'] + feature_cols_base
a_stats = train_df[['AwayTeam'] + away_cols].copy()
a_stats.columns = ['Team'] + feature_cols_base

all_perf = pd.concat([h_stats, a_stats], ignore_index=True)
team_profiles = all_perf.groupby('Team')[feature_cols_base].mean()

# EKSİK TAKIM YAMASI (Luton Town gibi yeni çıkanlar için)
# 23-24 sezonundaki takımları kontrol et, profili yoksa lig ortalaması ata
current_season_teams = pd.concat([real_23_24_df['HomeTeam'], real_23_24_df['AwayTeam']]).unique()
league_avg = team_profiles.mean()

for team in current_season_teams:
    if team not in team_profiles.index:
        print(f"UYARI: {team} eski verilerde yok (Yeni Çıktı). Ortalama veri atanıyor.")
        team_profiles.loc[team] = league_avg

# ---------------------------------------------------------
# 4. MODEL EĞİTİMİ (Sadece Eski Maçlarla)
# ---------------------------------------------------------
X = []
y = []

for index, row in train_df.iterrows():
    h, a = row['HomeTeam'], row['AwayTeam']
    if h in team_profiles.index and a in team_profiles.index:
        h_dna = team_profiles.loc[h].values
        a_dna = team_profiles.loc[a].values
        X.append(np.concatenate([h_dna, a_dna]))
        y.append(row['MatchResult'])

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X, y)
print("Model eski sezon verileriyle eğitildi.")

# ---------------------------------------------------------
# 5. KAYIT
# ---------------------------------------------------------
# Gerçek 23-24 fikstürünü de kaydediyoruz ki karşılaştırma yapabilelim
export_data = {
    'model': model,
    'team_profiles': team_profiles,
    'real_23_24_data': real_23_24_df, # Gerçek sonuçları buraya sakladık
    'feature_names': feature_cols_base
}

joblib.dump(export_data, 'comparison_data.pkl')
print("Veriler 'comparison_data.pkl' dosyasına kaydedildi.")