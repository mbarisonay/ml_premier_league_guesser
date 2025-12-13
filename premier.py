import pandas as pd

# 1. Veriyi Yükle
df = pd.read_csv('mac_sonuclari.csv')

# 2. Sütun İsimlerini Temizle (Olası boşlukları silmek için)
df.columns = df.columns.str.strip()

# 3. Son 5 Yılda Premier Lig'de Oynamış Takımların Listesi
# ÖNEMLİ: Veri setinizdeki yazım şekilleriyle (örn: "Man City" vs "Manchester City") 
# bu listenin birebir tutması gerekir. Yaygın varyasyonları ekledim.
pl_teams_whitelist = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Leeds", "Leicester", "Liverpool", "Luton", "Man City", "Manchester City",
    "Man United", "Man Utd", "Manchester United", "Newcastle", "Norwich",
    "Nott'm Forest", "Nottingham Forest", "Sheffield United", "Sheffield Utd",
    "Southampton", "Tottenham", "Watford", "West Brom", "West Ham",
    "Wolves", "Wolverhampton", "Wolverhampton Wanderers"
]

# 4. Filtreleme İşlemi
# Hem Ev Sahibi (HomeTeam) hem de Deplasman (AwayTeam) bu listede olmalı.
# Böylece eğer datasetinizde Şampiyonlar Ligi veya FA Cup varsa (örn: Arsenal vs Bayern Munich),
# Bayern listede olmadığı için o maçı dahil etmez. Sadece PL maçları kalır.
df_pl = df[
    df['HomeTeam'].isin(pl_teams_whitelist) & 
    df['AwayTeam'].isin(pl_teams_whitelist)
].copy()

# 5. Kontrol Et
print(f"Orijinal Veri Sayısı: {len(df)}")
print(f"Premier Lig Veri Sayısı: {len(df_pl)}")

print("\nFiltrelenen Takımlar:")
print(df_pl['HomeTeam'].unique())

#6. Temizlenmiş Veriyi Kaydet (İsterseniz)
df_pl.to_csv('sadece_premier_lig.csv', index=False)

# Şimdi önceki ML kodunda 'df' yerine 'df_pl' kullanabilirsiniz.