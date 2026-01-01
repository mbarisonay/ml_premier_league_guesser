import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Ayarlar
plt.style.use('ggplot')
if not os.path.exists('sunum_gorselleri'):
    os.makedirs('sunum_gorselleri')

# Verileri Yükle
print("Veriler yükleniyor...")
try:
    data = joblib.load('super_model.pkl')
    model = data['model']
    perf_profiles = data['performance_profiles']
    fifa_profiles = data['fifa_profiles']
    team_rosters = data.get('team_rosters', {})
except FileNotFoundError:
    print("Hata: 'super_model.pkl' bulunamadı! Önce eğitimi çalıştır.")
    exit()

# ---------------------------------------------------------
# TABLO 1: ÖZNİTELİK ÖNEM DÜZEYLERİ (Feature Importance)
# ---------------------------------------------------------
# Modelin hangi veriye ne kadar değer verdiğini gösterir.
print("\n--- TABLO 1: Modelin Karar Mekanizması ---")

# Model eğitilirken kullandığımız sütun sırası (model_advanced_training.py'den)
feature_names = [
    # Ev Sahibi Performans
    'Home_Shots', 'Home_SOT', 'Home_Corners', 'Home_Possession',
    # Ev Sahibi FIFA
    'Home_FIFA_Overall', 'Home_FIFA_Attack', 'Home_FIFA_Midfield', 'Home_FIFA_Defense', 'Home_FIFA_Physical',
    # Deplasman Performans
    'Away_Shots', 'Away_SOT', 'Away_Corners', 'Away_Possession',
    # Deplasman FIFA
    'Away_FIFA_Overall', 'Away_FIFA_Attack', 'Away_FIFA_Midfield', 'Away_FIFA_Defense', 'Away_FIFA_Physical'
]

importances = model.feature_importances_
feature_df = pd.DataFrame({'Özellik': feature_names, 'Önem (%)': importances * 100})
feature_df = feature_df.sort_values(by='Önem (%)', ascending=False)

print(feature_df.head(10).to_markdown(index=False, floatfmt=".2f"))

# Görselleştirme
plt.figure(figsize=(10, 8))
sns.barplot(x='Önem (%)', y='Özellik', data=feature_df.head(12), palette='viridis')
plt.title('Yapay Zekanın En Çok Dikkat Ettiği 12 Özellik')
plt.xlabel('Önem Düzeyi (%)')
plt.tight_layout()
plt.savefig('sunum_gorselleri/1_oznitelik_onemi.png')
print(">> '1_oznitelik_onemi.png' kaydedildi.")

# ---------------------------------------------------------
# TABLO 2: TAKIM GÜÇ KARŞILAŞTIRMASI (FIFA vs Performans)
# ---------------------------------------------------------
# Hangi takımın kadrosu iyi ama performansı kötü? (Underperforming)
print("\n--- TABLO 2: Potansiyel vs Gerçek Performans ---")

# Ortak takımları bul
common_teams = perf_profiles.index.intersection(fifa_profiles.index)
comparison_df = pd.DataFrame(index=common_teams)

# FIFA Gücü (Overall)
comparison_df['FIFA_Guc'] = fifa_profiles.loc[common_teams]['FIFA_Overall']

# Saha İçi Gücü (İsabetli Şut - SOT)
comparison_df['Saha_Guc'] = perf_profiles.loc[common_teams]['SOT']

# Verileri 0-100 arasına normalize et (Kıyaslamak için)
comparison_df['FIFA_Norm'] = (comparison_df['FIFA_Guc'] - comparison_df['FIFA_Guc'].min()) / (comparison_df['FIFA_Guc'].max() - comparison_df['FIFA_Guc'].min()) * 100
comparison_df['Saha_Norm'] = (comparison_df['Saha_Guc'] - comparison_df['Saha_Guc'].min()) / (comparison_df['Saha_Guc'].max() - comparison_df['Saha_Guc'].min()) * 100

# Farkı Hesapla
comparison_df['Fark'] = comparison_df['Saha_Norm'] - comparison_df['FIFA_Norm']
comparison_df = comparison_df.sort_values('FIFA_Guc', ascending=False)

# İlk 10 takımı göster
display_cols = ['FIFA_Guc', 'Saha_Guc', 'Fark']
print(comparison_df[display_cols].head(10).to_markdown(floatfmt=".2f"))

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=comparison_df, x='FIFA_Guc', y='Saha_Guc', s=100, color='blue')

# Takım isimlerini ekle (Sadece bazıları)
for team in comparison_df.index:
    # Sadece uç noktalardakileri yazalım karışmasın
    if comparison_df.loc[team, 'FIFA_Guc'] > 80 or comparison_df.loc[team, 'Saha_Guc'] > 5:
        plt.text(comparison_df.loc[team, 'FIFA_Guc']+0.2, comparison_df.loc[team, 'Saha_Guc'], team, fontsize=9)

plt.title('Kadro Kalitesi (FIFA) vs. Saha Üretkenliği (SOT)')
plt.xlabel('FIFA Overall Puanı')
plt.ylabel('Maç Başına İsabetli Şut Ortalaması')
plt.grid(True)
plt.savefig('sunum_gorselleri/2_takim_kiyaslama.png')
print(">> '2_takim_kiyaslama.png' kaydedildi.")

# ---------------------------------------------------------
# TABLO 3: VERİ SETİ ÖZETİ
# ---------------------------------------------------------
print("\n--- TABLO 3: Veri Seti İstatistikleri ---")
summary_data = {
    'Kategori': ['Toplam Takım Sayısı', 'Eğitilen Maç Sayısı', 'Oyuncu Verisi Olan Takım', 'Simüle Edilen Sezon'],
    'Değer': [len(all_teams := set(perf_profiles.index) | set(fifa_profiles.index)), 
              model.n_features_in_ if hasattr(model, 'n_features_in_') else "Bilinmiyor", # Tahmini
              len(fifa_profiles),
              "2023-2024"]
}
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_markdown(index=False))

# ---------------------------------------------------------
# TABLO 4: GOLCÜLER VE BİTİRİCİLİK (Top 5)
# ---------------------------------------------------------
print("\n--- TABLO 4: Ligin En Keskin Golcüleri (Veritabanından) ---")
all_players = []
for team, roster in team_rosters.items():
    for p in roster:
        p['Team'] = team
        all_players.append(p)

players_df = pd.DataFrame(all_players)
top_scorers = players_df.sort_values(by='Finishing', ascending=False).head(10)
print(top_scorers[['Name', 'Team', 'Finishing', 'Position']].to_markdown(index=False))

# Görselleştirme
plt.figure(figsize=(10, 5))
sns.barplot(x='Finishing', y='Name', data=top_scorers, palette='magma')
plt.title('Veritabanındaki En İyi Bitiriciler')
plt.xlabel('Finishing Puanı')
plt.xlim(80, 100) # 80'den başlat fark görünsün
plt.tight_layout()
plt.savefig('sunum_gorselleri/3_en_iyi_golculer.png')
print(">> '3_en_iyi_golculer.png' kaydedildi.")

print("\n✅ Tüm görseller 'sunum_gorselleri' klasörüne kaydedildi.")