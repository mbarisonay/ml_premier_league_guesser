import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------------------------------------------------
# DOSYA İSİMLERİ
# ---------------------------------------------------------
match_data_file = 'fbref_premier_league_stats_2014-2025_COMPLETE.csv'
player_data_file = 'ALL_FIFA_STATS_FINAL.csv'

print("Veriler yükleniyor...")
matches = pd.read_csv(match_data_file)
# Tüm string sütunlarını temizle
df_obj = matches.select_dtypes(['object'])
matches[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

players = pd.read_csv(player_data_file)
df_obj_p = players.select_dtypes(['object'])
players[df_obj_p.columns] = df_obj_p.apply(lambda x: x.str.strip())

# ---------------------------------------------------------
# 1. TEMİZLİK VE İSİM EŞLEŞTİRME (KAPSAMLI)
# ---------------------------------------------------------
def normalize_name(name):
    name = str(name).strip()
    replacements = {
        "Man City": "Manchester City", "Man Utd": "Manchester United", 
        "Man United": "Manchester United", "Spurs": "Tottenham Hotspur", 
        "Tottenham": "Tottenham Hotspur", "Newcastle": "Newcastle United", 
        "West Ham": "West Ham United", "Wolves": "Wolverhampton Wanderers", 
        "Wolverhampton": "Wolverhampton Wanderers",
        "Nott'm Forest": "Nottingham Forest", "Leicester": "Leicester City", 
        "Norwich": "Norwich City", "Leeds": "Leeds United", 
        "Luton": "Luton Town", "Sheff Utd": "Sheffield United",
        "Sheffield Utd": "Sheffield United", "Brentford": "Brentford",
        "Bournemouth": "Bournemouth", "AFC Bournemouth": "Bournemouth",
        "Fulham": "Fulham",
        # BRIGHTON VARYASYONLARI
        "Brighton & Hove Albion": "Brighton",
        "Brighton and Hove Albion": "Brighton",
        "Brighton & Hove Albion FC": "Brighton",
        "Brighton FC": "Brighton"
    }
    return replacements.get(name, name)

matches['HomeTeam'] = matches['HomeTeam'].apply(normalize_name)
matches['AwayTeam'] = matches['AwayTeam'].apply(normalize_name)
players['Team'] = players['Team'].apply(normalize_name)

matches['Date'] = pd.to_datetime(matches['Date'], dayfirst=True, errors='coerce')

def get_season_year(x):
    try: return int(str(x).split('-')[0])
    except: return int(x)
players['SeasonYear'] = players['Season'].apply(get_season_year)

# ---------------------------------------------------------
# 2. POZİSYON TAHMİNİ (Eğer Position sütunu yoksa)
# ---------------------------------------------------------
# Sütun isimlerini kontrol et
cols = players.columns.tolist()
pos_col = None
for c in ['Position', 'Best Position', 'Preferred Positions', 'Pos']:
    if c in cols:
        pos_col = c
        break

if pos_col is None:
    print("⚠️ 'Position' sütunu bulunamadı. İstatistiklere göre tahmin ediliyor...")
    
    def infer_position(row):
        # Güvenli veri alma
        def g(k): return row.get(k, 0)
        
        # Basit mantık
        gk_stats = (g('GKDiving') + g('GKHandling')) / 2
        def_stats = (g('StandingTackle') + g('Interceptions') + g('Marking')) / 3
        mid_stats = (g('ShortPassing') + g('BallControl') + g('Vision')) / 3
        att_stats = (g('Finishing') + g('Volleys') + g('Positioning')) / 3
        
        if gk_stats > 40: return 'GK'
        if def_stats > mid_stats and def_stats > att_stats: return 'DF'
        if att_stats > mid_stats and att_stats > def_stats: return 'FW'
        return 'MF'

    players['Position'] = players.apply(infer_position, axis=1)
    pos_col = 'Position'
else:
    print(f"✅ Pozisyon sütunu bulundu: {pos_col}")

# ---------------------------------------------------------
# 3. TAKIM GÜÇLERİ HESAPLAMA
# ---------------------------------------------------------
print("Takım güçleri hesaplanıyor...")

def calculate_team_power(group):
    top_15 = group.nlargest(15, 'Overall')
    return pd.Series({
        'FIFA_Overall': top_15['Overall'].mean(),
        'FIFA_Attack': top_15[['Finishing', 'ShotPower', 'Positioning', 'Volleys']].mean().mean(),
        'FIFA_Midfield': top_15[['ShortPassing', 'Vision', 'BallControl', 'Dribbling']].mean().mean(),
        'FIFA_Defense': top_15[['Marking', 'StandingTackle', 'Interceptions', 'SlidingTackle']].mean().mean(),
        'FIFA_Physical': top_15[['Stamina', 'Strength', 'SprintSpeed', 'Acceleration']].mean().mean()
    })

# Warning fix
team_fifa_stats = players.groupby(['Team', 'SeasonYear']).apply(calculate_team_power, include_groups=False).reset_index()

# ---------------------------------------------------------
# 4. KADRO LİSTELERİ
# ---------------------------------------------------------
print("Kadro listeleri oluşturuluyor...")
latest_year = players['SeasonYear'].max()
latest_players = players[players['SeasonYear'] == latest_year].copy()
all_match_teams = set(matches['HomeTeam'].unique()) | set(matches['AwayTeam'].unique())

# BRIGHTON ZORLAMASI: Listeye manuel ekle
all_match_teams.add("Brighton")

team_rosters = {}
for team in all_match_teams:
    team_p = latest_players[latest_players['Team'] == team]
    if len(team_p) > 0:
        roster = team_p.sort_values(by=['Finishing', 'Overall'], ascending=False).head(15)
        team_rosters[team] = roster[['Name', 'Finishing', pos_col]].rename(columns={pos_col: 'Position'}).to_dict('records')
    else:
        # Takım FIFA'da yoksa uydurma kadro
        team_rosters[team] = [
            {'Name': f'{team} Golcü', 'Finishing': 75, 'Position': 'FW'},
            {'Name': f'{team} Kaptan', 'Finishing': 70, 'Position': 'MF'},
            {'Name': f'{team} Kanat', 'Finishing': 72, 'Position': 'RW'}
        ]

# ---------------------------------------------------------
# 5. VERİ BİRLEŞTİRME
# ---------------------------------------------------------
print("Veriler birleştiriliyor...")
matches['SeasonYear'] = matches['Date'].apply(lambda x: x.year if x.month >= 8 else x.year - 1)

# Merge
df_merged = pd.merge(matches, team_fifa_stats, left_on=['HomeTeam', 'SeasonYear'], right_on=['Team', 'SeasonYear'], how='left')
cols = ['FIFA_Overall', 'FIFA_Attack', 'FIFA_Midfield', 'FIFA_Defense', 'FIFA_Physical']
df_merged.rename(columns={c: f'Home_{c}' for c in cols}, inplace=True)
df_merged.drop(columns=['Team'], inplace=True)

df_merged = pd.merge(df_merged, team_fifa_stats, left_on=['AwayTeam', 'SeasonYear'], right_on=['Team', 'SeasonYear'], how='left')
df_merged.rename(columns={c: f'Away_{c}' for c in cols}, inplace=True)
df_merged.drop(columns=['Team'], inplace=True)

# NaN Doldur (Eksik veriler için lig ortalaması)
for c in cols:
    mean_val = df_merged[f'Home_{c}'].mean()
    df_merged[f'Home_{c}'] = df_merged[f'Home_{c}'].fillna(mean_val)
    df_merged[f'Away_{c}'] = df_merged[f'Away_{c}'].fillna(mean_val)

def get_result(row):
    if row['FTHG'] > row['FTAG']: return 2
    elif row['FTHG'] == row['FTAG']: return 1
    else: return 0
df_merged['MatchResult'] = df_merged.apply(get_result, axis=1)

# ---------------------------------------------------------
# 6. MODEL EĞİTİMİ VE PROFİL OLUŞTURMA
# ---------------------------------------------------------
split_date = pd.to_datetime("2023-08-01")
train_df = df_merged[df_merged['Date'] < split_date].copy()
real_23_24_df = df_merged[df_merged['Date'] >= split_date].copy()

# Performans Profilleri (Matches'dan gelen)
stat_cols = ['Shots', 'SOT', 'Corners', 'Possession']
h_ren = {f'Home{c}': c for c in stat_cols}; h_ren['HomeTeam'] = 'Team'
a_ren = {f'Away{c}': c for c in stat_cols}; a_ren['AwayTeam'] = 'Team'

h_stats = train_df[['HomeTeam'] + ['Home' + c for c in stat_cols]].rename(columns=h_ren)
a_stats = train_df[['AwayTeam'] + ['Away' + c for c in stat_cols]].rename(columns=a_ren)
performance_profiles = pd.concat([h_stats, a_stats]).groupby('Team').mean()

# FIFA Profilleri
latest_fifa = team_fifa_stats[team_fifa_stats['SeasonYear'] == team_fifa_stats['SeasonYear'].max()]
latest_fifa = latest_fifa.drop_duplicates('Team').set_index('Team')[cols]

# --- BRIGHTON YAMASI (ZORLA EKLEME) ---
# Eğer Brighton istatistiklerde yoksa (ki yoktu), ortalama bir profil oluşturup ekle.
if "Brighton" not in performance_profiles.index:
    print("🛠️  Brighton performans profiline MANUEL ekleniyor...")
    avg_perf = performance_profiles.mean()
    performance_profiles.loc["Brighton"] = avg_perf

if "Brighton" not in latest_fifa.index:
    print("🛠️  Brighton FIFA profiline MANUEL ekleniyor...")
    avg_fifa = latest_fifa.mean()
    latest_fifa.loc["Brighton"] = avg_fifa

# Model Eğitimi
X, y = [], []
print("Model eğitiliyor...")

for idx, row in train_df.iterrows():
    h, a = row['HomeTeam'], row['AwayTeam']
    # Artık Brighton eklendiği için hata vermeyecek
    if h in performance_profiles.index and a in performance_profiles.index:
        h_perf = performance_profiles.loc[h].values
        a_perf = performance_profiles.loc[a].values
        h_fifa = row[[f'Home_{c}' for c in cols]].values.astype(float)
        a_fifa = row[[f'Away_{c}' for c in cols]].values.astype(float)
        
        X.append(np.concatenate([h_perf, h_fifa, a_perf, a_fifa]))
        y.append(row['MatchResult'])

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
model.fit(X, y)

# ---------------------------------------------------------
# 7. KAYIT
# ---------------------------------------------------------
export_data = {
    'model': model,
    'performance_profiles': performance_profiles,
    'fifa_profiles': latest_fifa,
    'team_rosters': team_rosters,
    'real_23_24_data': real_23_24_df
}

joblib.dump(export_data, 'super_model.pkl')
print(f"✅ İŞLEM TAMAM! 'Brighton' dahil tüm takımlar hazır.")
print(f"✅ {len(team_rosters)} takımın kadrosu kaydedildi.")