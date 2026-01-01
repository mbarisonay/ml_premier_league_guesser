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
df_obj = matches.select_dtypes(['object'])
matches[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

players = pd.read_csv(player_data_file)
df_obj_p = players.select_dtypes(['object'])
players[df_obj_p.columns] = df_obj_p.apply(lambda x: x.str.strip())

# ---------------------------------------------------------
# 1. İSİM EŞLEŞTİRME
# ---------------------------------------------------------
def normalize_name(name):
    name = str(name).strip()
    replacements = {
        "Man City": "Manchester City", "Man Utd": "Manchester United", 
        "Man United": "Manchester United", "Spurs": "Tottenham Hotspur", 
        "Tottenham": "Tottenham Hotspur", "Newcastle": "Newcastle United", 
        "West Ham": "West Ham United", "Wolves": "Wolverhampton Wanderers", 
        "Wolverhampton": "Wolverhampton Wanderers", "Nott'm Forest": "Nottingham Forest", 
        "Leicester": "Leicester City", "Norwich": "Norwich City", "Leeds": "Leeds United", 
        "Luton": "Luton Town", "Sheff Utd": "Sheffield United",
        "Sheffield Utd": "Sheffield United", "Brentford": "Brentford",
        "Bournemouth": "Bournemouth", "AFC Bournemouth": "Bournemouth", "Fulham": "Fulham",
        "Brighton & Hove Albion": "Brighton", "Brighton and Hove Albion": "Brighton", 
        "Brighton FC": "Brighton", "Nottingham Forest": "Nottingham Forest"
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
# 2. POZİSYON TAHMİNİ
# ---------------------------------------------------------
cols = players.columns.tolist()
pos_col = None
for c in ['Position', 'Best Position', 'Preferred Positions', 'Pos']:
    if c in cols:
        pos_col = c
        break

if pos_col is None:
    print("⚠️ 'Position' sütunu bulunamadı. Tahmin ediliyor...")
    def infer_position(row):
        def g(k): return row.get(k, 0)
        # Kaleci refleksi yüksekse GK
        if (g('GKDiving') + g('GKHandling'))/2 > 40: return 'GK'
        # Defansif özellikleri yüksekse DF
        if (g('StandingTackle') + g('Interceptions')) > (g('Finishing') + g('Volleys')): return 'DF'
        # Pas yüksekse MF
        if (g('ShortPassing') + g('Vision')) > (g('Finishing') + g('StandingTackle')): return 'MF'
        return 'FW'
    players['Position'] = players.apply(infer_position, axis=1)
    pos_col = 'Position'

# ---------------------------------------------------------
# 3. KADRO LİSTELERİ (DÜZELTİLDİ: OVERALL'A GÖRE SEÇİM)
# ---------------------------------------------------------
print("Kadro listeleri oluşturuluyor...")
latest_year = players['SeasonYear'].max()
latest_players = players[players['SeasonYear'] == latest_year].copy()
all_match_teams = set(matches['HomeTeam'].unique()) | set(matches['AwayTeam'].unique())
if "Brighton" not in all_match_teams: all_match_teams.add("Brighton")

team_rosters = {}
for team in all_match_teams:
    team_p = latest_players[latest_players['Team'] == team]
    if len(team_p) > 0:
        # DÜZELTME BURADA: Artık Finishing değil, OVERALL puanına göre en iyi 15 kişiyi alıyoruz.
        # Böylece Kaleci ve Defanslar da listeye giriyor.
        roster = team_p.sort_values(by=['Overall'], ascending=False).head(16)
        team_rosters[team] = roster[['Name', 'Finishing', 'Overall', pos_col]].rename(columns={pos_col: 'Position'}).to_dict('records')
    else:
        team_rosters[team] = []

# --- MANUEL KADRO YAMASI (Tam Kadrolar) ---
# Eksik takımlar için gerçekçi 11'ler
manual_squads = {
    "Luton Town": [
        {"Name": "Kaminski", "Finishing": 10, "Overall": 75, "Position": "GK"},
        {"Name": "Mengi", "Finishing": 30, "Overall": 73, "Position": "DF"},
        {"Name": "Bell", "Finishing": 40, "Overall": 72, "Position": "DF"},
        {"Name": "Burke", "Finishing": 35, "Overall": 71, "Position": "DF"},
        {"Name": "Doughty", "Finishing": 60, "Overall": 74, "Position": "MF"},
        {"Name": "Barkley", "Finishing": 74, "Overall": 78, "Position": "MF"},
        {"Name": "Lokonga", "Finishing": 65, "Overall": 75, "Position": "MF"},
        {"Name": "Townsend", "Finishing": 71, "Overall": 74, "Position": "MF"},
        {"Name": "Chong", "Finishing": 70, "Overall": 73, "Position": "MF"},
        {"Name": "Morris", "Finishing": 78, "Overall": 76, "Position": "FW"},
        {"Name": "Adebayo", "Finishing": 76, "Overall": 75, "Position": "FW"}
    ],
    # Diğer takımları FIFA verisinden çekecek, buraya sadece FIFA'da olmayanları ekle
}

for team_name, squad in manual_squads.items():
    if team_name in team_rosters and len(team_rosters[team_name]) < 5:
        team_rosters[team_name] = squad
    elif team_name not in team_rosters:
        team_rosters[team_name] = squad

# ---------------------------------------------------------
# 4. TAKIM GÜÇLERİ VE EĞİTİM
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

team_fifa_stats = players.groupby(['Team', 'SeasonYear']).apply(calculate_team_power, include_groups=False).reset_index()

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

for c in cols:
    mean_val = df_merged[f'Home_{c}'].mean()
    df_merged[f'Home_{c}'] = df_merged[f'Home_{c}'].fillna(mean_val)
    df_merged[f'Away_{c}'] = df_merged[f'Away_{c}'].fillna(mean_val)

def get_result(row):
    if row['FTHG'] > row['FTAG']: return 2
    elif row['FTHG'] == row['FTAG']: return 1
    else: return 0
df_merged['MatchResult'] = df_merged.apply(get_result, axis=1)

# Eğitim
split_date = pd.to_datetime("2023-08-01")
train_df = df_merged[df_merged['Date'] < split_date].copy()
real_23_24_df = df_merged[df_merged['Date'] >= split_date].copy()

stat_cols = ['Shots', 'SOT', 'Corners', 'Possession']
h_ren = {f'Home{c}': c for c in stat_cols}; h_ren['HomeTeam'] = 'Team'
a_ren = {f'Away{c}': c for c in stat_cols}; a_ren['AwayTeam'] = 'Team'
h_stats = train_df[['HomeTeam'] + ['Home' + c for c in stat_cols]].rename(columns=h_ren)
a_stats = train_df[['AwayTeam'] + ['Away' + c for c in stat_cols]].rename(columns=a_ren)
performance_profiles = pd.concat([h_stats, a_stats]).groupby('Team').mean()

latest_fifa = team_fifa_stats[team_fifa_stats['SeasonYear'] == team_fifa_stats['SeasonYear'].max()].drop_duplicates('Team').set_index('Team')[cols]

# Eksik Profiller
for t in all_match_teams:
    if t not in performance_profiles.index: performance_profiles.loc[t] = performance_profiles.mean()
    if t not in latest_fifa.index: latest_fifa.loc[t] = latest_fifa.mean()

X, y = [], []
print("Model eğitiliyor...")
for idx, row in train_df.iterrows():
    h, a = row['HomeTeam'], row['AwayTeam']
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

export_data = {
    'model': model,
    'performance_profiles': performance_profiles,
    'fifa_profiles': latest_fifa,
    'team_rosters': team_rosters,
    'real_23_24_data': real_23_24_df
}

joblib.dump(export_data, 'super_model.pkl')
print(f"✅ BAŞARILI! 'super_model.pkl' doğru kadrolarla yenilendi.")