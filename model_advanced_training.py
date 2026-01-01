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
# String temizliği
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
        if (g('GKDiving') + g('GKHandling'))/2 > 40: return 'GK'
        if (g('Finishing') + g('Volleys'))/2 > (g('StandingTackle') + g('Interceptions'))/2: return 'FW'
        return 'MF'
    players['Position'] = players.apply(infer_position, axis=1)
    pos_col = 'Position'

# ---------------------------------------------------------
# 3. TAKIM GÜÇLERİ
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

# ---------------------------------------------------------
# 4. KADRO LİSTELERİ VE GENİŞLETİLMİŞ MANUEL YAMA
# ---------------------------------------------------------
print("Kadro listeleri oluşturuluyor...")
latest_year = players['SeasonYear'].max()
latest_players = players[players['SeasonYear'] == latest_year].copy()
all_match_teams = set(matches['HomeTeam'].unique()) | set(matches['AwayTeam'].unique())

# Eksik olma ihtimali yüksek takımları manuel listeye ekleyelim
if "Brighton" not in all_match_teams: all_match_teams.add("Brighton")

team_rosters = {}
for team in all_match_teams:
    team_p = latest_players[latest_players['Team'] == team]
    if len(team_p) > 0:
        roster = team_p.sort_values(by=['Finishing', 'Overall'], ascending=False).head(15)
        team_rosters[team] = roster[['Name', 'Finishing', pos_col]].rename(columns={pos_col: 'Position'}).to_dict('records')
    else:
        team_rosters[team] = [] 

# --- MANUEL KADRO YAMASI (GENİŞLETİLMİŞ LİSTE) ---
# Buradaki amaç gollerin tek bir oyuncuya yığılmasını engellemek için her takıma en az 4-5 golcü yazmaktır.

manual_squads = {
    "Luton Town": [
        {"Name": "Carlton Morris", "Finishing": 78, "Position": "FW"},
        {"Name": "Elijah Adebayo", "Finishing": 76, "Position": "FW"},
        {"Name": "Ross Barkley", "Finishing": 74, "Position": "MF"},
        {"Name": "Tahith Chong", "Finishing": 70, "Position": "MF"},
        {"Name": "Chiedozie Ogbene", "Finishing": 69, "Position": "FW"},
        {"Name": "Andros Townsend", "Finishing": 71, "Position": "MF"}
    ],
    "Sheffield United": [
        {"Name": "Cameron Archer", "Finishing": 75, "Position": "FW"},
        {"Name": "Oli McBurnie", "Finishing": 74, "Position": "FW"},
        {"Name": "Gustavo Hamer", "Finishing": 73, "Position": "MF"},
        {"Name": "Ben Brereton Díaz", "Finishing": 76, "Position": "FW"},
        {"Name": "James McAtee", "Finishing": 71, "Position": "MF"}
    ],
    "Burnley": [
        {"Name": "Lyle Foster", "Finishing": 75, "Position": "FW"},
        {"Name": "Zeki Amdouni", "Finishing": 74, "Position": "FW"},
        {"Name": "David Datro Fofana", "Finishing": 73, "Position": "FW"},
        {"Name": "Josh Brownhill", "Finishing": 71, "Position": "MF"},
        {"Name": "Wilson Odobert", "Finishing": 70, "Position": "FW"}
    ],
    "Brighton": [
        {"Name": "Joao Pedro", "Finishing": 79, "Position": "FW"},
        {"Name": "Evan Ferguson", "Finishing": 78, "Position": "FW"},
        {"Name": "Kaoru Mitoma", "Finishing": 76, "Position": "FW"},
        {"Name": "Pascal Gross", "Finishing": 75, "Position": "MF"},
        {"Name": "Simon Adingra", "Finishing": 74, "Position": "FW"},
        {"Name": "Danny Welbeck", "Finishing": 75, "Position": "FW"}
    ],
    "Fulham": [ # ARTIK 'FULHAM FORVET' OLMAYACAK
        {"Name": "Rodrigo Muniz", "Finishing": 78, "Position": "FW"},
        {"Name": "Raúl Jiménez", "Finishing": 77, "Position": "FW"},
        {"Name": "Alex Iwobi", "Finishing": 73, "Position": "MF"},
        {"Name": "Andreas Pereira", "Finishing": 74, "Position": "MF"},
        {"Name": "Harry Wilson", "Finishing": 75, "Position": "FW"},
        {"Name": "Willian", "Finishing": 74, "Position": "FW"}
    ],
    "Bournemouth": [
        {"Name": "Dominic Solanke", "Finishing": 80, "Position": "FW"},
        {"Name": "Antoine Semenyo", "Finishing": 75, "Position": "FW"},
        {"Name": "Justin Kluivert", "Finishing": 74, "Position": "FW"},
        {"Name": "Marcus Tavernier", "Finishing": 72, "Position": "MF"},
        {"Name": "Enes Ünal", "Finishing": 76, "Position": "FW"}
    ],
    "Brentford": [
        {"Name": "Ivan Toney", "Finishing": 81, "Position": "FW"},
        {"Name": "Bryan Mbeumo", "Finishing": 77, "Position": "FW"},
        {"Name": "Yoane Wissa", "Finishing": 76, "Position": "FW"},
        {"Name": "Neal Maupay", "Finishing": 75, "Position": "FW"},
        {"Name": "Mathias Jensen", "Finishing": 70, "Position": "MF"}
    ],
    "Nottingham Forest": [
        {"Name": "Chris Wood", "Finishing": 78, "Position": "FW"},
        {"Name": "Taiwo Awoniyi", "Finishing": 77, "Position": "FW"},
        {"Name": "Morgan Gibbs-White", "Finishing": 74, "Position": "MF"},
        {"Name": "Anthony Elanga", "Finishing": 73, "Position": "FW"},
        {"Name": "Callum Hudson-Odoi", "Finishing": 72, "Position": "FW"}
    ],
    "Crystal Palace": [ # Eksik olma ihtimaline karşı
        {"Name": "Jean-Philippe Mateta", "Finishing": 78, "Position": "FW"},
        {"Name": "Eberechi Eze", "Finishing": 77, "Position": "MF"},
        {"Name": "Michael Olise", "Finishing": 76, "Position": "FW"},
        {"Name": "Odsonne Édouard", "Finishing": 75, "Position": "FW"},
        {"Name": "Jordan Ayew", "Finishing": 72, "Position": "FW"}
    ]
}

# Eksik takımları manuel listeyle doldur
for team_name, squad in manual_squads.items():
    # Eğer takımın kadrosu boşsa veya çok azsa (hata varsa) manuel listeyi kullan
    if team_name in team_rosters and len(team_rosters[team_name]) < 3:
        print(f"🛠️  {team_name} için manuel kadro yüklendi.")
        team_rosters[team_name] = squad
    # Eğer takım listede hiç yoksa ekle
    elif team_name not in team_rosters:
        team_rosters[team_name] = squad

# Hala boş kalan varsa son çare generic isimler
for team in all_match_teams:
    if not team_rosters.get(team):
        print(f"⚠️ {team} için hala kadro yok, generic isimler atanıyor.")
        team_rosters[team] = [
            {'Name': f'{team} Forvet', 'Finishing': 75, 'Position': 'FW'},
            {'Name': f'{team} Kaptan', 'Finishing': 70, 'Position': 'MF'}
        ]

# ---------------------------------------------------------
# 5. VERİ BİRLEŞTİRME VE EĞİTİM
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

# NaN Doldur
for c in cols:
    mean_val = df_merged[f'Home_{c}'].mean()
    df_merged[f'Home_{c}'] = df_merged[f'Home_{c}'].fillna(mean_val)
    df_merged[f'Away_{c}'] = df_merged[f'Away_{c}'].fillna(mean_val)

def get_result(row):
    if row['FTHG'] > row['FTAG']: return 2
    elif row['FTHG'] == row['FTAG']: return 1
    else: return 0
df_merged['MatchResult'] = df_merged.apply(get_result, axis=1)

# Profil Oluşturma
split_date = pd.to_datetime("2023-08-01")
train_df = df_merged[df_merged['Date'] < split_date].copy()
real_23_24_df = df_merged[df_merged['Date'] >= split_date].copy()

stat_cols = ['Shots', 'SOT', 'Corners', 'Possession']
h_ren = {f'Home{c}': c for c in stat_cols}; h_ren['HomeTeam'] = 'Team'
a_ren = {f'Away{c}': c for c in stat_cols}; a_ren['AwayTeam'] = 'Team'

h_stats = train_df[['HomeTeam'] + ['Home' + c for c in stat_cols]].rename(columns=h_ren)
a_stats = train_df[['AwayTeam'] + ['Away' + c for c in stat_cols]].rename(columns=a_ren)
performance_profiles = pd.concat([h_stats, a_stats]).groupby('Team').mean()

latest_fifa = team_fifa_stats[team_fifa_stats['SeasonYear'] == team_fifa_stats['SeasonYear'].max()]
latest_fifa = latest_fifa.drop_duplicates('Team').set_index('Team')[cols]

# Eksik Profilleri Tamamla
for t in all_match_teams:
    if t not in performance_profiles.index:
        performance_profiles.loc[t] = performance_profiles.mean()
    if t not in latest_fifa.index:
        latest_fifa.loc[t] = latest_fifa.mean()

# Model Eğitimi
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

# Kayıt
export_data = {
    'model': model,
    'performance_profiles': performance_profiles,
    'fifa_profiles': latest_fifa,
    'team_rosters': team_rosters, 
    'real_23_24_data': real_23_24_df
}

joblib.dump(export_data, 'super_model.pkl')
print(f"✅ BAŞARILI! Fulham ve diğer takımların kadroları eklendi. 'super_model.pkl' güncellendi.")