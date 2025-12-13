import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

st.set_page_config(page_title="FC 24 Destekli Simülasyon", layout="wide")

# 1. YÜKLEME
@st.cache_resource
def load_data():
    data = joblib.load('super_model.pkl')
    return data['model'], data['performance_profiles'], data['fifa_profiles'], data['real_23_24_data']

try:
    model, perf_profiles, fifa_profiles, real_df = load_data()
except FileNotFoundError:
    st.error("Lütfen önce 'model_advanced.py' dosyasını çalıştırın!")
    st.stop()

# Takımları Belirle (Gerçek fikstürden veya FIFA listesinden)
# Simülasyon için 23-24 sezonundaki takımları alalım
real_teams = sorted(pd.concat([real_df['HomeTeam'], real_df['AwayTeam']]).unique())

# 2. YARDIMCI FONKSİYONLAR
def get_team_vector(team_name):
    """Bir takımın hem saha içi istatistiğini hem FIFA gücünü getirir ve birleştirir."""
    # 1. Performans (Şut, Korner vs.)
    if team_name in perf_profiles.index:
        perf = perf_profiles.loc[team_name].values
    else:
        perf = perf_profiles.mean().values # Bilinmiyorsa lig ortalaması

    # 2. FIFA Gücü
    # Model eğitimi sırasında: ['FIFA_Overall', 'FIFA_Attack', 'FIFA_Defense', 'FIFA_Physical'] kullandık
    needed_cols = ['FIFA_Overall', 'FIFA_Attack', 'FIFA_Defense', 'FIFA_Physical']
    
    if team_name in fifa_profiles.index:
        fifa = fifa_profiles.loc[team_name][needed_cols].values
    else:
        fifa = fifa_profiles[needed_cols].mean().values # Bilinmiyorsa lig ortalaması
        
    return np.concatenate([perf, fifa])

def generate_sim_score(home, away, outcome):
    # Skor üretirken Takımın FIFA "FIFA_Attack" gücüne bakalım
    try:
        # FIFA profili varsa oradan, yoksa varsayılan
        h_att_power = fifa_profiles.loc[home]['FIFA_Attack'] if home in fifa_profiles.index else 75
        a_att_power = fifa_profiles.loc[away]['FIFA_Attack'] if away in fifa_profiles.index else 75
        
        # SOT (İsabetli Şut) verisi de var
        h_sot = perf_profiles.loc[home]['SOT'] if home in perf_profiles.index else 4
        a_sot = perf_profiles.loc[away]['SOT'] if away in perf_profiles.index else 3
    except:
        h_att_power, a_att_power = 75, 75
        h_sot, a_sot = 4, 3

    # Gol Beklentisi Formülü: (İsabetli Şut * 0.3) + (FIFA Gücü Bonusu)
    # FIFA gücü 85 üstüyse ekstra gol şansı artar
    h_lambda = (h_sot / 2.8) * (h_att_power / 75)
    a_lambda = (a_sot / 2.8) * (a_att_power / 75)
    
    h_goals = np.random.poisson(h_lambda)
    a_goals = np.random.poisson(a_lambda)
    
    # Sonuca Zorla
    if outcome == 1: # Draw
        g = int((h_goals + a_goals)/2)
        return g, g
    elif outcome == 2: # Home
        if h_goals <= a_goals: h_goals = a_goals + 1
        return h_goals, a_goals
    else: # Away
        if a_goals <= h_goals: a_goals = h_goals + 1
        return h_goals, a_goals

def run_simulation():
    table = {t: {'P':0, 'W':0, 'D':0, 'L':0, 'Pts':0, 'GF':0, 'GA':0, 'GD':0} for t in real_teams}
    
    cnt = 0
    total = len(real_teams) * (len(real_teams)-1)
    bar = st.progress(0)
    
    for home in real_teams:
        for away in real_teams:
            if home == away: continue
            
            # Vektörleri oluştur
            h_vec = get_team_vector(home)
            a_vec = get_team_vector(away)
            
            # Tahmin
            input_vec = np.concatenate([h_vec, a_vec]).reshape(1, -1)
            probs = model.predict_proba(input_vec)[0]
            res = np.random.choice([0,1,2], p=probs)
            
            # Skor
            hg, ag = generate_sim_score(home, away, res)
            
            # Tablo
            cnt+=1
            table[home]['P']+=1; table[away]['P']+=1
            table[home]['GF']+=hg; table[away]['GF']+=ag
            table[home]['GA']+=ag; table[away]['GA']+=hg
            table[home]['GD']+=(hg-ag); table[away]['GD']+=(ag-hg)
            
            if res==2:
                table[home]['W']+=1; table[home]['Pts']+=3; table[away]['L']+=1
            elif res==1:
                table[home]['D']+=1; table[home]['Pts']+=1; table[away]['D']+=1; table[away]['Pts']+=1
            else:
                table[away]['W']+=1; table[away]['Pts']+=3; table[home]['L']+=1
            
            if cnt % 20 == 0: bar.progress(cnt/total)
            
    bar.empty()
    return pd.DataFrame.from_dict(table, orient='index').sort_values(by=['Pts', 'GD'], ascending=False)

def get_real_table():
    table = {t: {'P':0, 'W':0, 'D':0, 'L':0, 'Pts':0} for t in real_teams}
    for _, row in real_df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        if h not in real_teams or a not in real_teams: continue
        hg, ag = row['FTHG'], row['FTAG']
        
        if hg > ag: table[h]['Pts']+=3
        elif hg == ag: table[h]['Pts']+=1; table[a]['Pts']+=1
        else: table[a]['Pts']+=3
    return pd.DataFrame.from_dict(table, orient='index').sort_values(by='Pts', ascending=False)

# 3. ARAYÜZ
st.title("🎮 FIFA Oyuncu Verileriyle Güçlendirilmiş Simülasyon")
st.markdown("""
Bu model artık takımların sadece geçmiş maç sonuçlarına değil, **oyuncu kadro kalitesine (Overall, Pace, Shooting, Defending)** de bakıyor.
""")

if st.button("🚀 SÜPER SİMÜLASYONU BAŞLAT", type="primary"):
    c1, c2 = st.columns(2)
    
    sim_table = run_simulation()
    real_table = get_real_table()
    
    def color_rows(s, df):
        try:
            rank = df.index.get_loc(s.name)
            if rank == 0: return ['background-color: #ffd700; color: black'] * len(s)
            elif rank < 4: return ['background-color: #e0f7fa; color: black'] * len(s)
            elif rank >= len(df)-3: return ['background-color: #ffcdd2; color: black'] * len(s)
            return [''] * len(s)
        except: return [''] * len(s)

    with c1:
        st.subheader("🤖 Yapay Zeka (Kadro Kalitesi Bazlı)")
        st.dataframe(sim_table.style.apply(lambda x: color_rows(x, sim_table), axis=1), height=800)
        
    with c2:
        st.subheader("🌍 Gerçek Puanlar (Referans)")
        st.dataframe(real_table[['Pts']].style.apply(lambda x: color_rows(x, real_table), axis=1), height=800)