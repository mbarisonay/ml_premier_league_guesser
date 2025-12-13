import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ---------------------------------------------------------
# 1. SAYFA AYARLARI VE STİL
# ---------------------------------------------------------
st.set_page_config(page_title="PL AI Super Hub", layout="wide", page_icon="⚽")

# Özel CSS Stilleri (İstatistik Barları İçin)
st.markdown("""
<style>
    .stat-container { margin-bottom: 15px; }
    .stat-header { display: flex; justify-content: space-between; font-size: 14px; font-weight: bold; margin-bottom: 5px; }
    .stat-bar-bg { display: flex; height: 10px; background-color: #262730; border-radius: 5px; overflow: hidden; }
    .bar-home { background-color: #4CAF50; height: 100%; }
    .bar-away { background-color: #FF5252; height: 100%; }
    .team-header { font-size: 18px; font-weight: bold; margin-bottom: 10px; border-bottom: 2px solid #333; padding-bottom: 5px; }
    .scorer-item { font-size: 14px; margin-bottom: 4px; color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# Güvenli Renklendirme
def safe_style(s, df):
    try:
        rank = df.index.get_loc(s.name)
        if rank == 0: return ['background-color: #ffd700; color: black'] * len(s) # Şampiyon
        elif rank < 4: return ['background-color: #e0f7fa; color: black'] * len(s) # UCL
        elif rank >= len(df) - 3: return ['background-color: #ffcdd2; color: black'] * len(s) # Düşme
        else: return [''] * len(s)
    except:
        return [''] * len(s)

# ---------------------------------------------------------
# 2. VERİ YÜKLEME
# ---------------------------------------------------------
@st.cache_resource
def load_data():
    try:
        data = joblib.load('super_model.pkl')
        return (data['model'], 
                data['performance_profiles'], 
                data['fifa_profiles'], 
                data.get('team_rosters', {}), 
                data['real_23_24_data'])
    except FileNotFoundError:
        return None, None, None, None, None

model, perf_profiles, fifa_profiles, team_rosters, real_df = load_data()

if model is None:
    st.error("❌ 'super_model.pkl' bulunamadı! Lütfen 'model_advanced_training.py' kodunu çalıştır.")
    st.stop()

all_teams = sorted(pd.concat([real_df['HomeTeam'], real_df['AwayTeam']]).unique())

# ---------------------------------------------------------
# 3. YARDIMCI MOTORLAR
# ---------------------------------------------------------

def get_team_vector(team_name):
    # Performans
    if team_name in perf_profiles.index: perf = perf_profiles.loc[team_name].values
    else: perf = perf_profiles.mean().values 

    # FIFA (18 feature hatasını önlemek için Midfield eklendi)
    needed = ['FIFA_Overall', 'FIFA_Attack', 'FIFA_Midfield', 'FIFA_Defense', 'FIFA_Physical']
    if team_name in fifa_profiles.index: fifa = fifa_profiles.loc[team_name][needed].values
    else: fifa = fifa_profiles[needed].mean().values
        
    return np.concatenate([perf, fifa])

def simulate_scorers(team, goals):
    """Kadro verisinden golcü seçer."""
    scorers = []
    if goals == 0: return []
    
    roster = team_rosters.get(team, [])
    if not roster:
        names = ["Forvet", "Ortasaha", "Sürpriz İsim"]
        weights = [0.6, 0.3, 0.1]
    else:
        names = [p['Name'] for p in roster]
        finishing = np.array([p['Finishing'] for p in roster], dtype=float)
        weights = np.exp(finishing / 12)
        weights /= weights.sum()
    
    for _ in range(goals):
        scorer = np.random.choice(names, p=weights)
        minute = np.random.randint(1, 98)
        scorers.append(f"⚽ {scorer} ({minute}')")
    
    scorers.sort(key=lambda x: int(x.split('(')[1].split("'")[0]))
    return scorers

def generate_live_stats(home, away, hg, ag):
    base_poss = 50
    if hg > ag: base_poss = 45
    elif ag > hg: base_poss = 55
    
    h_poss = np.random.randint(base_poss-5, base_poss+10)
    a_poss = 100 - h_poss
    
    h_shots = max(hg + np.random.randint(2, 8), int(hg * 2.5) + 5)
    a_shots = max(ag + np.random.randint(2, 8), int(ag * 2.5) + 5)
    
    h_sot = max(hg, int(h_shots * np.random.uniform(0.3, 0.6)))
    a_sot = max(ag, int(a_shots * np.random.uniform(0.3, 0.6)))
    
    h_xg = round(h_sot * 0.12 + (h_shots - h_sot) * 0.03 + (hg * 0.4), 2)
    a_xg = round(a_sot * 0.12 + (a_shots - a_sot) * 0.03 + (ag * 0.4), 2)
    
    h_pass = h_poss*4 + np.random.randint(50,100)
    a_pass = a_poss*4 + np.random.randint(50,100)

    # Değerleri sayısal olarak döndürüyoruz (Görselleştirme fonksiyonu işleyecek)
    return {
        'Topla Oynama (%)': (h_poss, a_poss),
        'Gol Beklentisi (xG)': (h_xg, a_xg),
        'Toplam Şut': (h_shots, a_shots),
        'İsabetli Şut': (h_sot, a_sot),
        'Pas Sayısı': (h_pass, a_pass),
        'Korner': (np.random.randint(2, 12), np.random.randint(2, 12)),
        'Faul': (np.random.randint(4, 15), np.random.randint(4, 15))
    }

# YENİ: GÜZEL İSTATİSTİK ÇUBUĞU ÇİZEN FONKSİYON
def draw_stat_bar(stat_name, h_val, a_val):
    total = h_val + a_val
    if total == 0: 
        h_pct = 50
        a_pct = 50
    else:
        h_pct = (h_val / total) * 100
        a_pct = (a_val / total) * 100
    
    # HTML ile özel progress bar
    html_code = f"""
    <div class="stat-container">
        <div class="stat-header">
            <span style="color: #4CAF50;">{h_val}</span>
            <span style="color: #ccc; font-weight: normal;">{stat_name}</span>
            <span style="color: #FF5252;">{a_val}</span>
        </div>
        <div class="stat-bar-bg">
            <div class="bar-home" style="width: {h_pct}%;"></div>
            <div class="bar-away" style="width: {a_pct}%;"></div>
        </div>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. SAYFA YÖNETİMİ
# ---------------------------------------------------------
if 'page' not in st.session_state: st.session_state['page'] = 'dashboard'
if 'view_team' not in st.session_state: st.session_state['view_team'] = None
if 'view_match' not in st.session_state: st.session_state['view_match'] = None

st.sidebar.title("Menü")
mode = st.sidebar.radio("Mod Seç:", ["Sezon Simülasyonu", "Gerçek vs Yapay Zeka"])

def go_home(): st.session_state['page'] = 'dashboard'
def go_team(t): 
    st.session_state['view_team'] = t
    st.session_state['page'] = 'team_detail'
def go_match(m):
    st.session_state['view_match'] = m
    st.session_state['page'] = 'match_detail'

# =========================================================
# MOD 1: SEZON SİMÜLASYONU
# =========================================================
if mode == "Sezon Simülasyonu":
    
    # --- MAÇ DETAYI SAYFASI ---
    if st.session_state['page'] == 'match_detail':
        m = st.session_state['view_match']
        st.button("🔙 Takım Fikstürüne Dön", on_click=lambda: st.session_state.update({'page': 'team_detail'}))
        
        # Skorboard
        st.markdown(f"""
        <div style="text-align: center; background: #0e1117; padding: 25px; border: 1px solid #333; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <h1 style="color: white; margin:0; font-family: sans-serif;">
                <span style="color:#4CAF50">{m['Ev']}</span> 
                <span style="margin: 0 15px; font-size: 1.2em;">{m['HG']} - {m['AG']}</span> 
                <span style="color:#FF5252">{m['Dep']}</span>
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 1], gap="large")
        
        # 1. GOLCÜLER (TAKIM ADI BAŞLIKLI)
        with c1:
            st.markdown("### ⚽ Maç Özeti")
            st.markdown("---")
            
            gc1, gc2 = st.columns(2)
            
            # Ev Sahibi Golcüleri
            with gc1:
                st.markdown(f"<div class='team-header' style='color: #4CAF50;'>{m['Ev']}</div>", unsafe_allow_html=True)
                h_scorers = simulate_scorers(m['Ev'], m['HG'])
                if h_scorers:
                    for s in h_scorers: st.markdown(f"<div class='scorer-item'>{s}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='scorer-item' style='color:#666'>Gol Yok</div>", unsafe_allow_html=True)
            
            # Deplasman Golcüleri
            with gc2:
                st.markdown(f"<div class='team-header' style='color: #FF5252; text-align: right;'>{m['Dep']}</div>", unsafe_allow_html=True)
                a_scorers = simulate_scorers(m['Dep'], m['AG'])
                if a_scorers:
                    for s in a_scorers: st.markdown(f"<div class='scorer-item' style='text-align: right;'>{s}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='scorer-item' style='color:#666; text-align: right;'>Gol Yok</div>", unsafe_allow_html=True)

        # 2. İSTATİSTİKLER (GÜZEL ÇUBUKLAR)
        with c2:
            st.markdown("### 📊 Maç İstatistikleri")
            st.markdown("---")
            stats = generate_live_stats(m['Ev'], m['Dep'], m['HG'], m['AG'])
            
            for stat_name, values in stats.items():
                # values = (Ev_Değeri, Dep_Değeri)
                draw_stat_bar(stat_name, values[0], values[1])

    # --- TAKIM FİKSTÜRÜ ---
    elif st.session_state['page'] == 'team_detail':
        team = st.session_state['view_team']
        st.button("🔙 Puan Tablosuna Dön", on_click=go_home)
        st.header(f"📅 {team} Fikstürü")
        
        if 'sim_history' in st.session_state:
            hist = st.session_state['sim_history']
            team_matches = hist[(hist['Ev'] == team) | (hist['Dep'] == team)].reset_index(drop=True)
            
            event = st.dataframe(
                team_matches[['Ev', 'Skor', 'Dep']],
                on_select="rerun", selection_mode="single-row", 
                height=600, use_container_width=True, hide_index=True
            )
            
            if len(event.selection.rows) > 0:
                match_data = team_matches.iloc[event.selection.rows[0]]
                go_match(match_data)
                st.rerun()

    # --- DASHBOARD ---
    else:
        st.title("🏆 Premier Lig Simülasyonu (FIFA + AI)")
        st.caption("Takımların kadro kalitesi ve geçmiş performansına göre 38 haftalık lig simüle edilir.")
        
        if st.button("🚀 SEZONU OYNAT", type="primary"):
            table = {t: {'P':0, 'W':0, 'D':0, 'L':0, 'Pts':0, 'GF':0, 'GA':0, 'GD':0} for t in all_teams}
            history = []
            
            prog = st.progress(0)
            total_matches = len(all_teams) * (len(all_teams)-1)
            cnt = 0
            
            for h in all_teams:
                for a in all_teams:
                    if h == a: continue
                    
                    vec = np.concatenate([get_team_vector(h), get_team_vector(a)]).reshape(1, -1)
                    probs = model.predict_proba(vec)[0]
                    res = np.random.choice([0, 1, 2], p=probs)
                    
                    h_att = fifa_profiles.loc[h]['FIFA_Attack'] if h in fifa_profiles.index else 75
                    a_att = fifa_profiles.loc[a]['FIFA_Attack'] if a in fifa_profiles.index else 75
                    
                    hg = np.random.poisson((h_att/75) * 1.4)
                    ag = np.random.poisson((a_att/75) * 1.0)
                    
                    if res == 2 and hg <= ag: hg = ag + 1
                    elif res == 0 and ag <= hg: ag = hg + 1
                    elif res == 1: hg = ag = int((hg+ag)/2)
                    
                    cnt += 1
                    history.append({'Ev': h, 'Dep': a, 'HG': hg, 'AG': ag, 'Skor': f"{hg}-{ag}"})
                    
                    table[h]['P']+=1; table[a]['P']+=1
                    table[h]['GF']+=hg; table[a]['GF']+=ag
                    table[h]['GA']+=ag; table[a]['GA']+=hg
                    table[h]['GD']+=(hg-ag); table[a]['GD']+=(ag-hg)
                    
                    if res==2: table[h]['W']+=1; table[h]['Pts']+=3; table[a]['L']+=1
                    elif res==1: table[h]['D']+=1; table[h]['Pts']+=1; table[a]['D']+=1; table[a]['Pts']+=1
                    else: table[a]['W']+=1; table[a]['Pts']+=3; table[h]['L']+=1
                    
                    if cnt % 20 == 0: prog.progress(cnt/total_matches)
            
            prog.empty()
            df_table = pd.DataFrame.from_dict(table, orient='index').sort_values(by=['Pts', 'GD'], ascending=False)
            st.session_state['sim_table'] = df_table
            st.session_state['sim_history'] = pd.DataFrame(history)
            
        if 'sim_table' in st.session_state:
            st.subheader("Puan Durumu")
            st.info("💡 Fikstür ve detaylar için takıma tıklayın.")
            
            df_show = st.session_state['sim_table']
            event = st.dataframe(
                df_show.style.apply(lambda x: safe_style(x, df_show), axis=1),
                on_select="rerun", selection_mode="single-row", height=600
            )
            
            if len(event.selection.rows) > 0:
                selected_team = df_show.index[event.selection.rows[0]]
                go_team(selected_team)
                st.rerun()

# =========================================================
# MOD 2: GERÇEK vs YAPAY ZEKA
# =========================================================
elif mode == "Gerçek vs Yapay Zeka":
    st.title("🤖 Yapay Zeka vs Gerçek (23/24)")
    
    @st.cache_data
    def get_real_table_cached():
        tbl = {t: {'Pts':0} for t in all_teams}
        for _, row in real_df.iterrows():
            h, a = row['HomeTeam'], row['AwayTeam']
            if h in tbl and a in tbl:
                if row['FTHG'] > row['FTAG']: tbl[h]['Pts']+=3
                elif row['FTHG'] == row['FTAG']: tbl[h]['Pts']+=1; tbl[a]['Pts']+=1
                else: tbl[a]['Pts']+=3
        return pd.DataFrame.from_dict(tbl, orient='index').sort_values(by='Pts', ascending=False)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Gerçek Puan Durumu")
        real_table = get_real_table_cached()
        st.dataframe(real_table.style.apply(lambda x: safe_style(x, real_table), axis=1), height=700)
        
    with col2:
        st.subheader("🤖 Yapay Zeka Tahmini")
        if 'sim_table' in st.session_state:
            ai_table = st.session_state['sim_table'][['Pts']]
            st.dataframe(ai_table.style.apply(lambda x: safe_style(x, ai_table), axis=1), height=700)
        else:
            st.warning("Yapay zeka verisi yok. Önce 'Sezon Simülasyonu' modunda sezonu oynatın!")