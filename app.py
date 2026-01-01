import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import random

# ---------------------------------------------------------
# 1. SAYFA AYARLARI VE STİL
# ---------------------------------------------------------
st.set_page_config(page_title="PL AI Super Hub", layout="wide", page_icon="⚽")

st.markdown("""
<style>
    .stat-container { margin-bottom: 8px; }
    .stat-header { display: flex; justify-content: space-between; font-size: 13px; font-weight: bold; margin-bottom: 2px; }
    .stat-bar-bg { display: flex; height: 8px; background-color: #333; border-radius: 4px; overflow: hidden; }
    .bar-home { background-color: #4CAF50; height: 100%; }
    .bar-away { background-color: #FF5252; height: 100%; }
    .match-card { background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #333; }
    .score-text { font-size: 20px; font-weight: bold; color: white; padding: 0 15px; }
    .team-name { font-size: 16px; font-weight: 500; }
    .scorer-item { font-size: 12px; color: #ccc; margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)

# Güvenli Renklendirme
def safe_style(s, df):
    try:
        rank = df.index.get_loc(s.name)
        if rank == 0: return ['background-color: #ffd700; color: black'] * len(s) 
        elif rank < 4: return ['background-color: #e0f7fa; color: black'] * len(s) 
        elif rank >= len(df) - 3: return ['background-color: #ffcdd2; color: black'] * len(s) 
        else: return [''] * len(s)
    except: return [''] * len(s)

# ---------------------------------------------------------
# 2. VERİ YÜKLEME
# ---------------------------------------------------------
@st.cache_resource
def load_data():
    try:
        data = joblib.load('super_model.pkl')
        return (data['model'], data['performance_profiles'], data['fifa_profiles'], 
                data.get('team_rosters', {}), data['real_23_24_data'])
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
    if team_name in perf_profiles.index: perf = perf_profiles.loc[team_name].values
    else: perf = perf_profiles.mean().values 
    
    needed = ['FIFA_Overall', 'FIFA_Attack', 'FIFA_Midfield', 'FIFA_Defense', 'FIFA_Physical']
    if team_name in fifa_profiles.index: fifa = fifa_profiles.loc[team_name][needed].values
    else: fifa = fifa_profiles[needed].mean().values
    return np.concatenate([perf, fifa])

def simulate_scorers(team, goals):
    scorers = []
    if goals == 0: return []
    roster = team_rosters.get(team, [])
    if not roster:
        names = ["Forvet", "Ortasaha", "Sürpriz"]; weights = [0.6, 0.3, 0.1]
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
    # İstatistikleri oluşturup sayısal olarak döndürür
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

    return {
        'Topla Oynama (%)': (h_poss, a_poss),
        'Gol Beklentisi (xG)': (h_xg, a_xg),
        'Toplam Şut': (h_shots, a_shots),
        'İsabetli Şut': (h_sot, a_sot),
        'Pas Sayısı': (h_pass, a_pass),
        'Korner': (np.random.randint(2, 12), np.random.randint(2, 12)),
        'Faul': (np.random.randint(4, 15), np.random.randint(4, 15))
    }

def draw_stat_bar(stat_name, h_val, a_val):
    total = h_val + a_val
    if total == 0: h_pct, a_pct = 50, 50
    else:
        h_pct = (h_val / total) * 100
        a_pct = (a_val / total) * 100
    
    st.markdown(f"""
    <div class="stat-container">
        <div class="stat-header">
            <span style="color: #4CAF50;">{h_val}</span>
            <span style="color: #bbb;">{stat_name}</span>
            <span style="color: #FF5252;">{a_val}</span>
        </div>
        <div class="stat-bar-bg">
            <div class="bar-home" style="width: {h_pct}%;"></div>
            <div class="bar-away" style="width: {a_pct}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. SAYFA YÖNETİMİ
# ---------------------------------------------------------
if 'page' not in st.session_state: st.session_state['page'] = 'dashboard'
if 'view_team' not in st.session_state: st.session_state['view_team'] = None
if 'view_match' not in st.session_state: st.session_state['view_match'] = None

st.sidebar.title("Menü")
mode = st.sidebar.radio("Mod Seç:", ["Haftalık İlerleme", "Tüm Sezonu Simüle Et", "Gerçek vs Yapay Zeka"])

# Navigasyon
def go_home(): st.session_state['page'] = 'dashboard'
def go_team(t): 
    st.session_state['view_team'] = t
    st.session_state['page'] = 'team_detail'
def go_match(m):
    st.session_state['view_match'] = m
    st.session_state['page'] = 'match_detail'

# =========================================================
# ORTAK SAYFALAR (MAÇ DETAYI VE TAKIM FİKSTÜRÜ)
# =========================================================

# --- MAÇ DETAYI SAYFASI (HER İKİ MOD İÇİN ORTAK) ---
if st.session_state['page'] == 'match_detail':
    m = st.session_state['view_match']
    # Nereye döneceğini belirle
    back_target = 'team_detail' if st.session_state.get('last_page') == 'team_detail' else 'dashboard'
    st.button("🔙 Geri Dön", on_click=lambda: st.session_state.update({'page': back_target}))
    
    # Skorboard
    st.markdown(f"""
    <div style="text-align: center; background: #0e1117; padding: 20px; border: 1px solid #333; border-radius: 12px; margin-bottom: 25px;">
        <h1 style="color: white; margin:0;">
            <span style="color:#4CAF50">{m['Ev']}</span> 
            <span style="margin: 0 15px;">{m['HG']} - {m['AG']}</span> 
            <span style="color:#FF5252">{m['Dep']}</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 1], gap="large")
    
    # İstatistikler (Eğer kayıtlıysa kullan, yoksa üret)
    match_stats = m.get('Stats')
    if not match_stats:
        match_stats = generate_live_stats(m['Ev'], m['Dep'], m['HG'], m['AG'])

    # Golcüler (Eğer kayıtlıysa kullan, yoksa üret)
    h_scorers = m.get('Ev_Goller')
    if not h_scorers: h_scorers = simulate_scorers(m['Ev'], m['HG'])
    
    a_scorers = m.get('Dep_Goller')
    if not a_scorers: a_scorers = simulate_scorers(m['Dep'], m['AG'])

    with c1:
        st.subheader("📊 İstatistikler")
        for k, v in match_stats.items():
            draw_stat_bar(k, v[0], v[1])

    with c2:
        st.subheader("⚽ Maç Olayları")
        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown(f"**{m['Ev']}**")
            if h_scorers:
                for s in h_scorers: st.markdown(f"<div class='scorer-item'>{s}</div>", unsafe_allow_html=True)
            else: st.caption("-")
        with gc2:
            st.markdown(f"**{m['Dep']}**")
            if a_scorers:
                for s in a_scorers: st.markdown(f"<div class='scorer-item'>{s}</div>", unsafe_allow_html=True)
            else: st.caption("-")

# --- TAKIM FİKSTÜRÜ SAYFASI ---
elif st.session_state['page'] == 'team_detail':
    team = st.session_state['view_team']
    st.button("🔙 Puan Tablosuna Dön", on_click=go_home)
    st.header(f"📅 {team} Fikstürü")
    
    # Hangi modun geçmişine bakacağız?
    hist = pd.DataFrame()
    if mode == "Haftalık İlerleme" and 'weekly_history' in st.session_state:
        hist = pd.DataFrame(st.session_state['weekly_history'])
    elif mode == "Tüm Sezonu Simüle Et" and 'sim_history' in st.session_state:
        hist = st.session_state['sim_history']
    
    if not hist.empty:
        # Takımın maçlarını bul
        team_matches = hist[(hist['Ev'] == team) | (hist['Dep'] == team)].reset_index(drop=True)
        
        # Tıklanabilir tablo
        event = st.dataframe(
            team_matches[['Ev', 'Skor', 'Dep']],
            on_select="rerun", selection_mode="single-row", 
            height=600, use_container_width=True, hide_index=True
        )
        
        if len(event.selection.rows) > 0:
            match_data = team_matches.iloc[event.selection.rows[0]].to_dict()
            st.session_state['last_page'] = 'team_detail' # Geri dönüş için işaret
            go_match(match_data)
            st.rerun()
    else:
        st.warning("Henüz oynanmış maç yok.")

# =========================================================
# MOD: HAFTALIK İLERLEME (DASHBOARD)
# =========================================================
elif mode == "Haftalık İlerleme" and st.session_state['page'] == 'dashboard':
    st.title("📅 Haftalık Lig Simülasyonu")
    
    if 'weekly_fixture' not in st.session_state:
        st.session_state['weekly_fixture'] = None
        st.session_state['current_week'] = 0
        st.session_state['weekly_table'] = {t: {'P':0, 'W':0, 'D':0, 'L':0, 'Pts':0, 'GF':0, 'GA':0, 'GD':0} for t in all_teams}
        st.session_state['weekly_history'] = []

    if st.session_state['weekly_fixture'] is None:
        if st.button("🏁 Ligi Başlat", type="primary"):
            # Fikstür oluştur
            teams = list(all_teams)
            random.shuffle(teams)
            fixtures, return_matches = [], []
            if len(teams) % 2: teams.append('Bay')
            n = len(teams)
            for _ in range(n-1):
                week = []
                for i in range(n//2):
                    if teams[i] != 'Bay' and teams[n-1-i] != 'Bay': week.append((teams[i], teams[n-1-i]))
                fixtures.append(week)
                teams.insert(1, teams.pop())
            for w in fixtures: return_matches.append([(a, h) for h, a in w])
            
            st.session_state['weekly_fixture'] = fixtures + return_matches
            st.session_state['current_week'] = 0
            st.session_state['weekly_table'] = {t: {'P':0, 'W':0, 'D':0, 'L':0, 'Pts':0, 'GF':0, 'GA':0, 'GD':0} for t in all_teams}
            st.session_state['weekly_history'] = []
            st.rerun()
    else:
        # Kontrol Paneli
        fixtures = st.session_state['weekly_fixture']
        curr = st.session_state['current_week']
        
        c1, c2, c3 = st.columns([1, 2, 1])
        c1.metric("Hafta", f"{curr} / {len(fixtures)}")
        
        if curr < len(fixtures):
            if c2.button(f"⚽ {curr + 1}. Haftayı Oyna", type="primary", use_container_width=True):
                week_matches = fixtures[curr]
                results = []
                for h, a in week_matches:
                    vec = np.concatenate([get_team_vector(h), get_team_vector(a)]).reshape(1, -1)
                    probs = model.predict_proba(vec)[0]
                    res = np.random.choice([0,1,2], p=probs)
                    
                    h_att = fifa_profiles.loc[h]['FIFA_Attack'] if h in fifa_profiles.index else 75
                    a_att = fifa_profiles.loc[a]['FIFA_Attack'] if a in fifa_profiles.index else 75
                    hg = np.random.poisson((h_att/75)*1.4)
                    ag = np.random.poisson((a_att/75)*1.0)
                    
                    if res==2 and hg<=ag: hg=ag+1
                    elif res==0 and ag<=hg: ag=hg+1
                    elif res==1: hg=ag=int((hg+ag)/2)
                    
                    # Verileri Üret ve KAYDET
                    stats = generate_live_stats(h, a, hg, ag)
                    h_sc = simulate_scorers(h, hg)
                    a_sc = simulate_scorers(a, ag)
                    
                    # Tablo Güncelle
                    tbl = st.session_state['weekly_table']
                    tbl[h]['P']+=1; tbl[a]['P']+=1
                    tbl[h]['GF']+=hg; tbl[a]['GF']+=ag
                    tbl[h]['GA']+=ag; tbl[a]['GA']+=hg
                    tbl[h]['GD']+=(hg-ag); tbl[a]['GD']+=(ag-hg)
                    if res==2: tbl[h]['W']+=1; tbl[h]['Pts']+=3; tbl[a]['L']+=1
                    elif res==1: tbl[h]['D']+=1; tbl[h]['Pts']+=1; tbl[a]['D']+=1; tbl[a]['Pts']+=1
                    else: tbl[a]['W']+=1; tbl[a]['Pts']+=3; tbl[h]['L']+=1
                    
                    match_data = {
                        'Ev': h, 'Dep': a, 'HG': hg, 'AG': ag, 'Skor': f"{hg}-{ag}",
                        'Stats': stats, 'Ev_Goller': h_sc, 'Dep_Goller': a_sc
                    }
                    st.session_state['weekly_history'].append(match_data)
                    results.append(match_data)
                
                st.session_state['current_week'] += 1
                st.session_state['last_results'] = results
                st.rerun()
        else:
            c2.success("Sezon Bitti!")
            
        if c3.button("Sıfırla"):
            st.session_state['weekly_fixture'] = None
            st.rerun()
            
        # Gösterim
        col_res, col_tab = st.columns([4, 5])
        with col_res:
            if 'last_results' in st.session_state:
                st.subheader(f"{st.session_state['current_week']}. Hafta Sonuçları")
                for m in st.session_state['last_results']:
                    with st.container():
                        # Maç Kartı
                        st.markdown(f"""
                        <div class="match-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div style="width:40%; text-align:right;" class="team-name">{m['Ev']}</div>
                                <div class="score-text">{m['HG']} - {m['AG']}</div>
                                <div style="width:40%; text-align:left;" class="team-name">{m['Dep']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Detaylar Expander (Goller + İstatistikler)
                        with st.expander("Maç Detayları & İstatistikler"):
                            # Golcüler
                            if m['HG'] > 0 or m['AG'] > 0:
                                gc1, gc2 = st.columns(2)
                                with gc1: 
                                    for s in m['Ev_Goller']: st.caption(f"⚽ {s}")
                                with gc2:
                                    for s in m['Dep_Goller']: st.caption(f"⚽ {s}")
                                st.divider()
                            
                            # İstatistik Barları
                            for k, v in m['Stats'].items():
                                draw_stat_bar(k, v[0], v[1])

        with col_tab:
            st.subheader("Canlı Puan Durumu")
            df = pd.DataFrame.from_dict(st.session_state['weekly_table'], orient='index').sort_values(by=['Pts', 'GD'], ascending=False)
            
            event = st.dataframe(
                df.style.apply(lambda x: safe_style(x, df), axis=1),
                on_select="rerun", selection_mode="single-row", height=700
            )
            if len(event.selection.rows) > 0:
                team = df.index[event.selection.rows[0]]
                st.session_state['last_page'] = 'dashboard'
                go_team(team)
                st.rerun()

# =========================================================
# MOD: TÜM SEZONU SİMÜLE ET (DASHBOARD)
# =========================================================
elif mode == "Tüm Sezonu Simüle Et" and st.session_state['page'] == 'dashboard':
    st.title("⚡ Hızlı Simülasyon")
    
    if st.button("🚀 38 Haftayı Simüle Et", type="primary"):
        tbl = {t: {'P':0, 'W':0, 'D':0, 'L':0, 'Pts':0, 'GF':0, 'GA':0, 'GD':0} for t in all_teams}
        hist = []
        prog = st.progress(0)
        total = len(all_teams)*(len(all_teams)-1)
        cnt = 0
        
        for h in all_teams:
            for a in all_teams:
                if h == a: continue
                # (Tahmin Kodları Aynı...)
                vec = np.concatenate([get_team_vector(h), get_team_vector(a)]).reshape(1, -1)
                probs = model.predict_proba(vec)[0]; res = np.random.choice([0,1,2], p=probs)
                h_att = fifa_profiles.loc[h]['FIFA_Attack'] if h in fifa_profiles.index else 75
                a_att = fifa_profiles.loc[a]['FIFA_Attack'] if a in fifa_profiles.index else 75
                hg = np.random.poisson((h_att/75)*1.4); ag = np.random.poisson((a_att/75)*1.0)
                if res==2 and hg<=ag: hg=ag+1
                elif res==0 and ag<=hg: ag=hg+1
                elif res==1: hg=ag=int((hg+ag)/2)
                
                # Verileri Üret (Toplu simülasyon için de kaydet)
                stats = generate_live_stats(h, a, hg, ag)
                h_sc = simulate_scorers(h, hg)
                a_sc = simulate_scorers(a, ag)
                
                hist.append({
                    'Ev': h, 'Dep': a, 'HG': hg, 'AG': ag, 'Skor': f"{hg}-{ag}",
                    'Stats': stats, 'Ev_Goller': h_sc, 'Dep_Goller': a_sc
                })
                
                # Puan Tablosu
                tbl[h]['P']+=1; tbl[a]['P']+=1
                tbl[h]['GF']+=hg; tbl[a]['GF']+=ag; tbl[h]['GA']+=ag; tbl[a]['GA']+=hg
                tbl[h]['GD']+=(hg-ag); tbl[a]['GD']+=(ag-hg)
                if res==2: tbl[h]['W']+=1; tbl[h]['Pts']+=3; tbl[a]['L']+=1
                elif res==1: tbl[h]['D']+=1; tbl[h]['Pts']+=1; tbl[a]['D']+=1; tbl[a]['Pts']+=1
                else: tbl[a]['W']+=1; tbl[a]['Pts']+=3; tbl[h]['L']+=1
                
                cnt+=1; 
                if cnt%20==0: prog.progress(cnt/total)
        
        prog.empty()
        st.session_state['sim_table'] = pd.DataFrame.from_dict(tbl, orient='index').sort_values(by=['Pts', 'GD'], ascending=False)
        st.session_state['sim_history'] = pd.DataFrame(hist)

    if 'sim_table' in st.session_state:
        df = st.session_state['sim_table']
        st.subheader("Sezon Sonu Puan Durumu")
        event = st.dataframe(
            df.style.apply(lambda x: safe_style(x, df), axis=1),
            on_select="rerun", selection_mode="single-row", height=700
        )
        if len(event.selection.rows) > 0:
            team = df.index[event.selection.rows[0]]
            st.session_state['last_page'] = 'dashboard'
            go_team(team)
            st.rerun()

# =========================================================
# MOD: GERÇEK VS YAPAY ZEKA
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

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌍 Gerçek")
        rt = get_real_table_cached()
        st.dataframe(rt.style.apply(lambda x: safe_style(x, rt), axis=1), height=700)
    with c2:
        st.subheader("🤖 Yapay Zeka")
        if 'sim_table' in st.session_state:
            ai = st.session_state['sim_table'][['Pts']]
            st.dataframe(ai.style.apply(lambda x: safe_style(x, ai), axis=1), height=700)
        else:
            st.warning("Veri için 'Tüm Sezonu Simüle Et' modunu kullanın.")