import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. SAYFA AYARLARI VE STİL
# ---------------------------------------------------------
st.set_page_config(page_title="PL AI Super Hub", layout="wide", page_icon="⚽")

# Grafik Ayarları (Karanlık Mod)
plt.style.use('dark_background')

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
                .dashboard-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .pitch-container {
        background: linear-gradient(to bottom, #2e7d32, #1b5e20); /* Çim Rengi */
        border: 2px solid #fff;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        height: 400px;
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        position: relative;
    }
    .pitch-line {
        border-top: 1px solid rgba(255,255,255,0.3);
        width: 100%;
        position: absolute;
        top: 50%;
    }
    .player-dot {
        background-color: white;
        color: black;
        border-radius: 50%;
        width: 25px;
        height: 25px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        font-weight: bold;
        margin: 0 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .player-row {
        display: flex;
        justify-content: center;
        margin: 5px 0;
    }
    .small-table-header { font-size: 12px; color: #aaa; }
    .small-table-row { font-size: 13px; border-bottom: 1px solid #333; padding: 3px 0; }
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
    base_poss = 50
    if hg > ag: base_poss = 45
    elif ag > hg: base_poss = 55
    h_poss = np.random.randint(base_poss-5, base_poss+10); a_poss = 100 - h_poss
    
    h_shots = max(hg + np.random.randint(2, 8), int(hg * 2.5) + 5)
    a_shots = max(ag + np.random.randint(2, 8), int(ag * 2.5) + 5)
    h_sot = max(hg, int(h_shots * np.random.uniform(0.3, 0.6)))
    a_sot = max(ag, int(a_shots * np.random.uniform(0.3, 0.6)))
    h_xg = round(h_sot * 0.12 + (h_shots - h_sot) * 0.03 + (hg * 0.4), 2)
    a_xg = round(a_sot * 0.12 + (a_shots - a_sot) * 0.03 + (ag * 0.4), 2)
    h_pass = h_poss*4 + np.random.randint(50,100); a_pass = a_poss*4 + np.random.randint(50,100)

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
    else: h_pct = (h_val / total) * 100; a_pct = (a_val / total) * 100
    st.markdown(f"""
    <div class="stat-container">
        <div class="stat-header">
            <span style="color: #4CAF50;">{h_val}</span><span style="color: #bbb;">{stat_name}</span><span style="color: #FF5252;">{a_val}</span>
        </div>
        <div class="stat-bar-bg"><div class="bar-home" style="width: {h_pct}%;"></div><div class="bar-away" style="width: {a_pct}%;"></div></div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. GELİŞMİŞ ANALİZ FONKSİYONLARI (YENİ GRAFİKLER)
# ---------------------------------------------------------
def plot_title_race(history_df, all_teams):
    # Puanları hafta hafta hesapla
    points = {t: [0] for t in all_teams}
    # Her 10 maç = 1 Hafta (20 takım için)
    matches_per_week = len(all_teams) // 2
    
    current_week = 0
    match_counter = 0
    
    # Geçici puan tablosu
    temp_points = {t: 0 for t in all_teams}
    
    for _, row in history_df.iterrows():
        h, a = row['Ev'], row['Dep']
        hg, ag = row['HG'], row['AG']
        
        if hg > ag: temp_points[h] += 3
        elif ag > hg: temp_points[a] += 3
        else: temp_points[h] += 1; temp_points[a] += 1
        
        match_counter += 1
        if match_counter % matches_per_week == 0:
            current_week += 1
            for t in all_teams:
                points[t].append(temp_points[t])
                
    # DataFrame'e çevir
    race_df = pd.DataFrame(points)
    
    # Sadece ilk 6 takımı ve seçili takımları çiz (karmaşayı önlemek için)
    top_teams = sorted(temp_points, key=temp_points.get, reverse=True)[:6]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for team in top_teams:
        ax.plot(race_df[team], label=team, linewidth=2.5)
    
    ax.set_title("🏆 Şampiyonluk Yarışı (Haftalık Puan Değişimi)")
    ax.set_xlabel("Hafta")
    ax.set_ylabel("Puan")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)

def plot_attack_vs_defense(table_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=table_df, x='GF', y='GA', s=100, hue='Pts', palette='viridis', legend=False, ax=ax)
    
    # Ortalamalar
    avg_gf = table_df['GF'].mean()
    avg_ga = table_df['GA'].mean()
    
    ax.axvline(avg_gf, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(avg_ga, color='gray', linestyle='--', alpha=0.5)
    
    # Bölgeleri İsimlendir
    ax.text(table_df['GF'].max(), table_df['GA'].min(), "ŞAMPİYON ADAYLARI\n(Çok Atıp, Az Yiyen)", ha='right', va='bottom', color='#4CAF50', fontsize=9)
    ax.text(table_df['GF'].min(), table_df['GA'].max(), "KÜME DÜŞME ADAYLARI\n(Az Atıp, Çok Yiyen)", ha='left', va='top', color='#FF5252', fontsize=9)
    
    # Takım isimlerini yaz
    for i in range(table_df.shape[0]):
        ax.text(table_df.GF[i]+0.5, table_df.GA[i], table_df.index[i], fontsize=8, color='white')
        
    ax.set_title("🛡️ Hücum vs. Defans Performansı")
    ax.set_xlabel("Atılan Gol (GF)")
    ax.set_ylabel("Yenilen Gol (GA) - (Aşağısı Daha İyi)")
    ax.invert_yaxis() # Futbolda az gol yemek iyidir, ekseni ters çevir
    st.pyplot(fig)

def plot_wdl_distribution(table_df):
    # W-D-L verilerini al
    wdl_df = table_df[['W', 'D', 'L']].sort_values(by='W', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    wdl_df.plot(kind='barh', stacked=True, color=['#4CAF50', '#FFC107', '#FF5252'], ax=ax)
    
    ax.set_title("📊 Takım Karneleri (Galibiyet / Beraberlik / Mağlubiyet)")
    ax.set_xlabel("Maç Sayısı")
    ax.legend(["Galibiyet", "Beraberlik", "Mağlubiyet"])
    st.pyplot(fig)

def plot_top_scorers(history_df):
    all_goals = []
    for _, row in history_df.iterrows():
        if 'Ev_Goller' in row and isinstance(row['Ev_Goller'], list):
            for g in row['Ev_Goller']: all_goals.append({'Name': g.split('(')[0].replace('⚽', '').strip(), 'Team': row['Ev']})
        if 'Dep_Goller' in row and isinstance(row['Dep_Goller'], list):
            for g in row['Dep_Goller']: all_goals.append({'Name': g.split('(')[0].replace('⚽', '').strip(), 'Team': row['Dep']})
    
    if not all_goals: st.warning("Gol verisi yok."); return

    df_goals = pd.DataFrame(all_goals)
    top_scorers = df_goals['Name'].value_counts().head(10).reset_index()
    top_scorers.columns = ['Name', 'Goals']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Goals', y='Name', data=top_scorers, palette='magma', ax=ax)
    ax.set_title('🥇 Simülasyon Gol Krallığı')
    st.pyplot(fig)

def plot_feature_importance():
    feature_names = [
        'Home_Shots', 'Home_SOT', 'Home_Corners', 'Home_Possession',
        'Home_FIFA_Overall', 'Home_FIFA_Attack', 'Home_FIFA_Midfield', 'Home_FIFA_Defense', 'Home_FIFA_Physical',
        'Away_Shots', 'Away_SOT', 'Away_Corners', 'Away_Possession',
        'Away_FIFA_Overall', 'Away_FIFA_Attack', 'Away_FIFA_Midfield', 'Away_FIFA_Defense', 'Away_FIFA_Physical'
    ]
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis', ax=ax)
    ax.set_title('🧠 Yapay Zeka Neye Dikkat Etti?')
    st.pyplot(fig)

# ---------------------------------------------------------
# 5. SAYFA YÖNETİMİ
# ---------------------------------------------------------
if 'page' not in st.session_state: st.session_state['page'] = 'dashboard'
if 'view_team' not in st.session_state: st.session_state['view_team'] = None
if 'view_match' not in st.session_state: st.session_state['view_match'] = None

st.sidebar.title("Menü")
mode = st.sidebar.radio("Mod Seç:", ["Haftalık İlerleme", "Tüm Sezonu Simüle Et", "Gerçek vs Yapay Zeka"])

def go_home(): st.session_state['page'] = 'dashboard'
def go_team(t): st.session_state['view_team'] = t; st.session_state['page'] = 'team_detail'
def go_match(m): st.session_state['view_match'] = m; st.session_state['page'] = 'match_detail'

# =========================================================
# MAÇ DETAYI VE TAKIM FİKSTÜRÜ
# =========================================================
if st.session_state['page'] == 'match_detail':
    m = st.session_state['view_match']
    back_target = 'team_detail' if st.session_state.get('last_page') == 'team_detail' else 'dashboard'
    
    # Üst Bar: Geri Butonu ve Başlık
    c_back, c_title = st.columns([1, 5])
    c_back.button("🔙 Geri", on_click=lambda: st.session_state.update({'page': back_target}))
    c_title.markdown(f"## 🏟️ Maç Merkezi: {m['Ev']} vs {m['Dep']}")

    # --- ANA DÜZEN (3 SÜTUN) ---
    col1, col2, col3 = st.columns([1.2, 2, 1.2], gap="medium")

    # SOL KOLON: MİNİ LİG TABLOSU & FORM
    with col1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Ligdeki Durum")
        
        # Güncel tabloyu al
        current_table = pd.DataFrame()
        if 'weekly_table' in st.session_state:
            current_table = pd.DataFrame.from_dict(st.session_state['weekly_table'], orient='index').sort_values(by=['Pts', 'GD'], ascending=False)
        elif 'sim_table' in st.session_state:
            current_table = st.session_state['sim_table']
        
        if not current_table.empty:
            # Sadece bu iki takımı ve etrafındakileri gösterelim (veya ilk 5'i)
            st.markdown(f"<div class='small-table-header'>Takım &nbsp;&nbsp;&nbsp; P &nbsp;&nbsp; Av &nbsp;&nbsp; Pts</div>", unsafe_allow_html=True)
            
            # İlk 5 + Bizim takımlar
            display_teams = list(current_table.index[:4])
            if m['Ev'] not in display_teams: display_teams.append(m['Ev'])
            if m['Dep'] not in display_teams: display_teams.append(m['Dep'])
            
            # Tekrar sırala
            subset = current_table.loc[current_table.index.intersection(display_teams)]
            
            for team_name, row in subset.iterrows():
                color = "#4CAF50" if team_name == m['Ev'] else ("#FF5252" if team_name == m['Dep'] else "white")
                st.markdown(f"""
                <div class='small-table-row' style='color:{color}; display:flex; justify-content:space-between;'>
                    <span>{team_name}</span>
                    <span>{row['P']} | {row['GD']} | <b>{row['Pts']}</b></span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Lig tablosu henüz oluşmadı.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Golcüler Kutusu
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("⚽ Goller")
        
        h_sc = m.get('Ev_Goller', [])
        a_sc = m.get('Dep_Goller', [])
        
        st.markdown(f"**{m['Ev']}**")
        if h_sc:
            for s in h_sc: st.caption(s)
        else: st.caption("-")
        
        st.divider()
        
        st.markdown(f"**{m['Dep']}**")
        if a_sc:
            for s in a_sc: st.caption(s)
        else: st.caption("-")
        st.markdown('</div>', unsafe_allow_html=True)

    # ORTA KOLON: SAHA GÖRÜNÜMÜ & SKOR
    with col2:
        # Skor Kartı
        st.markdown(f"""
        <div style="text-align: center; background: #000; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #444;">
            <h1 style="color: white; margin:0; font-size: 36px;">
                <span style="color:#4CAF50">{m['HG']}</span> 
                <span style="color:#888; font-size: 20px;">vs</span> 
                <span style="color:#FF5252">{m['AG']}</span>
            </h1>
            <p style="color:#aaa; margin:0;">Maç Sonucu</p>
        </div>
        """, unsafe_allow_html=True)

        # SAHA SİMÜLASYONU
        # Takımların kadrolarını alıp pozisyona göre dizer
        def get_formation(team_name):
            roster = team_rosters.get(team_name, [])
            # Pozisyonlara göre grupla (Basit 4-4-2 mantığı gibi gösterelim)
            fw = [p['Name'].split()[-1] for p in roster if p['Position'] == 'FW'][:2]
            mf = [p['Name'].split()[-1] for p in roster if p['Position'] == 'MF'][:4]
            df = [p['Name'].split()[-1] for p in roster if p['Position'] in ['DF', 'CB', 'LB', 'RB']][:4]
            gk = [p['Name'].split()[-1] for p in roster if p['Position'] == 'GK'][:1]
            if not gk: gk = ["GK"]
            if not fw: fw = ["FW1", "FW2"]
            return gk, df, mf, fw

        ev_gk, ev_df, ev_mf, ev_fw = get_formation(m['Ev'])
        dep_gk, dep_df, dep_mf, dep_fw = get_formation(m['Dep'])

        # Sahayı Çiz (HTML/CSS)
        st.markdown(f"""
        <div class="pitch-container">
            <div class="pitch-line"></div>
            
            <!-- DEPLASMAN (Üstte) -->
            <div style="color: #FF5252; font-weight:bold; margin-bottom:5px;">{m['Dep']} (4-4-2)</div>
            <div class="player-row">{ "".join([f"<div class='player-dot' title='{x}'>{x[0]}</div>" for x in dep_gk]) }</div>
            <div class="player-row">{ "".join([f"<div class='player-dot' title='{x}'>{x[0]}</div>" for x in dep_df]) }</div>
            <div class="player-row">{ "".join([f"<div class='player-dot' title='{x}'>{x[0]}</div>" for x in dep_mf]) }</div>
            <div class="player-row">{ "".join([f"<div class='player-dot' title='{x}'>{x[0]}</div>" for x in dep_fw]) }</div>
            
            <div style="height: 20px;"></div> <!-- Orta Saha Boşluğu -->
            
            <!-- EV SAHİBİ (Altta) -->
            <div class="player-row">{ "".join([f"<div class='player-dot' title='{x}'>{x[0]}</div>" for x in ev_fw]) }</div>
            <div class="player-row">{ "".join([f"<div class='player-dot' title='{x}'>{x[0]}</div>" for x in ev_mf]) }</div>
            <div class="player-row">{ "".join([f"<div class='player-dot' title='{x}'>{x[0]}</div>" for x in ev_df]) }</div>
            <div class="player-row">{ "".join([f"<div class='player-dot' title='{x}'>{x[0]}</div>" for x in ev_gk]) }</div>
            <div style="color: #4CAF50; font-weight:bold; margin-top:5px;">{m['Ev']} (4-4-2)</div>
        </div>
        """, unsafe_allow_html=True)

    # SAĞ KOLON: İSTATİSTİKLER & GRAFİKLER
    with col3:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("📊 Maç İstatistikleri")
        
        # Eğer istatistik yoksa üret
        stats = m.get('Stats', generate_live_stats(m['Ev'], m['Dep'], m['HG'], m['AG']))
        
        for k, v in stats.items():
            draw_stat_bar(k, v[0], v[1])
        st.markdown('</div>', unsafe_allow_html=True)

        # Kazanma İhtimali Grafiği (Basit Pie Chart)
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.write("📈 **Olasılık Dağılımı**")
        
        # Modelden olasılıkları tekrar çekmemiz lazım (Anlık görsel için)
        vec = np.concatenate([get_team_vector(m['Ev']), get_team_vector(m['Dep'])]).reshape(1, -1)
        probs = model.predict_proba(vec)[0] # [Dep, Ber, Ev]
        
        fig_pie, ax_pie = plt.subplots(figsize=(3, 3))
        ax_pie.pie(
            [probs[2], probs[1], probs[0]], 
            labels=[m['Ev'], 'X', m['Dep']],
            colors=['#4CAF50', '#888', '#FF5252'],
            autopct='%1.0f%%',
            textprops={'color':"white", 'fontsize': 8}
        )
        fig_pie.patch.set_alpha(0) # Şeffaf arka plan
        st.pyplot(fig_pie)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# MOD: HAFTALIK İLERLEME
# =========================================================
elif mode == "Haftalık İlerleme" and st.session_state['page'] == 'dashboard':
    st.title("📅 Haftalık Lig Simülasyonu")
    
    if 'weekly_fixture' not in st.session_state:
        st.session_state['weekly_fixture'] = None; st.session_state['current_week'] = 0
        st.session_state['weekly_table'] = {t: {'P':0, 'W':0, 'D':0, 'L':0, 'Pts':0, 'GF':0, 'GA':0, 'GD':0} for t in all_teams}
        st.session_state['weekly_history'] = []

    if st.session_state['weekly_fixture'] is None:
        if st.button("🏁 Ligi Başlat", type="primary"):
            teams = list(all_teams); random.shuffle(teams)
            fixtures, return_matches = [], []
            if len(teams)%2: teams.append('Bay')
            n = len(teams)
            for _ in range(n-1):
                week = []
                for i in range(n//2):
                    if teams[i] != 'Bay' and teams[n-1-i] != 'Bay': week.append((teams[i], teams[n-1-i]))
                fixtures.append(week); teams.insert(1, teams.pop())
            for w in fixtures: return_matches.append([(a, h) for h, a in w])
            st.session_state['weekly_fixture'] = fixtures + return_matches
            st.session_state['current_week'] = 0
            st.session_state['weekly_table'] = {t: {'P':0, 'W':0, 'D':0, 'L':0, 'Pts':0, 'GF':0, 'GA':0, 'GD':0} for t in all_teams}
            st.session_state['weekly_history'] = []
            st.rerun()
    else:
        fixtures = st.session_state['weekly_fixture']; curr = st.session_state['current_week']
        c1, c2, c3 = st.columns([1, 2, 1])
        c1.metric("Hafta", f"{curr} / {len(fixtures)}")
        if curr < len(fixtures):
            if c2.button(f"⚽ {curr + 1}. Haftayı Oyna", type="primary", use_container_width=True):
                week_matches = fixtures[curr]; results = []
                for h, a in week_matches:
                    vec = np.concatenate([get_team_vector(h), get_team_vector(a)]).reshape(1, -1)
                    probs = model.predict_proba(vec)[0]; res = np.random.choice([0,1,2], p=probs)
                    h_att = fifa_profiles.loc[h]['FIFA_Attack'] if h in fifa_profiles.index else 75
                    a_att = fifa_profiles.loc[a]['FIFA_Attack'] if a in fifa_profiles.index else 75
                    hg = np.random.poisson((h_att/75)*1.4); ag = np.random.poisson((a_att/75)*1.0)
                    if res==2 and hg<=ag: hg=ag+1
                    elif res==0 and ag<=hg: ag=hg+1
                    elif res==1: hg=ag=int((hg+ag)/2)
                    
                    stats = generate_live_stats(h, a, hg, ag); h_sc = simulate_scorers(h, hg); a_sc = simulate_scorers(a, ag)
                    tbl = st.session_state['weekly_table']
                    tbl[h]['P']+=1; tbl[a]['P']+=1; tbl[h]['GF']+=hg; tbl[a]['GF']+=ag; tbl[h]['GA']+=ag; tbl[a]['GA']+=hg; tbl[h]['GD']+=(hg-ag); tbl[a]['GD']+=(ag-hg)
                    if res==2: tbl[h]['W']+=1; tbl[h]['Pts']+=3; tbl[a]['L']+=1
                    elif res==1: tbl[h]['D']+=1; tbl[h]['Pts']+=1; tbl[a]['D']+=1; tbl[a]['Pts']+=1
                    else: tbl[a]['W']+=1; tbl[a]['Pts']+=3; tbl[h]['L']+=1
                    st.session_state['weekly_history'].append({'Ev': h, 'Dep': a, 'HG': hg, 'AG': ag, 'Skor': f"{hg}-{ag}", 'Stats': stats, 'Ev_Goller': h_sc, 'Dep_Goller': a_sc})
                    results.append({'Ev': h, 'Dep': a, 'HG': hg, 'AG': ag, 'Skor': f"{hg}-{ag}", 'Stats': stats, 'Ev_Goller': h_sc, 'Dep_Goller': a_sc})
                st.session_state['current_week'] += 1; st.session_state['last_results'] = results; st.rerun()
        else:
            c2.success("Sezon Bitti!")
            if st.button("📊 DETAYLI SEZON RAPORU", type="primary"): st.session_state['show_analysis'] = True
        
        if c3.button("Sıfırla"): st.session_state['weekly_fixture'] = None; st.session_state['show_analysis'] = False; st.rerun()
        
        # --- ANALİZ PANELİ (YENİ) ---
        if st.session_state.get('show_analysis'):
            st.divider(); st.title("📈 Sezon Sonu Analizi")
            # Sekmeler: Gol Krallığı | Şampiyonluk Yarışı | Hücum vs Defans | Galibiyet Karnesi | Model Beyni
            t1, t2, t3, t4, t5 = st.tabs(["Gol Krallığı", "🏆 Şampiyonluk Yarışı", "🛡️ Hücum vs Defans", "📊 Galibiyet Karnesi", "🧠 Yapay Zeka"])
            
            sim_df = pd.DataFrame.from_dict(st.session_state['weekly_table'], orient='index')
            hist_df = pd.DataFrame(st.session_state['weekly_history'])
            
            with t1: plot_top_scorers(hist_df)
            with t2: plot_title_race(hist_df, all_teams)
            with t3: plot_attack_vs_defense(sim_df)
            with t4: plot_wdl_distribution(sim_df)
            with t5: plot_feature_importance()
            st.divider()

        # Maçlar ve Tablo
        col_res, col_tab = st.columns([4, 5])
        with col_res:
            if 'last_results' in st.session_state:
                st.subheader(f"{st.session_state['current_week']}. Hafta Sonuçları")
                for m in st.session_state['last_results']:
                    with st.expander(f"{m['Ev']} {m['Skor']} {m['Dep']}"):
                        for k, v in m['Stats'].items(): draw_stat_bar(k, v[0], v[1])
                        c_h, c_a = st.columns(2)
                        with c_h: 
                            for s in m['Ev_Goller']: st.caption(s)
                        with c_a: 
                            for s in m['Dep_Goller']: st.caption(s)
        with col_tab:
            st.subheader("Canlı Puan Durumu")
            df = pd.DataFrame.from_dict(st.session_state['weekly_table'], orient='index').sort_values(by=['Pts', 'GD'], ascending=False)
            event = st.dataframe(df.style.apply(lambda x: safe_style(x, df), axis=1), on_select="rerun", selection_mode="single-row", height=700)
            if len(event.selection.rows) > 0:
                team = df.index[event.selection.rows[0]]; st.session_state['last_page'] = 'dashboard'; go_team(team); st.rerun()

# =========================================================
# MOD: TÜM SEZONU SİMÜLE ET (DASHBOARD)
# =========================================================
elif mode == "Tüm Sezonu Simüle Et" and st.session_state['page'] == 'dashboard':
    st.title("⚡ Hızlı Simülasyon")
    if st.button("🚀 38 Haftayı Simüle Et", type="primary"):
        tbl = {t: {'P':0, 'W':0, 'D':0, 'L':0, 'Pts':0, 'GF':0, 'GA':0, 'GD':0} for t in all_teams}
        hist = []; prog = st.progress(0); total = len(all_teams)*(len(all_teams)-1); cnt = 0
        for h in all_teams:
            for a in all_teams:
                if h == a: continue
                vec = np.concatenate([get_team_vector(h), get_team_vector(a)]).reshape(1, -1)
                probs = model.predict_proba(vec)[0]; res = np.random.choice([0,1,2], p=probs)
                h_att = fifa_profiles.loc[h]['FIFA_Attack'] if h in fifa_profiles.index else 75
                a_att = fifa_profiles.loc[a]['FIFA_Attack'] if a in fifa_profiles.index else 75
                hg = np.random.poisson((h_att/75)*1.4); ag = np.random.poisson((a_att/75)*1.0)
                if res==2 and hg<=ag: hg=ag+1
                elif res==0 and ag<=hg: ag=hg+1
                elif res==1: hg=ag=int((hg+ag)/2)
                stats = generate_live_stats(h, a, hg, ag); h_sc = simulate_scorers(h, hg); a_sc = simulate_scorers(a, ag)
                hist.append({'Ev': h, 'Dep': a, 'HG': hg, 'AG': ag, 'Skor': f"{hg}-{ag}", 'Stats': stats, 'Ev_Goller': h_sc, 'Dep_Goller': a_sc})
                tbl[h]['P']+=1; tbl[a]['P']+=1; tbl[h]['GF']+=hg; tbl[a]['GF']+=ag; tbl[h]['GA']+=ag; tbl[a]['GA']+=hg; tbl[h]['GD']+=(hg-ag); tbl[a]['GD']+=(ag-hg)
                if res==2: tbl[h]['W']+=1; tbl[h]['Pts']+=3; tbl[a]['L']+=1
                elif res==1: tbl[h]['D']+=1; tbl[h]['Pts']+=1; tbl[a]['D']+=1; tbl[a]['Pts']+=1
                else: tbl[a]['W']+=1; tbl[a]['Pts']+=3; tbl[h]['L']+=1
                cnt+=1; 
                if cnt%20==0: prog.progress(cnt/total)
        prog.empty()
        st.session_state['sim_table'] = pd.DataFrame.from_dict(tbl, orient='index').sort_values(by=['Pts', 'GD'], ascending=False)
        st.session_state['sim_history'] = pd.DataFrame(hist)
        st.session_state['sim_done'] = True

    if st.session_state.get('sim_done'):
        # --- GELİŞMİŞ RAPOR ---
        if st.button("📊 DETAYLI SEZON RAPORU", type="primary"):
            st.divider(); st.title("📈 Sezon Sonu Analizi")
            t1, t2, t3, t4, t5 = st.tabs(["Gol Krallığı", "🏆 Şampiyonluk Yarışı", "🛡️ Hücum vs Defans", "📊 Galibiyet Karnesi", "🧠 Yapay Zeka"])
            
            sim_df = st.session_state['sim_table']
            hist_df = st.session_state['sim_history']
            
            with t1: plot_top_scorers(hist_df)
            with t2: plot_title_race(hist_df, all_teams)
            with t3: plot_attack_vs_defense(sim_df)
            with t4: plot_wdl_distribution(sim_df)
            with t5: plot_feature_importance()
            st.divider()

    if 'sim_table' in st.session_state:
        df = st.session_state['sim_table']
        st.subheader("Sezon Sonu Puan Durumu")
        event = st.dataframe(df.style.apply(lambda x: safe_style(x, df), axis=1), on_select="rerun", selection_mode="single-row", height=700)
        if len(event.selection.rows) > 0:
            team = df.index[event.selection.rows[0]]; st.session_state['last_page'] = 'dashboard'; go_team(team); st.rerun()

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
        st.subheader("🌍 Gerçek"); rt = get_real_table_cached()
        st.dataframe(rt.style.apply(lambda x: safe_style(x, rt), axis=1), height=700)
    with c2:
        st.subheader("🤖 Yapay Zeka")
        if 'sim_table' in st.session_state: ai = st.session_state['sim_table'][['Pts']]; st.dataframe(ai.style.apply(lambda x: safe_style(x, ai), axis=1), height=700)
        else: st.warning("Veri için 'Tüm Sezonu Simüle Et' modunu kullanın.")