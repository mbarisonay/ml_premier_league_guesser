import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ---------------------------------------------------------
# 1. AYARLAR VE VERİ YÜKLEME
# ---------------------------------------------------------
st.set_page_config(page_title="PL Simülasyonu", layout="wide")

@st.cache_resource
def load_data():
    data = joblib.load('premier_league_sim_data.pkl')
    return data['model'], data['team_profiles'], data['feature_names']

try:
    model, team_profiles, feature_names = load_data()
except FileNotFoundError:
    st.error("Lütfen önce 'model_egitimi.py' dosyasını çalıştırın!")
    st.stop()

# ---------------------------------------------------------
# 2. TAKIM LİSTESİ
# ---------------------------------------------------------
target_teams = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", 
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham", 
    "Liverpool", "Manchester City", "Manchester United", "Newcastle United", 
    "Nottingham Forest", "Sheffield United", "Tottenham Hotspur", 
    "West Ham United", "Wolverhampton Wanderers", "Leicester City"
]
team_list = [t for t in target_teams if t in team_profiles.index]

if len(team_list) < 20:
    st.warning(f"Dikkat: {len(team_list)} takım yüklendi.")

# ---------------------------------------------------------
# 3. İSTATİSTİK VE SKOR ÜRETME MOTORU
# ---------------------------------------------------------
def generate_match_stats_on_click(home, away, hg, ag):
    """
    Maça tıklandığında, skora ve takım güçlerine uygun detaylı istatistikler üretir.
    """
    # Takımların geçmiş ortalamalarını çek
    h_prof = team_profiles.loc[home]
    a_prof = team_profiles.loc[away]
    
    # 1. Topla Oynama (Possession)
    # Ortalamalara göre bir oran belirle ve rastgelelik kat
    total_poss = h_prof.get('Possession', 50) + a_prof.get('Possession', 50)
    h_ratio = h_prof.get('Possession', 50) / total_poss
    
    # Kazanan takım genelde topa biraz daha hakim olabilir (veya skor koruyabilir)
    # Burada basitçe ortalamayı baz alıp +/- %10 varyasyon ekliyoruz
    h_poss = int(h_ratio * 100) + np.random.randint(-5, 6)
    
    # Sınırlandırma (min %30, max %70 gibi)
    h_poss = max(25, min(75, h_poss))
    a_poss = 100 - h_poss
    
    # 2. Şutlar ve İsabetli Şutlar (Shots & SOT)
    # İsabetli şut, en az atılan gol kadar olmalı!
    h_sot = max(hg, int(np.random.poisson(h_prof.get('SOT', 4)) * (0.8 + 0.4 * (h_poss/50))))
    a_sot = max(ag, int(np.random.poisson(a_prof.get('SOT', 4)) * (0.8 + 0.4 * (a_poss/50))))
    
    # Toplam şut, isabetli şuttan fazla olmalı
    h_shots = int(h_sot + np.random.poisson(h_prof.get('Shots', 10) - h_prof.get('SOT', 4)))
    a_shots = int(a_sot + np.random.poisson(a_prof.get('Shots', 10) - a_prof.get('SOT', 4)))
    
    # 3. Paslar
    # Topla oynama ile doğru orantılı
    h_passes = int(h_poss * 5 + np.random.randint(50, 150))
    a_passes = int(a_poss * 5 + np.random.randint(50, 150))
    
    h_pass_acc = int(h_prof.get('PassAccuracy', 80) + np.random.randint(-5, 5))
    a_pass_acc = int(a_prof.get('PassAccuracy', 80) + np.random.randint(-5, 5))
    
    # 4. Kornerler ve Fauller
    h_corners = np.random.poisson(h_prof.get('Corners', 5))
    a_corners = np.random.poisson(a_prof.get('Corners', 5))
    
    h_fouls = np.random.poisson(h_prof.get('Fouls', 10))
    a_fouls = np.random.poisson(a_prof.get('Fouls', 10))
    
    h_cards = np.random.poisson(h_prof.get('YellowCards', 1.5))
    a_cards = np.random.poisson(a_prof.get('YellowCards', 1.5))

    return {
        'Topla Oynama': (f"%{h_poss}", f"%{a_poss}", h_poss, a_poss),
        'Toplam Şut': (h_shots, a_shots, h_shots, a_shots),
        'İsabetli Şut': (h_sot, a_sot, h_sot, a_sot),
        'Pas Sayısı': (h_passes, a_passes, h_passes, a_passes),
        'Pas İsabeti': (f"%{h_pass_acc}", f"%{a_pass_acc}", h_pass_acc, a_pass_acc),
        'Korner': (h_corners, a_corners, h_corners, a_corners),
        'Faul': (h_fouls, a_fouls, h_fouls, a_fouls),
        'Sarı Kart': (h_cards, a_cards, h_cards, a_cards)
    }

def generate_score(home_team, away_team, outcome):
    try:
        h_att = team_profiles.loc[home_team]['SOT'] if 'SOT' in team_profiles.columns else 4.0
        a_att = team_profiles.loc[away_team]['SOT'] if 'SOT' in team_profiles.columns else 3.0
    except:
        h_att, a_att = 1.2, 1.0

    h_goals = np.random.poisson(h_att / 2.5)
    a_goals = np.random.poisson(a_att / 2.5)

    if outcome == 1: # Beraberlik
        goals = int((h_goals + a_goals) / 2)
        h_goals, a_goals = goals, goals
    elif outcome == 2: # Ev Sahibi
        if h_goals <= a_goals: h_goals = a_goals + 1
    else: # Deplasman
        if a_goals <= h_goals: a_goals = h_goals + 1

    return h_goals, a_goals

def simulate_match(home_team, away_team):
    h_dna = team_profiles.loc[home_team].values
    a_dna = team_profiles.loc[away_team].values
    input_vector = np.concatenate([h_dna, a_dna]).reshape(1, -1)
    
    probs = model.predict_proba(input_vector)[0]
    result_code = np.random.choice([0, 1, 2], p=probs)
    hg, ag = generate_score(home_team, away_team, result_code)
    
    return result_code, hg, ag

def run_season_simulation():
    table = {team: {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'Pts': 0, 'GF': 0, 'GA': 0, 'GD': 0} for team in team_list}
    match_history = [] 
    
    match_count = 0
    total_matches = len(team_list) * (len(team_list) - 1)
    progress_bar = st.progress(0)
    
    for home in team_list:
        for away in team_list:
            if home == away:
                continue
            
            res_code, hg, ag = simulate_match(home, away)
            match_count += 1
            score_text = f"{hg} - {ag}"
            
            # Benzersiz ID oluştur (Tıklama için)
            match_id = f"{home}_{away}_{time.time()}"
            
            match_history.append({
                'match_id': match_id,
                'Ev Sahibi': home,
                'Skor': score_text,
                'Deplasman': away,
                'Sonuç Kodu': res_code,
                'HG': hg,
                'AG': ag
            })
            
            table[home]['P'] += 1; table[away]['P'] += 1
            table[home]['GF'] += hg; table[away]['GF'] += ag
            table[home]['GA'] += ag; table[away]['GA'] += hg
            table[home]['GD'] += (hg - ag); table[away]['GD'] += (ag - hg)
            
            if res_code == 2:
                table[home]['W'] += 1; table[home]['Pts'] += 3; table[away]['L'] += 1
            elif res_code == 1:
                table[home]['D'] += 1; table[home]['Pts'] += 1; table[away]['D'] += 1; table[away]['Pts'] += 1
            else:
                table[away]['W'] += 1; table[away]['Pts'] += 3; table[home]['L'] += 1
                
        progress_bar.progress(match_count / total_matches)
    
    time.sleep(0.5); progress_bar.empty()
    df_table = pd.DataFrame.from_dict(table, orient='index')
    df_table = df_table.sort_values(by=['Pts', 'GD', 'W'], ascending=False)
    
    return df_table, pd.DataFrame(match_history)

# ---------------------------------------------------------
# 4. SAYFA YÖNETİMİ
# ---------------------------------------------------------

# State Başlatma
if 'page' not in st.session_state: st.session_state['page'] = 'home'
if 'selected_team' not in st.session_state: st.session_state['selected_team'] = None
if 'selected_match' not in st.session_state: st.session_state['selected_match'] = None

def go_home():
    st.session_state['page'] = 'home'
    st.session_state['selected_team'] = None
    st.session_state['selected_match'] = None

def go_team_view():
    st.session_state['page'] = 'team_view'
    st.session_state['selected_match'] = None

# --- SAYFA 3: MAÇ DETAYI ---
if st.session_state['page'] == 'match_view':
    match_data = st.session_state['selected_match']
    home = match_data['Ev Sahibi']
    away = match_data['Deplasman']
    hg = match_data['HG']
    ag = match_data['AG']
    
    st.button("← Fikstüre Dön", on_click=go_team_view)
    
    # 1. Skorboard
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: #1e1e1e; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #ccc;">MAÇ SONUCU</h2>
        <h1 style="font-size: 60px; margin: 0;">
            <span style="color: #4CAF50;">{home}</span> 
            <span style="color: white; margin: 0 20px;">{hg} - {ag}</span> 
            <span style="color: #FF5252;">{away}</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. İstatistikleri Üret
    stats = generate_match_stats_on_click(home, away, hg, ag)
    
    st.subheader("📊 Maç İstatistikleri")
    
    # İstatistikleri görselleştir
    for stat_name, values in stats.items():
        val_home_str, val_away_str, val_home_num, val_away_num = values
        
        # Sütun yapısı: [Ev Değeri] [Progress Bar] [Stat İsmi] [Progress Bar] [Deplasman Değeri]
        c1, c2, c3, c4, c5 = st.columns([1, 3, 2, 3, 1])
        
        with c1: st.markdown(f"<h4 style='text-align: right; color:#4CAF50'>{val_home_str}</h4>", unsafe_allow_html=True)
        with c5: st.markdown(f"<h4 style='text-align: left; color:#FF5252'>{val_away_str}</h4>", unsafe_allow_html=True)
        with c3: st.markdown(f"<p style='text-align: center; font-weight:bold; margin-top:10px'>{stat_name}</p>", unsafe_allow_html=True)
        
        # Progress Barlar (Toplam üzerinden oranla)
        total = val_home_num + val_away_num
        if total == 0: total = 1
        
        with c2: 
            st.progress(val_home_num / total if val_home_num <= total else 1.0)
        with c4: 
            # Streamlit progress bar soldan sağa dolar, deplasman için ters mantık görselleştirmek zor
            # O yüzden sadece value bar koyuyoruz
            st.progress(val_away_num / total if val_away_num <= total else 1.0)
            
    st.divider()
    st.info("Bu istatistikler, takımların sezonluk ortalamaları ve maç skoruna göre yapay zeka tarafından simüle edilmiştir.")

# --- SAYFA 2: TAKIM FİKSTÜRÜ ---
elif st.session_state['page'] == 'team_view':
    selected_team = st.session_state['selected_team']
    
    st.button("← Puan Durumuna Dön", on_click=go_home)
    st.title(f"📅 {selected_team} - Fikstürü")
    st.caption("Maçın detaylarını görmek için tablodaki maça tıklayın.")
    
    history_df = st.session_state['match_history']
    
    # Takımın maçlarını filtrele
    team_matches = history_df[
        (history_df['Ev Sahibi'] == selected_team) | 
        (history_df['Deplasman'] == selected_team)
    ].reset_index(drop=True) # Reset index önemli, tıklama için
    
    # Tabloyu hazırla
    display_df = team_matches[['Ev Sahibi', 'Skor', 'Deplasman']].copy()
    
    # Renklendirme mantığı (Görsel amaçlı ayrı sütun ekleyelim)
    def color_results(row):
        # Bu fonksiyon st.dataframe style için değil, satırı analiz etmek için
        return [''] * len(row)

    # Tıklanabilir Tablo
    event = st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True
    )
    
    if len(event.selection.rows) > 0:
        selected_idx = event.selection.rows[0]
        # Orijinal veriden o satırı çek
        match_data = team_matches.iloc[selected_idx]
        
        # Maç detay sayfasına git
        st.session_state['selected_match'] = match_data
        st.session_state['page'] = 'match_view'
        st.rerun()

# --- SAYFA 1: ANA SAYFA ---
else:
    st.title("⚽ Premier Lig Yapay Zeka Simülasyonu")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Kontrol Paneli")
        
        if st.button("🏆 SEZONU SİMÜLE ET", type="primary"):
            with st.spinner('Maçlar oynanıyor...'):
                final_table, match_log = run_season_simulation()
                st.session_state['result'] = final_table
                st.session_state['match_history'] = match_log
                st.success("Sezon Tamamlandı!")
        
        if 'result' in st.session_state:
            st.info("💡 Tabloda bir takıma tıklayarak fikstürünü gör. Fikstürde maça tıklayarak istatistikleri gör.")

    with col2:
        if 'result' in st.session_state:
            st.subheader("📊 Puan Durumu")
            
            def highlight_rows(s):
                current_rank = st.session_state['result'].index.get_loc(s.name)
                if current_rank == 0: return ['background-color: #ffd700; color: black'] * len(s)
                elif current_rank < 4: return ['background-color: #e0f7fa; color: black'] * len(s)
                elif current_rank >= len(team_list) - 3: return ['background-color: #ffcdd2; color: black'] * len(s)
                else: return [''] * len(s)

            event = st.dataframe(
                st.session_state['result'].style.apply(highlight_rows, axis=1),
                use_container_width=True,
                height=800,
                on_select="rerun",
                selection_mode="single-row"
            )
            
            if len(event.selection.rows) > 0:
                selected_row_index = event.selection.rows[0]
                team_name = st.session_state['result'].index[selected_row_index]
                st.session_state['selected_team'] = team_name
                st.session_state['page'] = 'team_view'
                st.rerun()
        else:
            st.info("👈 Sol taraftaki butona basarak sezonu başlatın.")


st.set_page_config(page_title="23/24 Sezonu vs Yapay Zeka", layout="wide")

# 1. VERİ YÜKLEME
@st.cache_resource
def load_data():
    data = joblib.load('comparison_data.pkl')
    return data['model'], data['team_profiles'], data['real_23_24_data']

try:
    model, team_profiles, real_df = load_data()
except FileNotFoundError:
    st.error("Lütfen önce 'model_backtest.py' kodunu çalıştırın!")
    st.stop()

# 23-24 Sezonundaki Takımları Otomatik Bul
real_teams = sorted(pd.concat([real_df['HomeTeam'], real_df['AwayTeam']]).unique())

# 2. FONKSİYONLAR

# A) Simülasyon Skor Üretici
def generate_sim_score(home, away, outcome):
    try:
        h_att = team_profiles.loc[home]['SOT'] if 'SOT' in team_profiles.columns else 4.0
        a_att = team_profiles.loc[away]['SOT'] if 'SOT' in team_profiles.columns else 3.0
    except: h_att, a_att = 1.2, 1.0
    
    h_goals = np.random.poisson(h_att / 2.5)
    a_goals = np.random.poisson(a_att / 2.5)
    
    if outcome == 1: # Draw
        g = int((h_goals + a_goals)/2)
        return g, g
    elif outcome == 2: # Home
        if h_goals <= a_goals: h_goals = a_goals + 1
        return h_goals, a_goals
    else: # Away
        if a_goals <= h_goals: a_goals = h_goals + 1
        return h_goals, a_goals

# B) Yapay Zeka Simülasyonu
def run_ai_simulation():
    table = {t: {'P':0, 'W':0, 'D':0, 'L':0, 'Pts':0, 'GF':0, 'GA':0, 'GD':0} for t in real_teams}
    
    total = len(real_teams) * (len(real_teams)-1)
    cnt = 0
    bar = st.progress(0)
    
    for home in real_teams:
        for away in real_teams:
            if home == away: continue
            
            # Model Tahmini
            try:
                # Takım verisi yoksa hata vermesin diye kontrol
                if home not in team_profiles.index or away not in team_profiles.index:
                    res = 1 # Veri yoksa berabere bitir
                    hg, ag = 1, 1
                else:
                    input_vec = np.concatenate([team_profiles.loc[home].values, team_profiles.loc[away].values]).reshape(1,-1)
                    probs = model.predict_proba(input_vec)[0]
                    res = np.random.choice([0,1,2], p=probs)
                    hg, ag = generate_sim_score(home, away, res)
            except:
                res = 1
                hg, ag = 0, 0
            
            # Tablo Güncelle
            cnt += 1
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
            
            if cnt % 10 == 0: bar.progress(cnt/total)
            
    bar.empty()
    df = pd.DataFrame.from_dict(table, orient='index')
    return df.sort_values(by=['Pts', 'GD'], ascending=False)

# C) Gerçek Tabloyu Hesaplama
def calculate_real_table():
    table = {t: {'P':0, 'W':0, 'D':0, 'L':0, 'Pts':0, 'GF':0, 'GA':0, 'GD':0} for t in real_teams}
    
    for _, row in real_df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        if h not in real_teams or a not in real_teams: continue
        
        hg, ag = int(row['FTHG']), int(row['FTAG'])
        
        table[h]['P']+=1; table[a]['P']+=1
        table[h]['GF']+=hg; table[a]['GF']+=ag
        table[h]['GA']+=ag; table[a]['GA']+=hg
        table[h]['GD']+=(hg-ag); table[a]['GD']+=(ag-hg)
        
        if hg > ag:
            table[h]['W']+=1; table[h]['Pts']+=3; table[a]['L']+=1
        elif hg == ag:
            table[h]['D']+=1; table[h]['Pts']+=1; table[a]['D']+=1; table[a]['Pts']+=1
        else:
            table[a]['W']+=1; table[a]['Pts']+=3; table[h]['L']+=1
            
    df = pd.DataFrame.from_dict(table, orient='index')
    return df.sort_values(by=['Pts', 'GD'], ascending=False)

# 3. ARAYÜZ TASARIMI
st.title("🤖 Yapay Zeka vs. 🌍 Gerçek (2023-2024)")
st.markdown("""
Bu sayfa, modelin **geçmiş sezon verileriyle eğitilip**, 2023-2024 sezonunu simüle etmesini sağlar.
Sonuçlar, gerçekte yaşanan 2023-2024 puan durumu ile yan yana kıyaslanır.
""")

if st.button("🏁 KARŞILAŞTIRMAYI BAŞLAT", type="primary"):
    
    col1, col2 = st.columns(2)
    
    # Hesaplamalar
    real_table = calculate_real_table()
    ai_table = run_ai_simulation()
    
    # --- DÜZELTİLEN KISIM BAŞLANGICI ---
    # Renklendirme fonksiyonunu daha güvenli hale getirdik.
    # Artık fonksiyon, hangi tabloya bakacağını parametre olarak alıyor.
    def get_color_style(s, df):
        # s.name takımın ismidir.
        # df.index.get_loc(s.name) ile o takımın sıralamadaki yerini buluyoruz.
        try:
            rank = df.index.get_loc(s.name)
            
            if rank == 0: # Şampiyon
                return ['background-color: #ffd700; color: black'] * len(s) 
            elif rank < 4: # UCL
                return ['background-color: #e0f7fa; color: black'] * len(s) 
            elif rank >= len(df) - 3: # Düşme Hattı
                return ['background-color: #ffcdd2; color: black'] * len(s) 
            else:
                return [''] * len(s)
        except KeyError:
            return [''] * len(s)

    with col1:
        st.subheader("🤖 Yapay Zeka Tahmini")
        # style.apply içinde lambda kullanarak tabloyu (ai_table) içeri gönderiyoruz
        st.dataframe(ai_table.style.apply(lambda x: get_color_style(x, ai_table), axis=1), height=800)
        
    with col2:
        st.subheader("🌍 Gerçek 23/24 Tablosu")
        # style.apply içinde lambda kullanarak tabloyu (real_table) içeri gönderiyoruz
        st.dataframe(real_table.style.apply(lambda x: get_color_style(x, real_table), axis=1), height=800)
    # --- DÜZELTİLEN KISIM BİTİŞİ ---

    # Fark Analizi
    st.divider()
    st.subheader("📊 Doğruluk Analizi")
    
    real_champ = real_table.index[0]
    ai_champ = ai_table.index[0]
    
    if real_champ == ai_champ:
        st.success(f"BAŞARILI! Model şampiyonu doğru bildi: **{real_champ}**")
    else:
        st.warning(f"Model şampiyonu **{ai_champ}** tahmin etti, ama gerçek şampiyon **{real_champ}**.")