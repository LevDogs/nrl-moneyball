import streamlit as st
import pandas as pd
import plotly.express as px
from math import exp

st.set_page_config(page_title="NRL Moneyball + Referee Factor", page_icon="🏈", layout="wide")
st.title("🏉 NRL Moneyball + Referee Factor")
st.markdown("**Round 15, 2026 - Transparent Model - Lineup & Ref Adjusted**")

# ====================== SIDEBAR ======================
st.sidebar.header("Model Weights")
attack_w = st.sidebar.slider("Attack (Runm + LB)", 0.3, 0.7, 0.45)
defence_w = st.sidebar.slider("Defence / Form / PD", 0.1, 0.4, 0.23)
home_w = st.sidebar.slider("Home Advantage", 0.05, 0.25, 0.12)
context_w = st.sidebar.slider("Context (Ref + Lineup)", 0.05, 0.3, 0.20)

st.sidebar.header("Adjustments")
ref_boost = st.sidebar.number_input("Ref Boost for Home (pp)", value=0.0, step=0.1)
home_yardage_adj = st.sidebar.slider("Home Yardage Reduction %", 0, 25, 0)
away_yardage_adj = st.sidebar.slider("Away Yardage Reduction %", 0, 25, 0)

# ====================== DATA ======================
data = {
    'Match': ['Rabbitohs vs Broncos', 'Dolphins vs Roosters', 'Warriors vs Sharks', 'Eels vs Raiders', 'Tigers vs Titans'],
    'Home': ['Rabbitohs', 'Dolphins', 'Warriors', 'Eels', 'Tigers'],
    'Away': ['Broncos', 'Roosters', 'Sharks', 'Raiders', 'Titans'],
    'Lineup_Notes': [
        "Rabbitohs: Strong forwards (Tatola, Fifita, Hubner). Broncos: Heavily depleted - no Haas, Walsh, Staggs",
        "Both sides heavily depleted by Origin",
        "Warriors home core strong. Sharks missing key forwards",
        "Eels without Moses. Raiders reshuffled",
        "Tigers at home. Titans competitive"
    ],
    'Home_Runm': [1687, 1650, 1720, 1580, 1620],
    'Away_Runm': [1520, 1670, 1590, 1610, 1600],
    'Home_LB': [6.3, 5.8, 6.5, 5.2, 5.5],
    'Away_LB': [4.2, 5.5, 5.0, 5.4, 5.3],
    'Home_PD': [44, 60, 120, -20, -80],
    'Away_PD': [-66, 40, 80, 30, -40],
    'Ref_Boost_Home': [-9.1, 0.0, -31.3, 0.0, 0.0],
    'Mkt_Home': [1.48, 1.43, 1.35, 2.10, 2.50]
}

df = pd.DataFrame(data)

# ====================== MODEL ======================
def calculate_prob(row):
    adj_home_runm = row['Home_Runm'] * (1 - home_yardage_adj/100)
    adj_away_runm = row['Away_Runm'] * (1 - away_yardage_adj/100)

    attack_diff = ((adj_home_runm - adj_away_runm) / 130) + ((row['Home_LB'] - row['Away_LB']) * 1.8)
    defence_diff = (row['Home_PD'] - row['Away_PD']) / 100
    ref_adj = row['Ref_Boost_Home'] / 160

    score = (attack_diff * attack_w) + (defence_diff * defence_w) + (home_w * 0.9) + (context_w * ref_adj)

    prob_home = 1 / (1 + exp(-score * 0.42))
    prob_home = max(0.35, min(0.90, prob_home))

    return round(prob_home, 4)

df['Home_Prob'] = df.apply(calculate_prob, axis=1)
df['Away_Prob'] = 1 - df['Home_Prob']
df['Fair_Home'] = round(1 / df['Home_Prob'], 2)
df['Fair_Away'] = round(1 / df['Away_Prob'], 2)
df['Edge_pp'] = round((df['Home_Prob'] - (1 / df['Mkt_Home'])) * 100, 1)

# Suggested Bet Logic
def get_suggested_bet(row):
    if row['Edge_pp'] > 7:
        return f"{row['Home']} -5.5 (STRONG)"
    elif row['Edge_pp'] > 4:
        return f"{row['Home']} H2H (value)"
    elif row['Edge_pp'] < -6:
        return f"{row['Away']} +6.5 (STRONG)"
    elif row['Edge_pp'] < -3:
        return f"{row['Away']} +X.5 (value)"
    else:
        return "No strong edge"

df['Suggested_Bet'] = df.apply(get_suggested_bet, axis=1)

# ====================== DISPLAY ======================
st.subheader("Round 15 - Full Analysis")

def color_edge(val):
    if val > 5:
        return 'background-color: #006400; color: white'
    elif val < -5:
        return 'background-color: #8B0000; color: white'
    else:
        return 'background-color: #333333'

display_cols = ['Match', 'Lineup_Notes', 'Home_Prob', 'Away_Prob', 'Fair_Home', 'Fair_Away', 'Mkt_Home', 'Edge_pp', 'Suggested_Bet']

styled_df = df[display_cols].style.format({
    'Home_Prob': '{:.1%}',
    'Away_Prob': '{:.1%}',
    'Fair_Home': '${:.2f}',
    'Fair_Away': '${:.2f}',
    'Mkt_Home': '${:.2f}',
    'Edge_pp': '{:+.1f}'
}).map(color_edge, subset=['Edge_pp'])

st.dataframe(styled_df, use_container_width=True, height=300)

st.subheader("Value Bets (Edge >= 4.0pp)")
value_bets = df[abs(df['Edge_pp']) >= 4].copy()
st.dataframe(value_bets[['Match', 'Suggested_Bet', 'Edge_pp']], use_container_width=True)

# Season Tracking
st.subheader("2026 Season Win % Tracking")
season = pd.DataFrame({
    'Team': ['Warriors', 'Roosters', 'Dolphins', 'Rabbitohs', 'Broncos', 'Eels', 'Tigers', 'Sharks'],
    'Wins': [9, 8, 7, 6, 5, 6, 4, 7],
    'Losses': [3, 4, 5, 6, 8, 7, 9, 6],
})
season['Win%'] = round(season['Wins'] / (season['Wins'] + season['Losses']) * 100, 1)
st.dataframe(season, use_container_width=True)

# Charts
col1, col2 = st.columns(2)
with col1:
    fig = px.bar(df, x='Match', y='Home_Prob', title="Home Win Probability")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig2 = px.bar(df, x='Match', y='Edge_pp', title="Edge (pp)")
    st.plotly_chart(fig2, use_container_width=True)

st.caption("Data from official NRL lists, ZeroTackle stats & refereebias.com - Gamble responsibly")
