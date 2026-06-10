import streamlit as st
import pandas as pd
import plotly.express as px
from math import exp

st.set_page_config(page_title="NRL Moneyball + Referee Factor", layout="wide")
st.title("🏉 NRL Moneyball + Referee Factor")
st.markdown("**Round 15, 2026 • Transparent Model • Editable Data • Your Edge**")

# Sidebar
st.sidebar.header("Model Weights")
attack_w = st.sidebar.slider("Attack (Runm + LB)", 0.3, 0.7, 0.48)
defence_w = st.sidebar.slider("Defence / Form / PD", 0.1, 0.4, 0.22)
home_w = st.sidebar.slider("Home Advantage", 0.05, 0.25, 0.12)
context_w = st.sidebar.slider("Context (Ref + Lineup)", 0.05, 0.3, 0.18)

st.sidebar.header("Referee & Adjustments")
ref_boost = st.sidebar.number_input("Ref Boost for Home (pp)", value=0.0, step=0.1)
home_yardage_adj = st.sidebar.slider("Home Yardage Reduction %", 0, 25, 0)
away_yardage_adj = st.sidebar.slider("Away Yardage Reduction %", 0, 25, 0)

# Round 15 Data
data = {
    'Match': ['Rabbitohs vs Broncos', 'Dolphins vs Roosters', 'Warriors vs Sharks', 'Eels vs Raiders', 'Tigers vs Titans'],
    'Home': ['Rabbitohs', 'Dolphins', 'Warriors', 'Eels', 'Tigers'],
    'Away': ['Broncos', 'Roosters', 'Sharks', 'Raiders', 'Titans'],
    'Home_Runm': [1687, 1650, 1720, 1580, 1620],
    'Away_Runm': [1520, 1670, 1590, 1610, 1600],
    'Home_LB': [6.3, 5.8, 6.5, 5.2, 5.5],
    'Away_LB': [4.2, 5.5, 5.0, 5.4, 5.3],
    'Home_PD': [44, 60, 120, -20, -80],
    'Away_PD': [-66, 40, 80, 30, -40],
    'Ref_Boost_Home': [-9.1, 0.0, -31.3, 0.0, 0.0],
    'Mkt_Home': [1.72, 2.87, 2.20, 2.10, 2.50]
}

df = pd.DataFrame(data)

# Model Calculation
def calculate_prob(row):
    # Yardage adjustments
    adj_home_runm = row['Home_Runm'] * (1 - home_yardage_adj/100)
    adj_away_runm = row['Away_Runm'] * (1 - away_yardage_adj/100)

    attack_diff = ((adj_home_runm - adj_away_runm) / 100) + ((row['Home_LB'] - row['Away_LB']) * 2.5)
    defence_diff = (row['Home_PD'] - row['Away_PD']) / 60
    ref_adj = row['Ref_Boost_Home'] / 100 * 0.8   # scaled impact

    score = (attack_diff * attack_w) + (defence_diff * defence_w) + (home_w * 1.0) + (context_w * ref_adj)
    prob_home = 1 / (1 + exp(-score * 0.6))
    return round(prob_home, 4)

df['Home_Prob'] = df.apply(calculate_prob, axis=1)
df['Home_Odds'] = round(1 / df['Home_Prob'], 2)
df['Away_Odds'] = round(1 / (1 - df['Home_Prob']), 2)
df['Edge_pp'] = round((df['Home_Prob'] - (1 / df['Mkt_Home'])) * 100, 1)

# Display
st.subheader("Round 15 Predictions")
st.dataframe(df[['Match', 'Home_Prob', 'Home_Odds', 'Away_Odds', 'Mkt_Home', 'Edge_pp']], use_container_width=True)

# Value Bets
st.subheader("Value Bets (Edge ≥ 5.0pp)")
value_bets = df[abs(df['Edge_pp']) >= 5]
st.dataframe(value_bets[['Match', 'Home_Prob', 'Home_Odds', 'Mkt_Home', 'Edge_pp']])

# Charts
col1, col2 = st.columns(2)
with col1:
    fig = px.bar(df, x='Match', y='Home_Prob', title="Home Win Probability")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig2 = px.bar(df, x='Match', y='Edge_pp', title="Edge Map (pp)")
    st.plotly_chart(fig2, use_container_width=True)

st.caption("Data: 2026 season + lineups + refereebias.com • Gamble responsibly")
