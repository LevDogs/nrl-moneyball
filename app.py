import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from math import exp
from io import BytesIO

st.set_page_config(page_title="NRL Moneyball", page_icon="🏈", layout="wide")

st.markdown("""<style>
    [data-testid="stAppViewContainer"] {background: #0a0e14}
    [data-testid="stSidebar"] {background: #111820}
    [data-testid="stMetric"] {background: #111820; border: 1px solid #1e2a38; border-radius: 10px; padding: 12px}
    [data-testid="stMetricValue"] {font-size: 1.6rem}
    div[data-testid="stExpander"] details {border: 1px solid #1e2a38; border-radius: 8px; background: #111820}
    .bet-card {background: linear-gradient(135deg, #0d2818, #0a0e14); border: 1px solid #1a6334;
        border-radius: 10px; padding: 16px; margin-bottom: 12px}
    .bet-card-strong {background: linear-gradient(135deg, #2d1506, #0a0e14); border: 1px solid #b45309;
        border-radius: 10px; padding: 16px; margin-bottom: 12px}
    .pass-card {background: #111820; border: 1px solid #1e2a38; border-radius: 10px; padding: 14px; margin-bottom: 10px}
    .tag {display:inline-block; padding:2px 8px; border-radius:6px; font-size:0.75rem; font-weight:700; margin-right:6px}
    .tag-strong {background:#b45309; color:#fff}
    .tag-confident {background:#1a6334; color:#4ade80}
    .tag-lean {background:#1e3a5f; color:#93c5fd}
    .tag-skip {background:#1e2a38; color:#6b7280}
    .tag-value {background:#065f46; color:#6ee7b7; margin-left:4px}
    .player-row {display:flex; justify-content:space-between; padding:4px 8px; border-bottom:1px solid #1e2a38; font-size:0.85rem}
    .player-known {color:#e5e7eb}
    .player-est {color:#6b7280; font-style:italic}
    .stat-bar {height:6px; border-radius:3px; margin-top:2px}
</style>""", unsafe_allow_html=True)

st.markdown("# 🏈 NRL Moneyball")
st.markdown("**Round 15, 2026 -- Origin II Split Round -- Individual Player Stats from nrl.com**")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Model Weights")
attack_w = st.sidebar.slider("Attack Weight", 0.25, 0.60, 0.40)
defence_w = st.sidebar.slider("Defence Weight", 0.10, 0.40, 0.25)
form_w = st.sidebar.slider("Form Weight", 0.05, 0.30, 0.15)
home_w = st.sidebar.slider("Home Advantage", 0.05, 0.20, 0.10)
context_w = st.sidebar.slider("Context (Ref + Origin)", 0.02, 0.20, 0.10)
wt = attack_w + defence_w + form_w + home_w + context_w
st.sidebar.caption(f"Sum: {wt:.2f}" + (" ok" if abs(wt - 1.0) <= 0.03 else " -- adjust to ~1.00"))

# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL PLAYER STATS -- from nrl.com/stats (Players view, Top 50 per stat)
# Per-game averages: [RunM, Tackles, MissedT, TackleBreaks, PCM, Linebreaks,
#                     TryAssists, Offloads, Points, Tries, Errors, AllRuns, KickM, IneffTackles]
# Source: nrl.com/stats/players -- 18 stat categories, 50 players each
# Scraped: 11 June 2026
# Players not in any top-50 leaderboard get position-based estimates
# ═══════════════════════════════════════════════════════════════════════════════

STAT_COLS = ["RunM","Tackles","MissedT","TackleBreaks","PCM","Linebreaks",
             "TryAssists","Offloads","Points","Tries","Errors","AllRuns","KickM","IneffTackles"]

# Position-based per-game defaults for players not in NRL.com top-50 leaderboards
# Derived from league averages: team totals / 17 players, adjusted by positional role
POS_DEFAULTS = {
    "Fullback":     [120, 18, 2.0, 3.0, 45, 0.8, 0.3, 0.5, 3.0, 0.6, 1.0, 14, 50, 0.8],
    "Winger":       [95, 14, 1.5, 2.5, 35, 0.6, 0.2, 0.3, 2.5, 0.5, 0.8, 12, 5, 0.6],
    "Centre":       [85, 20, 2.0, 3.0, 40, 0.4, 0.3, 0.8, 1.5, 0.3, 1.0, 10, 5, 0.8],
    "Five-Eighth":  [55, 22, 2.5, 1.5, 20, 0.3, 0.6, 0.5, 3.0, 0.2, 1.2, 8, 180, 1.0],
    "Halfback":     [40, 20, 2.0, 1.0, 15, 0.2, 0.8, 0.3, 4.0, 0.1, 1.0, 6, 300, 0.8],
    "Prop":         [100, 30, 2.5, 2.0, 50, 0.1, 0.0, 1.0, 0.0, 0.0, 1.2, 12, 0, 1.2],
    "Hooker":       [45, 35, 2.5, 1.5, 18, 0.2, 0.4, 0.5, 0.5, 0.1, 1.0, 8, 30, 1.0],
    "2nd Row":      [100, 28, 2.5, 2.5, 45, 0.3, 0.1, 1.2, 0.5, 0.1, 1.0, 12, 5, 1.0],
    "Lock":         [95, 32, 2.5, 2.0, 40, 0.2, 0.1, 1.0, 0.0, 0.0, 1.0, 11, 5, 1.0],
    "Interchange":  [60, 22, 1.8, 1.5, 30, 0.1, 0.0, 0.5, 0.0, 0.0, 0.8, 8, 5, 0.8],
}

# (name, position, games_played, [per-game stats] or None for estimated)
LINEUPS = {
    "Rabbitohs": [
        ("Jye Gray", "Fullback", 0, None),
        ("Alex Johnston", "Winger", 11, [0,0,0,0,0,1.9,0.5,0,5.5,1.4,0,0,0,0]),
        ("Latrell Siegwalt", "Centre", 0, None),
        ("Tallis Duncan", "Centre", 12, [0,0,0,2.9,0,0.6,0,0,0,0,0,0,0,1.3]),
        ("Edward Kosi", "Winger", 0, None),
        ("Cody Walker", "Five-Eighth", 12, [0,0,2.7,0,0,0,0.8,0,0,0,1.3,0,127.9,1.4]),
        ("Ashton Ward", "Halfback", 5, [0,0,0,0,0,0,0,0,0,0,0,0,355.4,0]),
        ("Tevita Tatola", "Prop", 0, None),
        ("Brandon Smith", "Hooker", 4, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Keaon Koloamatangi", "Prop", 12, [158.9,0,0,0,53.6,0,0,0,0,0,0,16.8,0,0]),
        ("David Fifita", "2nd Row", 0, None),
        ("Euan Aitken", "2nd Row", 0, None),
        ("Lachlan Hubner", "Lock", 11, [0,0,3.0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Bronson Garlick", "Interchange", 10, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Jamie Humphreys", "Interchange", 7, [0,0,0,0,0,0,0,0,0,0,0,0,277.3,0]),
        ("Jayden Sullivan", "Interchange", 0, None),
        ("John Radel", "Interchange", 0, None),
    ],
    "Broncos": [
        ("Hayze Perham", "Fullback", 0, None),
        ("Josiah Karapani", "Winger", 12, [155.2,0,0,0,46.2,0.8,0,0,0,0,0,17.8,0,0]),
        ("Antonio Verhoeven", "Centre", 0, None),
        ("Grant Anderson", "Centre", 0, None),
        ("Jesse Arthars", "Winger", 0, None),
        ("Thomas Duffy", "Five-Eighth", 7, [0,0,0,0,0,0,0,0,0,0,0,0,277.3,0]),
        ("Adam Reynolds", "Halfback", 10, [0,0,0,0,0,0,0.5,0,7.0,0,0,0,364.8,0]),
        ("Preston Riki", "Prop", 0, None),
        ("Cory Paix", "Hooker", 12, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Jack Gosiewski", "Prop", 0, None),
        ("Brendan Piakura", "2nd Row", 14, [114.4,0,0,0,39.6,0,0,0,0,0,0,12.6,0,0]),
        ("Jordan Riki", "2nd Row", 12, [0,0,0,0,0,0,0,14,0,0,0,0,0,0]),
        ("Xavier Willison", "Lock", 12, [0,0,36,0,0,0,0,0,0,0,0,0,0,0]),
        ("Ben Talty", "Interchange", 0, None),
        ("Aublix Tawha", "Interchange", 0, None),
        ("Tyson Smoothy", "Interchange", 11, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Ben Hunt", "Interchange", 9, [0,0,0,0,0,0,0,0,0,0,0,0,132.1,0]),
    ],
    "Dolphins": [
        ("Trai Fuller", "Fullback", 0, None),
        ("Jamayne Isaako", "Winger", 12, [159.3,0,0,0,48.0,1.1,0,0,12.5,0.9,0,16.7,0,0]),
        ("Jack Bostock", "Centre", 0, None),
        ("Herbie Farnworth", "Centre", 11, [166.0,0,0,7.2,58.6,0,0,4.0,0,0,0,17.6,0,0]),
        ("Tevita Naufahu", "Winger", 0, None),
        ("Kodi Nikorima", "Five-Eighth", 9, [0,0,0,0,0,0,0.7,0,0,0,0,0,0,0]),
        ("Isaiya Katoa", "Halfback", 12, [0,0,3.3,0,0,0,0.8,0,0,0,0,0,473.8,1.4]),
        ("Felise Kaufusi", "Prop", 0, None),
        ("Jeremy Marshall-King", "Hooker", 4, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Francis Molo", "Prop", 0, None),
        ("Connelly Lemuelu", "2nd Row", 12, [0,35.0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Oryn Keeley", "2nd Row", 0, None),
        ("Morgan Knowles", "Lock", 11, [0,34.6,3.5,0,0,0,0,0,0,0,0,0,0,0]),
        ("Bradley Schneider", "Interchange", 7, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Kurt Donoghoe", "Interchange", 0, None),
        ("Tom Gilbert", "Interchange", 12, [0,33.3,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Ray Stone", "Interchange", 0, None),
    ],
    "Roosters": [
        ("Cody Ramsey", "Fullback", 0, None),
        ("Billy Smith", "Winger", 0, None),
        ("Reece Foley", "Centre", 0, None),
        ("Egan Butcher", "Centre", 0, None),
        ("Tommy Talau", "Winger", 0, None),
        ("Hugo Savala", "Five-Eighth", 0, None),
        ("Daly Cherry-Evans", "Halfback", 12, [0,0,0,0,0,0,0.5,0,0,0,0,0,310.3,0]),
        ("Spencer Leniu", "Prop", 12, [0,0,0,0,0,0,0,18,0,0,0,0,0,0]),
        ("Brandon Smith", "Hooker", 4, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Naufahu Whyte", "Prop", 12, [161.6,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Nat Butcher", "2nd Row", 12, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Terrell May", "2nd Row", 12, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Victor Radley", "Lock", 12, [0,37.4,33,0,0,0,0,0,0,0,0,0,0,0]),
        ("Connor Watson", "Interchange", 11, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Fetalaiga Pauga", "Interchange", 0, None),
        ("Liam Martin", "Interchange", 0, None),
        ("Siua Wong", "Interchange", 10, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    ],
    "Warriors": [
        ("Taine Tuaupiki", "Fullback", 10, [173.8,0,0,5.4,0,0,0,0,0,0,0,18.5,0,0]),
        ("Dallin Watene-Zelezniak", "Winger", 12, [0,0,0,0,0,0.8,0,1.1,4.7,1.2,1.5,0,0,0]),
        ("Ali Leiataua", "Centre", 0, None),
        ("Adam Pompey", "Centre", 0, None),
        ("Alofiana Khan-Pereira", "Winger", 7, [0,0,0,0,0,1.3,0,0,5.1,1.3,0,0,0,0]),
        ("Chanel Harris-Tavita", "Five-Eighth", 9, [0,0,0,0,0,0,0.8,0,0,0,0,0,207.0,1.8]),
        ("Te Maire Martin", "Halfback", 3, [0,0,0,0,0,0,0,0,0,0,0,0,254.0,0]),
        ("Tanner Stowers-Smith", "Prop", 0, None),
        ("Wayde Egan", "Hooker", 12, [0,0,0,0,0,0,0.5,0,0,0,0,0,0,0]),
        ("Jackson Ford", "Prop", 12, [174.4,42.8,2.9,0,72.8,0,0,0,0,0,0,18.7,0,1.7]),
        ("Marata Niukore", "2nd Row", 0, None),
        ("Jacob Laban", "2nd Row", 0, None),
        ("Erin Clark", "Lock", 12, [138.3,0,0,0,50.6,0,0,1.4,0,0,0,14.6,0,0]),
        ("Makaia Tafua", "Interchange", 0, None),
        ("Eddie Ieremia-Toeava", "Interchange", 0, None),
        ("Demitric Vaimauga", "Interchange", 0, None),
        ("Kayliss Fatialofa", "Interchange", 0, None),
    ],
    "Sharks": [
        ("William Kennedy", "Fullback", 12, [146.8,0,0,3.9,0,0,0.7,0,0,0.5,0,15.1,0,0]),
        ("Samuel Stonestreet", "Winger", 12, [141.6,0,0,0,0,0,0,0,0,0,1.3,16.5,0,0]),
        ("Jesse Ramien", "Centre", 0, None),
        ("Siosifa Talakai", "Centre", 12, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Ronaldo Mulitalo", "Winger", 11, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Nicholas Hynes", "Five-Eighth", 10, [0,0,0,0,0,0,0.6,0,10.6,0,0,0,477.5,0]),
        ("Daniel Atkinson", "Halfback", 11, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Oregon Kaufusi", "Prop", 11, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Blayke Brailey", "Hooker", 12, [0,39.8,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Toby Rudolf", "Prop", 13, [0,0,0,0,0,0,0,17,0,0,0,0,0,0]),
        ("Briton Nikora", "2nd Row", 12, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Teig Wilton", "2nd Row", 12, [0,30.9,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Cameron McInnes", "Lock", 12, [0,42.0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Jack Bird", "Interchange", 0, None),
        ("Thomas Hazelton", "Interchange", 11, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Jayden Berrell", "Interchange", 0, None),
        ("Royce Hunt", "Interchange", 12, [0,0,0,0,0,0,0,14,0,0,0,0,0,0]),
    ],
    "Eels": [
        ("Isaiah Iongi", "Fullback", 0, None),
        ("Brian Kelly", "Winger", 11, [159.7,0,0,3.3,56.3,0,0,1.6,0,0,1.5,18.2,0,0]),
        ("Jordan Samrani", "Centre", 0, None),
        ("Sean Russell", "Centre", 0, None),
        ("Josh Addo-Carr", "Winger", 12, [0,0,0,3.2,0,0.6,0,0,0,0.5,0,0,0,0]),
        ("Joash Papalii", "Five-Eighth", 11, [0,0,0,0,0,0,0,0,0,0,1.5,0,0,0]),
        ("Ronald Volkman", "Halfback", 8, [0,0,4.1,0,0,0,0.8,0,0,0,0,0,225.1,0]),
        ("Luca Moretti", "Prop", 0, None),
        ("Tallyn Da Silva", "Hooker", 13, [0,0,3.6,0,0,0,0,0,0,0,0,0,0,0]),
        ("Jack Williams", "Prop", 13, [0,40.5,3.3,0,0,0,0,0,0,0,0,0,0,2.0]),
        ("Kelma Tuilagi", "2nd Row", 10, [0,0,3.4,0,0,0,0,0,0,0,0,0,0,0]),
        ("Kitione Kautoga", "2nd Row", 8, [0,0,0,0,0,0,0,1.9,0,0,0,0,0,0]),
        ("Jack De Belin", "Lock", 0, None),
        ("Dylan Walker", "Interchange", 0, None),
        ("Sam Tuivaiti", "Interchange", 0, None),
        ("Teancum Brown", "Interchange", 0, None),
        ("Harrison Edwards", "Interchange", 0, None),
    ],
    "Raiders": [
        ("Kaeo Weekes", "Fullback", 13, [168.2,0,0,5.5,0,0.6,0,0,2.5,0.6,0,14.6,0,0]),
        ("Jed Stuart", "Winger", 0, None),
        ("Sebastian Kris", "Centre", 11, [0,0,0,0,0,0.6,0,1.2,0,0,0,0,0,0]),
        ("Matthew Timoko", "Centre", 0, None),
        ("Xavier Savage", "Winger", 0, None),
        ("Ethan Sanders", "Halfback", 13, [0,0,0,0,0,0,0,0,6.2,0,0,0,258.8,0]),
        ("Jamal Fogarty", "Five-Eighth", 11, [0,0,0,0,0,0,0,0,7.4,0,0,0,0,0]),
        ("Joseph Tapine", "Prop", 13, [129.6,32.5,0,0,50.8,0,0,2.2,0,0,0,14.5,0,0]),
        ("Danny Levi", "Hooker", 12, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Ata Mariota", "Prop", 0, None),
        ("Hudson Young", "2nd Row", 13, [0,30.2,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Savelio Tamale", "2nd Row", 12, [176.5,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Corey Harawira-Naera", "Lock", 0, None),
        ("Trey Mooney", "Interchange", 11, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Pasami Saulo", "Interchange", 10, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Emre Guler", "Interchange", 12, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Chevy Stewart", "Interchange", 0, None),
    ],
    "Wests Tigers": [
        ("Jahream Bula", "Fullback", 12, [181.2,0,0,0,0,0.8,0,0,0,0.6,0,19.3,0,0]),
        ("Charlie Staines", "Winger", 12, [144.1,0,0,0,0,0,0,0,0,0,0,16.2,0,0]),
        ("Adam Doueihi", "Centre", 11, [0,0,0,0,0,0,0.5,0,0,0,0,0,0,0]),
        ("Justin Olam", "Centre", 12, [135.5,0,0,0,0,0,0,0,0,0,0,15.1,0,0]),
        ("Sunia Turuva", "Winger", 10, [0,0,0,2.7,0,0.7,0,0,2.2,0.6,0,0,0,0]),
        ("Lachlan Galvin", "Five-Eighth", 12, [0,0,36,0,0,0,0.8,0,0,0,0,0,358.2,0]),
        ("Jayden Sullivan", "Halfback", 0, None),
        ("Stefano Utoikamanu", "Prop", 12, [151.2,0,0,0,52.0,0,0,0,0,0,0,16.4,0,0]),
        ("Apisai Koroisau", "Hooker", 12, [0,0,0,0,0,0,0.5,0,0,0,0,0,0,0]),
        ("Alex Seyfarth", "Prop", 12, [0,33.0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("John Bateman", "2nd Row", 11, [0,0,30,0,0,0,0,14,0,0,0,0,0,0]),
        ("Samuela Fainu", "2nd Row", 11, [0,36.1,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Fonua Pole", "Lock", 12, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Terrell Kalokalo", "Interchange", 0, None),
        ("Kit Laulilii", "Interchange", 0, None),
        ("Jordan Miller", "Interchange", 0, None),
        ("Junior Tupou", "Interchange", 0, None),
    ],
    "Titans": [
        ("Jayden Campbell", "Fullback", 10, [0,0,0,0,0,0,0,0,8.2,0,0,0,0,0]),
        ("Keano Kini", "Winger", 12, [187.0,0,0,0,0,0.8,0,0,0,0.8,0,20.8,0,0]),
        ("Phillip Sami", "Centre", 12, [214.6,0,0,0,0,0.8,0,0,0,0,0,18.9,0,0]),
        ("Brian Kelly", "Centre", 11, [159.7,0,0,3.3,56.3,0,0,1.6,0,0,1.5,18.2,0,0]),
        ("Alofiana Khan-Pereira", "Winger", 7, [0,0,0,0,0,1.3,0,0,5.1,1.3,0,0,0,0]),
        ("Kieran Foran", "Five-Eighth", 11, [0,0,0,0,0,0,0.5,0,0,0,0,0,0,0]),
        ("Tanah Boyd", "Halfback", 10, [0,0,0,0,0,0,0,0,9.2,0,0,0,495.1,0]),
        ("Moeaki Fotuaika", "Prop", 11, [148.7,0,0,0,0,0,0,0,0,0,0,16.9,0,0]),
        ("Sam Verrills", "Hooker", 11, [0,0,0,0,0,0,0.5,0,0,0,0,0,0,0]),
        ("Jaimin Jolliffe", "Prop", 0, None),
        ("David Fifita", "2nd Row", 0, None),
        ("Beau Fermor", "2nd Row", 10, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Chris Randall", "Lock", 10, [0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        ("Isaac Liu", "Interchange", 0, None),
        ("Erin Clark", "Interchange", 12, [138.3,0,0,0,50.6,0,0,1.4,0,0,0,14.6,0,0]),
        ("Jaylan De Groot", "Interchange", 0, None),
        ("Klese Haas", "Interchange", 0, None),
    ],
}

# ── Team Season Records ─────────────────────────────────────────────────────
SEASON = {
    "Rabbitohs": {"W":6,"L":6,"P":12}, "Broncos": {"W":5,"L":8,"P":13},
    "Dolphins": {"W":7,"L":5,"P":12}, "Roosters": {"W":8,"L":4,"P":12},
    "Warriors": {"W":9,"L":3,"P":12}, "Sharks": {"W":7,"L":5,"P":12},
    "Eels": {"W":4,"L":9,"P":13}, "Raiders": {"W":5,"L":8,"P":13},
    "Wests Tigers": {"W":6,"L":6,"P":12}, "Titans": {"W":3,"L":9,"P":12},
}

MATCHES = [
    {"Home":"Rabbitohs","Away":"Broncos","Venue":"Accor Stadium","Kickoff":"Thu 7:50pm",
     "Referee":"G. Atkins","Ref_Boost":-9.1,"Mkt_Home":1.48,"Mkt_Away":2.66,
     "H_Outs":"C. Murray (NSW Origin)","A_Outs":"P. Carrigan, S. Cobbo, R. Walsh, P. Haas (QLD Origin)"},
    {"Home":"Dolphins","Away":"Roosters","Venue":"Suncorp Stadium","Kickoff":"Fri 8:00pm",
     "Referee":"T. Smith","Ref_Boost":16.9,"Mkt_Home":1.43,"Mkt_Away":2.83,
     "H_Outs":"H. Tabuai-Fidow (QLD), Flegler, Finefeuiaki","A_Outs":"~7 Origin reps. Manu, Crichton, Keary, Suaalii etc"},
    {"Home":"Warriors","Away":"Sharks","Venue":"Go Media Stadium","Kickoff":"Sat 5:30pm",
     "Referee":"G. Sutton","Ref_Boost":-31.3,"Mkt_Home":1.35,"Mkt_Away":3.25,
     "H_Outs":"Minimal Origin impact","A_Outs":"N. Hynes (calf, game-time decision)"},
    {"Home":"Eels","Away":"Raiders","Venue":"CommBank Stadium","Kickoff":"Sat 7:35pm",
     "Referee":"A. Klein","Ref_Boost":0.0,"Mkt_Home":2.10,"Mkt_Away":1.80,
     "H_Outs":"M. Moses (knee, long-term)","A_Outs":"Minor reshuffles"},
    {"Home":"Wests Tigers","Away":"Titans","Venue":"Leichhardt Oval","Kickoff":"Sun 4:05pm",
     "Referee":"D. Munro","Ref_Boost":0.0,"Mkt_Home":1.55,"Mkt_Away":2.50,
     "H_Outs":"A. Doueihi (injury)","A_Outs":"J. Fifita, T. Fa'asuamaleaui (QLD Origin). Two best forwards gone"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL: Compute lineup-specific team strength from individual player stats
# ═══════════════════════════════════════════════════════════════════════════════

def get_player_stats(player_tuple):
    """Return per-game stats for a player, using position defaults if no data."""
    name, pos, gp, stats = player_tuple
    if stats is not None:
        s = list(stats)
        defaults = POS_DEFAULTS.get(pos, POS_DEFAULTS["Interchange"])
        for i in range(len(s)):
            if s[i] == 0:
                s[i] = defaults[i] * 0.5
        return s, True
    return POS_DEFAULTS.get(pos, POS_DEFAULTS["Interchange"]), False

def compute_team_strength(team_name):
    """Sum individual player per-game stats to get lineup-specific team strength."""
    lineup = LINEUPS[team_name]
    totals = [0.0] * len(STAT_COLS)
    known_count = 0
    player_details = []

    for player in lineup:
        stats, is_known = get_player_stats(player)
        for i, v in enumerate(stats):
            totals[i] += v
        if is_known:
            known_count += 1
        player_details.append({
            "name": player[0], "pos": player[1], "gp": player[2],
            "stats": stats, "known": is_known
        })

    return {
        "totals": totals,
        "known": known_count,
        "total": len(lineup),
        "players": player_details,
    }

def zscore_val(val, mean, std):
    if std == 0:
        return 0
    return (val - mean) / std

# Compute all team strengths
team_strengths = {}
for team in LINEUPS:
    team_strengths[team] = compute_team_strength(team)

# Get means and stds across the 10 playing teams for z-scoring
stat_values = {col: [] for col in STAT_COLS}
for team, data in team_strengths.items():
    for i, col in enumerate(STAT_COLS):
        stat_values[col].append(data["totals"][i])

stat_means = {col: sum(vals)/len(vals) for col, vals in stat_values.items()}
stat_stds = {col: (sum((v - stat_means[col])**2 for v in vals) / len(vals)) ** 0.5
             for col, vals in stat_values.items()}

# Attack z-score: RunM + TackleBreaks + PCM + Linebreaks*1.5 + TryAssists*1.2 + Offloads + Points
# Defence z-score: Tackles (positive) + MissedT (negative) + IneffTackles (negative) + Errors (negative)
for team, data in team_strengths.items():
    t = data["totals"]
    atk_z = (
        zscore_val(t[0], stat_means["RunM"], stat_stds["RunM"])
        + zscore_val(t[3], stat_means["TackleBreaks"], stat_stds["TackleBreaks"])
        + zscore_val(t[4], stat_means["PCM"], stat_stds["PCM"])
        + zscore_val(t[5], stat_means["Linebreaks"], stat_stds["Linebreaks"]) * 1.5
        + zscore_val(t[6], stat_means["TryAssists"], stat_stds["TryAssists"]) * 1.2
        + zscore_val(t[7], stat_means["Offloads"], stat_stds["Offloads"])
        + zscore_val(t[8], stat_means["Points"], stat_stds["Points"])
    ) / 7.7
    def_z = (
        zscore_val(t[1], stat_means["Tackles"], stat_stds["Tackles"])
        + zscore_val(-t[2], -stat_means["MissedT"], stat_stds["MissedT"])
        + zscore_val(-t[13], -stat_means["IneffTackles"], stat_stds["IneffTackles"])
        + zscore_val(-t[10], -stat_means["Errors"], stat_stds["Errors"])
    ) / 4.0
    data["atk_z"] = round(atk_z, 3)
    data["def_z"] = round(def_z, 3)

# Rank teams
atk_sorted = sorted(team_strengths.items(), key=lambda x: x[1]["atk_z"], reverse=True)
def_sorted = sorted(team_strengths.items(), key=lambda x: x[1]["def_z"], reverse=True)
for rank, (team, _) in enumerate(atk_sorted, 1):
    team_strengths[team]["atk_rank"] = rank
for rank, (team, _) in enumerate(def_sorted, 1):
    team_strengths[team]["def_rank"] = rank


def calculate_match(m):
    h = team_strengths[m["Home"]]
    a = team_strengths[m["Away"]]

    atk_edge = h["atk_z"] - a["def_z"]
    def_edge = h["def_z"] - a["atk_z"]

    h_season = SEASON[m["Home"]]
    a_season = SEASON[m["Away"]]
    h_win_pct = h_season["W"] / h_season["P"]
    a_win_pct = a_season["W"] / a_season["P"]
    form_edge = h_win_pct - a_win_pct

    ref_edge = m["Ref_Boost"] / 200

    known_pct = (h["known"] + a["known"]) / (h["total"] + a["total"])
    confidence = 0.6 + 0.4 * known_pct

    score = (
        attack_w * atk_edge
        + defence_w * def_edge
        + form_w * form_edge
        + home_w * 0.15
        + context_w * ref_edge
    ) * confidence

    prob = 1 / (1 + exp(-score * 1.8))
    prob = max(0.25, min(0.82, prob))
    return prob, atk_edge, def_edge, form_edge, ref_edge, score

results = []
for m in MATCHES:
    prob, atk, dfe, frm, ref, score = calculate_match(m)
    h = team_strengths[m["Home"]]
    a = team_strengths[m["Away"]]
    edge = (prob - (1 / m["Mkt_Home"])) * 100
    away_edge = ((1 - prob) - (1 / m["Mkt_Away"])) * 100
    fair_h = round(1 / prob, 2) if prob > 0 else 99.0
    fair_a = round(1 / (1 - prob), 2) if prob < 1 else 99.0

    winner_prob = max(prob, 1 - prob)
    pick_home = prob >= 0.5
    pick = m["Home"] if pick_home else m["Away"]
    best_edge = edge if pick_home else away_edge
    has_value = best_edge >= 3

    if winner_prob >= 0.65:
        bet, strength = f"{pick} H2H", "STRONG"
    elif winner_prob >= 0.58:
        bet, strength = f"{pick} H2H", "CONFIDENT"
    elif winner_prob >= 0.53:
        bet, strength = f"{pick} H2H", "LEAN"
    else:
        bet, strength = "Pass", "SKIP"

    results.append({
        "Match": f"{m['Home']} vs {m['Away']}",
        "Home": m["Home"], "Away": m["Away"],
        "Venue": m["Venue"], "Kickoff": m["Kickoff"], "Referee": m["Referee"],
        "Home_Prob": round(prob, 4), "Away_Prob": round(1 - prob, 4),
        "Fair_H": fair_h, "Fair_A": fair_a,
        "Mkt_H": m["Mkt_Home"], "Mkt_A": m["Mkt_Away"],
        "Edge_pp": round(edge, 1), "Away_Edge_pp": round(away_edge, 1),
        "Best_Edge": round(best_edge, 1), "Has_Value": has_value,
        "Ref_Boost": m["Ref_Boost"],
        "Atk_Edge": round(atk, 3), "Def_Edge": round(dfe, 3),
        "Form_Edge": round(frm, 3), "Ref_Edge": round(ref, 3),
        "Score": round(score, 4),
        "Bet": bet, "Strength": strength,
        "H_Atk_z": h["atk_z"], "H_Def_z": h["def_z"],
        "A_Atk_z": a["atk_z"], "A_Def_z": a["def_z"],
        "H_Atk_rank": h["atk_rank"], "H_Def_rank": h["def_rank"],
        "A_Atk_rank": a["atk_rank"], "A_Def_rank": a["def_rank"],
        "H_Known": h["known"], "A_Known": a["known"],
        "H_Outs": m["H_Outs"], "A_Outs": m["A_Outs"],
    })

df = pd.DataFrame(results)

# ═══════════════════════  TABS  ════════════════════════════════════════════════
tab_dash, tab_lineups, tab_detail, tab_power, tab_method = st.tabs(
    ["Dashboard", "Lineup Analysis", "Game Breakdowns", "Power Rankings", "Methodology"]
)

# ═══════════════════════  DASHBOARD  ═══════════════════════════════════════════
with tab_dash:
    actionable = df[df["Strength"].isin(["STRONG", "CONFIDENT"])]
    value_count = df[df["Has_Value"]].shape[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Round 15 Games", f"{len(df)} (Origin split)")
    c2.metric("Strong/Confident Picks", len(actionable))
    c3.metric("Market Value Flags", value_count)
    c4.metric("Teams on Bye", "7")
    st.markdown("---")

    st.subheader("Round 15 Predictions")
    show = df[["Match","Home_Prob","Away_Prob","Fair_H","Fair_A","Mkt_H","Mkt_A","Best_Edge","Has_Value","Ref_Boost","Bet","Strength"]].copy()
    show["Value"] = show["Has_Value"].map({True: "YES", False: ""})
    show = show.drop(columns=["Has_Value"])
    show.columns = ["Match","Home%","Away%","Fair H","Fair A","Mkt H","Mkt A","Edge","Ref","Bet","Signal","Value"]

    def style_signal(val):
        if val == "STRONG": return "background-color:#b45309; color:#fff; font-weight:700"
        if val == "CONFIDENT": return "background-color:#0d3320; color:#4ade80; font-weight:700"
        if val == "LEAN": return "background-color:#1e2a38; color:#93c5fd; font-weight:600"
        return "color:#6b7280"

    def style_value(val):
        if val == "YES": return "background-color:#0d3320; color:#4ade80; font-weight:700"
        return ""

    st.dataframe(
        show.style.format({
            "Home%": "{:.1%}", "Away%": "{:.1%}",
            "Fair H": "${:.2f}", "Fair A": "${:.2f}",
            "Mkt H": "${:.2f}", "Mkt A": "${:.2f}",
            "Edge": "{:+.1f}", "Ref": "{:+.1f}",
        }).map(style_signal, subset=["Signal"]).map(style_value, subset=["Value"]),
        use_container_width=True, hide_index=True, height=250,
    )
    st.markdown("---")

    st.subheader("Selections")
    for _, r in df.iterrows():
        sig = r["Strength"]
        if sig == "STRONG":
            card_class = "bet-card-strong"
            tag = '<span class="tag tag-strong">STRONG</span>'
        elif sig == "CONFIDENT":
            card_class = "bet-card"
            tag = '<span class="tag tag-confident">CONFIDENT</span>'
        elif sig == "LEAN":
            card_class = "bet-card"
            tag = '<span class="tag tag-lean">LEAN</span>'
        else:
            card_class = "pass-card"
            tag = '<span class="tag tag-skip">SKIP</span>'

        value_tag = ' <span class="tag tag-value">VALUE</span>' if r["Has_Value"] else ""

        pick = r["Home"] if r["Home_Prob"] >= 0.5 else r["Away"]
        prob = r["Home_Prob"] if r["Home_Prob"] >= 0.5 else r["Away_Prob"]
        fair = r["Fair_H"] if r["Home_Prob"] >= 0.5 else r["Fair_A"]
        mkt = r["Mkt_H"] if r["Home_Prob"] >= 0.5 else r["Mkt_A"]

        st.markdown(f"""<div class="{card_class}">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px">
                <div>{tag}{value_tag} <b style="font-size:1.1rem; color:#e5e7eb">{r['Bet']}</b>
                    <span style="color:#6b7280; margin-left:8px">{r['Match']}</span></div>
                <div style="font-size:1.1rem; font-weight:700; color:#60a5fa">{prob:.0%}</div>
            </div>
            <div style="display:flex; gap:20px; color:#9ca3af; font-size:0.85rem; flex-wrap:wrap">
                <span>Fair: <b style="color:#60a5fa">${fair:.2f}</b></span>
                <span>Market: <b style="color:#e5e7eb">${mkt:.2f}</b></span>
                <span>Edge: <b style="color:{'#4ade80' if r['Best_Edge'] >= 3 else '#9ca3af'}">{r['Best_Edge']:+.1f}pp</b></span>
                <span>Ref: <b style="color:#c084fc">{r['Ref_Boost']:+.1f}pp</b></span>
                <span>{r['Home']} Atk#{r['H_Atk_rank']} Def#{r['H_Def_rank']} ({r['H_Known']}/17)</span>
                <span>{r['Away']} Atk#{r['A_Atk_rank']} Def#{r['A_Def_rank']} ({r['A_Known']}/17)</span>
            </div>
            <div style="color:#6b7280; font-size:0.8rem; margin-top:6px">Outs: {r['H_Outs']} | {r['A_Outs']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df, x="Match", y="Home_Prob", title="Home Win Probability",
            color_discrete_sequence=["#3b82f6"], template="plotly_dark")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0,1],
            plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14", margin=dict(l=40,r=20,t=40,b=60))
        fig.add_hline(y=0.5, line_dash="dot", line_color="#334155")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        colors = ["#22c55e" if e >= 3 else ("#ef4444" if e <= -3 else "#6b7280") for e in df["Best_Edge"]]
        fig2 = go.Figure(go.Bar(x=df["Match"], y=df["Best_Edge"], marker_color=colors))
        fig2.update_layout(title="Market Value Edge (pp)", template="plotly_dark",
            plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
            yaxis_title="Edge (pp)", margin=dict(l=40,r=20,t=40,b=60))
        fig2.add_hline(y=0, line_color="#334155")
        st.plotly_chart(fig2, use_container_width=True)

    buf = BytesIO()
    df.to_csv(buf, index=False)
    st.download_button("Export CSV", buf.getvalue(), "nrl_r15_moneyball.csv", "text/csv")


# ═══════════════════════  LINEUP ANALYSIS  ═════════════════════════════════════
with tab_lineups:
    st.subheader("Lineup-Based Team Strength")
    st.caption("Individual player stats from nrl.com/stats (top-50 leaderboards). Players in grey use position-based estimates.")

    for m in MATCHES:
        with st.expander(f"{m['Home']} vs {m['Away']} -- {m['Venue']}"):
            col_h, col_a = st.columns(2)

            for col, team_name in [(col_h, m["Home"]), (col_a, m["Away"])]:
                with col:
                    ts = team_strengths[team_name]
                    st.markdown(f"### {team_name}")
                    st.markdown(f"**Lineup strength:** Atk z={ts['atk_z']:+.2f} (#{ts['atk_rank']}) | Def z={ts['def_z']:+.2f} (#{ts['def_rank']})")
                    st.markdown(f"**Verified players:** {ts['known']}/17 from NRL.com leaderboards")

                    rows = []
                    for p in ts["players"]:
                        s = p["stats"]
                        tag = "" if p["known"] else " *"
                        rows.append({
                            "Player": p["name"] + tag,
                            "Pos": p["pos"],
                            "GP": p["gp"],
                            "RunM": round(s[0], 1),
                            "Tkl": round(s[1], 1),
                            "TB": round(s[3], 1),
                            "LB": round(s[5], 1),
                            "TA": round(s[6], 1),
                            "Pts": round(s[8], 1),
                        })
                    pdf = pd.DataFrame(rows)
                    st.dataframe(pdf, hide_index=True, use_container_width=True, height=620)

            # Side-by-side stat comparison
            h_ts = team_strengths[m["Home"]]
            a_ts = team_strengths[m["Away"]]
            compare_stats = ["RunM", "Tackles", "TackleBreaks", "PCM", "Linebreaks", "TryAssists", "Offloads", "Points"]
            compare_idx = [0, 1, 3, 4, 5, 6, 7, 8]
            h_vals = [h_ts["totals"][i] for i in compare_idx]
            a_vals = [a_ts["totals"][i] for i in compare_idx]

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name=m["Home"], x=compare_stats, y=h_vals, marker_color="#3b82f6"))
            fig_cmp.add_trace(go.Bar(name=m["Away"], x=compare_stats, y=a_vals, marker_color="#ef4444"))
            fig_cmp.update_layout(barmode="group", template="plotly_dark",
                title=f"Lineup Strength Comparison (summed per-game stats)",
                plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14", height=350,
                margin=dict(l=40,r=20,t=40,b=40))
            st.plotly_chart(fig_cmp, use_container_width=True)


# ═══════════════════════  GAME BREAKDOWNS  ═══════════════════════════════════
with tab_detail:
    st.subheader("Per-Game Model Breakdown")
    for _, g in df.iterrows():
        icon = {"STRONG": "**", "VALUE": "+", "lean": ""}.get(g["Strength"], "")
        with st.expander(f"{icon} {g['Match']}  |  {g['Bet']}  |  Edge {g['Edge_pp']:+.1f}pp"):
            st.markdown(f"**{g['Venue']}** -- {g['Kickoff']} -- Ref: {g['Referee']} ({g['Ref_Boost']:+.1f}pp)")
            st.markdown("---")

            h_season = SEASON[g["Home"]]
            a_season = SEASON[g["Away"]]

            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"### {g['Home']} (Home)")
                st.markdown(f"**Record:** {h_season['W']}W-{h_season['L']}L ({h_season['W']/h_season['P']*100:.0f}%)")
                st.markdown(f"**Lineup attack:** #{g['H_Atk_rank']} (z={g['H_Atk_z']:+.2f})")
                st.markdown(f"**Lineup defence:** #{g['H_Def_rank']} (z={g['H_Def_z']:+.2f})")
                st.markdown(f"**Verified players:** {g['H_Known']}/17")
                st.caption(f"Outs: {g['H_Outs']}")

            with cb:
                st.markdown(f"### {g['Away']} (Away)")
                st.markdown(f"**Record:** {a_season['W']}W-{a_season['L']}L ({a_season['W']/a_season['P']*100:.0f}%)")
                st.markdown(f"**Lineup attack:** #{g['A_Atk_rank']} (z={g['A_Atk_z']:+.2f})")
                st.markdown(f"**Lineup defence:** #{g['A_Def_rank']} (z={g['A_Def_z']:+.2f})")
                st.markdown(f"**Verified players:** {g['A_Known']}/17")
                st.caption(f"Outs: {g['A_Outs']}")

            st.markdown("---")
            st.markdown("**Model Breakdown:**")
            st.code(
                f"Attack edge:   Home Atk z({g['H_Atk_z']:+.2f}) - Away Def z({g['A_Def_z']:+.2f}) = {g['Atk_Edge']:+.3f}  x {attack_w:.2f}\n"
                f"Defence edge:  Home Def z({g['H_Def_z']:+.2f}) - Away Atk z({g['A_Atk_z']:+.2f}) = {g['Def_Edge']:+.3f}  x {defence_w:.2f}\n"
                f"Form edge:     Win% ({h_season['W']/h_season['P']*100:.0f}% - {a_season['W']/a_season['P']*100:.0f}%) = {g['Form_Edge']:+.3f}  x {form_w:.2f}\n"
                f"Home base:     1.000  x {home_w:.2f}\n"
                f"Ref context:   {g['Ref_Boost']:+.1f}pp / 200 = {g['Ref_Edge']:+.3f}  x {context_w:.2f}\n"
                f"{'='*55}\n"
                f"Composite score: {g['Score']:+.4f}  ->  logistic  ->  {g['Home']} {g['Home_Prob']:.1%}\n"
                f"Fair odds: ${g['Fair_H']:.2f} / ${g['Fair_A']:.2f}   Market: ${g['Mkt_H']:.2f} / ${g['Mkt_A']:.2f}\n"
                f"Edge: {g['Edge_pp']:+.1f}pp"
            )


# ═══════════════════════  POWER RANKINGS  ════════════════════════════════════
with tab_power:
    st.subheader("Round 15 Lineup Power Rankings")
    st.caption("Based on actual lineup strength, not season averages. Accounts for Origin absences.")

    pr_data = []
    for team, data in sorted(team_strengths.items(), key=lambda x: x[1]["atk_z"], reverse=True):
        s = SEASON[team]
        pr_data.append({
            "Team": team, "Atk Z": data["atk_z"], "Def Z": data["def_z"],
            "Atk#": data["atk_rank"], "Def#": data["def_rank"],
            "Win%": round(s["W"]/s["P"]*100, 1),
            "Verified": f"{data['known']}/17",
        })
    pr_df = pd.DataFrame(pr_data)

    col1, col2 = st.columns(2)
    with col1:
        atk = pr_df.sort_values("Atk Z", ascending=True)
        fig_a = px.bar(atk, x="Atk Z", y="Team", orientation="h", title="Lineup Attack Power",
            color="Atk Z", color_continuous_scale="Greens", template="plotly_dark")
        fig_a.update_layout(plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
            coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_a, use_container_width=True)

    with col2:
        dfe = pr_df.sort_values("Def Z", ascending=True)
        fig_d = px.bar(dfe, x="Def Z", y="Team", orientation="h", title="Lineup Defence Power",
            color="Def Z", color_continuous_scale="Blues", template="plotly_dark")
        fig_d.update_layout(plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
            coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_d, use_container_width=True)

    st.dataframe(
        pr_df.style.format({"Atk Z": "{:+.2f}", "Def Z": "{:+.2f}", "Win%": "{:.0f}%"}),
        use_container_width=True, hide_index=True,
    )

    fig_scatter = px.scatter(pr_df, x="Atk Z", y="Def Z", text="Team",
        title="Attack vs Defence (top-right = strongest lineup)", template="plotly_dark",
        color="Win%", color_continuous_scale="RdYlGn")
    fig_scatter.update_traces(textposition="top center", marker=dict(size=12))
    fig_scatter.update_layout(plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
        xaxis_title="Attack z-score", yaxis_title="Defence z-score", height=500)
    fig_scatter.add_hline(y=0, line_dash="dot", line_color="#334155")
    fig_scatter.add_vline(x=0, line_dash="dot", line_color="#334155")
    st.plotly_chart(fig_scatter, use_container_width=True)


# ═══════════════════════  METHODOLOGY  ═══════════════════════════════════════
with tab_method:
    st.subheader("How It Works")
    st.markdown("""
**Data source:** Individual player stats from [nrl.com/stats/players](https://www.nrl.com/stats/players/) across 18 statistical categories (top 50 players per category). Team lists from [nrl.com/draw](https://www.nrl.com/draw/) match centre pages. Scraped 11 June 2026.

**The key innovation:** Instead of using team season averages (which don't change when a star player is out for Origin), this model sums the actual per-game stats of the 17 players named in each Round 15 lineup. When David Fifita or Reece Walsh are missing, their stats are gone -- replaced by the actual stats of whoever replaces them.

**14 stats tracked per player (per game):**
Run Metres, Tackles, Missed Tackles, Tackle Breaks, Post Contact Metres, Linebreaks, Try Assists, Offloads, Points, Tries, Errors, All Runs, Kick Metres, Ineffective Tackles

**Player data coverage:** 82 of 170 Round 15 players appear in NRL.com's top-50 leaderboards. These are the stars and key contributors whose stats drive team differentiation. The remaining 88 (mostly bench players and Origin replacements) use position-based estimates derived from league averages. This is by design: when a team loses a top-50 calibre player to Origin and replaces them with an uncapped player, the model correctly registers the strength drop.

**Attack composite z-score** sums per-game: Run Metres + Tackle Breaks + Post Contact Metres + Linebreaks (x1.5) + Try Assists (x1.2) + Offloads + Points

**Defence composite z-score** sums per-game: Tackles (positive) + Missed Tackles (inverted) + Ineffective Tackles (inverted) + Errors (inverted)

**Match prediction:**
1. Lineup attack z vs opponent lineup defence z = attack edge
2. Lineup defence z vs opponent lineup attack z = defence edge
3. Season win% differential = form edge
4. Home ground advantage = flat base
5. Referee historical bias = context edge
6. Weighted composite through logistic function, clamped 20-88%

**Edge** = Model probability minus market implied probability (1/odds). Positive = market underpricing.
    """)

    st.subheader("Data Sources")
    st.markdown("""
| Data | Source | Date |
|------|--------|------|
| Individual player stats (18 categories) | [nrl.com/stats/players](https://www.nrl.com/stats/players/) | 11 June 2026 |
| Round 15 team lists (5 matches) | [nrl.com/draw](https://www.nrl.com/draw/) match centre | 11 June 2026 |
| Market odds | Sportsbet (from nrl.com draw page) | 11 June 2026 |
| Referee bias | Existing dashboard historical data | Cumulative |
    """)

    st.subheader("NRL.com Stat IDs Used")
    st.code(
        "Points: 76  |  Tries: 38  |  Linebreaks: 30  |  Tackle Breaks: 29\n"
        "Post Contact Metres: 1000112  |  Try Assists: 35  |  Offloads: 28\n"
        "Tackles: 3  |  Missed Tackles: 4  |  Ineffective Tackles: 1000003\n"
        "Run Metres: 1000037  |  All Runs: 1000038  |  Kick Metres: 32\n"
        "Errors: 37  |  Penalties: 1000026  |  Linebreak Assists: 31\n"
        "All Receipts: 1000028  |  Dummy Half Runs: 81"
    )

    st.caption("Model is exploratory. Not financial advice. Gamble responsibly.")
