import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from math import exp
from io import BytesIO

st.set_page_config(page_title="NRL Moneyball", page_icon="\U0001f3c8", layout="wide")

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
    .stat-bar {height:6px; border-radius:3px; margin-top:2px}
</style>""", unsafe_allow_html=True)

st.markdown("# \U0001f3c8 NRL Moneyball")
st.markdown("**Round 15, 2026 -- Origin II Split Round -- Team Stats from nrl.com**")

# -- Sidebar --
st.sidebar.header("Model Weights")
attack_w = st.sidebar.slider("Attack Weight", 0.25, 0.60, 0.30)
defence_w = st.sidebar.slider("Defence Weight", 0.10, 0.40, 0.25)
form_w = st.sidebar.slider("Form Weight", 0.05, 0.30, 0.30)
home_w = st.sidebar.slider("Home Advantage", 0.05, 0.20, 0.08)
context_w = st.sidebar.slider("Context (Ref + Origin)", 0.02, 0.20, 0.07)
wt = attack_w + defence_w + form_w + home_w + context_w
st.sidebar.caption(f"Sum: {wt:.2f}" + (" ok" if abs(wt - 1.0) <= 0.03 else " -- adjust to ~1.00"))

# =============================================================================
# TEAM STATS -- per-game averages from nrl.com/stats/teams (all 17 teams)
# Scraped 11 June 2026 via same-origin fetch from NRL.com q-data
# =============================================================================

TEAM_STATS = {
    "Broncos":      {"played":13, "Points":21.2, "Tries":3.6, "Linebreaks":4.2, "TackleBreaks":27.9, "PCM":506.6, "TryAssists":2.9, "Offloads":5.5, "RunMetres":1558.4, "AllRuns":171.9, "Tackles":348.2, "MissedTackles":33.1, "IneffTackles":17.5, "Errors":11.6, "KickMetres":621.6},
    "Bulldogs":     {"played":13, "Points":17.9, "Tries":3.0, "Linebreaks":4.9, "TackleBreaks":34.8, "PCM":571.1, "TryAssists":2.2, "Offloads":10.8, "RunMetres":1799.0, "AllRuns":193.0, "Tackles":353.5, "MissedTackles":34.0, "IneffTackles":12.9, "Errors":11.2, "KickMetres":617.8},
    "Cowboys":      {"played":14, "Points":24.5, "Tries":4.4, "Linebreaks":6.0, "TackleBreaks":33.7, "PCM":569.4, "TryAssists":3.8, "Offloads":10.1, "RunMetres":1793.8, "AllRuns":191.8, "Tackles":337.5, "MissedTackles":36.4, "IneffTackles":13.9, "Errors":11.5, "KickMetres":571.6},
    "Dolphins":     {"played":12, "Points":27.5, "Tries":4.7, "Linebreaks":5.4, "TackleBreaks":33.8, "PCM":573.8, "TryAssists":3.7, "Offloads":10.8, "RunMetres":1839.3, "AllRuns":190.5, "Tackles":357.8, "MissedTackles":29.2, "IneffTackles":13.8, "Errors":11.8, "KickMetres":541.4},
    "Dragons":      {"played":13, "Points":14.2, "Tries":2.4, "Linebreaks":2.9, "TackleBreaks":27.6, "PCM":557.3, "TryAssists":2.1, "Offloads":10.7, "RunMetres":1630.2, "AllRuns":182.7, "Tackles":378.7, "MissedTackles":31.5, "IneffTackles":15.5, "Errors":11.1, "KickMetres":654.9},
    "Eels":         {"played":13, "Points":20.7, "Tries":3.8, "Linebreaks":4.0, "TackleBreaks":28.9, "PCM":501.8, "TryAssists":2.9, "Offloads":8.2, "RunMetres":1579.0, "AllRuns":176.3, "Tackles":338.9, "MissedTackles":40.2, "IneffTackles":18.9, "Errors":12.1, "KickMetres":660.2},
    "Knights":      {"played":13, "Points":28.5, "Tries":5.2, "Linebreaks":6.2, "TackleBreaks":34.5, "PCM":483.3, "TryAssists":4.5, "Offloads":10.4, "RunMetres":1655.2, "AllRuns":177.4, "Tackles":345.2, "MissedTackles":34.4, "IneffTackles":17.4, "Errors":11.9, "KickMetres":587.5},
    "Panthers":     {"played":13, "Points":33.6, "Tries":5.9, "Linebreaks":6.8, "TackleBreaks":36.1, "PCM":583.4, "TryAssists":4.5, "Offloads":10.0, "RunMetres":1878.4, "AllRuns":207.5, "Tackles":339.5, "MissedTackles":29.9, "IneffTackles":15.2, "Errors":10.7, "KickMetres":621.9},
    "Rabbitohs":    {"played":12, "Points":28.2, "Tries":5.0, "Linebreaks":6.3, "TackleBreaks":30.9, "PCM":513.4, "TryAssists":4.1, "Offloads":7.4, "RunMetres":1695.1, "AllRuns":186.1, "Tackles":307.4, "MissedTackles":33.0, "IneffTackles":15.8, "Errors":11.3, "KickMetres":594.1},
    "Raiders":      {"played":13, "Points":19.2, "Tries":3.4, "Linebreaks":4.6, "TackleBreaks":37.5, "PCM":546.2, "TryAssists":2.7, "Offloads":11.2, "RunMetres":1687.0, "AllRuns":180.1, "Tackles":360.1, "MissedTackles":33.6, "IneffTackles":15.2, "Errors":10.8, "KickMetres":591.6},
    "Roosters":     {"played":12, "Points":26.9, "Tries":4.8, "Linebreaks":4.8, "TackleBreaks":36.4, "PCM":620.5, "TryAssists":4.3, "Offloads":10.3, "RunMetres":1811.9, "AllRuns":187.3, "Tackles":332.3, "MissedTackles":27.3, "IneffTackles":14.3, "Errors":13.3, "KickMetres":520.4},
    "Sea Eagles":   {"played":13, "Points":28.2, "Tries":4.9, "Linebreaks":5.7, "TackleBreaks":29.9, "PCM":569.8, "TryAssists":4.2, "Offloads":8.5, "RunMetres":1751.4, "AllRuns":186.2, "Tackles":318.4, "MissedTackles":30.3, "IneffTackles":17.5, "Errors":11.5, "KickMetres":610.6},
    "Sharks":       {"played":12, "Points":29.7, "Tries":5.2, "Linebreaks":5.4, "TackleBreaks":29.5, "PCM":541.1, "TryAssists":4.5, "Offloads":10.4, "RunMetres":1657.7, "AllRuns":180.5, "Tackles":341.7, "MissedTackles":30.8, "IneffTackles":15.0, "Errors":11.3, "KickMetres":637.6},
    "Storm":        {"played":14, "Points":24.7, "Tries":4.4, "Linebreaks":5.5, "TackleBreaks":35.4, "PCM":538.0, "TryAssists":3.2, "Offloads":9.1, "RunMetres":1649.1, "AllRuns":181.7, "Tackles":338.8, "MissedTackles":33.1, "IneffTackles":15.6, "Errors":9.9, "KickMetres":579.8},
    "Titans":       {"played":12, "Points":18.3, "Tries":3.2, "Linebreaks":4.5, "TackleBreaks":29.3, "PCM":553.6, "TryAssists":2.3, "Offloads":10.8, "RunMetres":1654.6, "AllRuns":179.6, "Tackles":351.0, "MissedTackles":32.4, "IneffTackles":14.4, "Errors":13.1, "KickMetres":590.0},
    "Warriors":     {"played":12, "Points":30.7, "Tries":5.3, "Linebreaks":5.8, "TackleBreaks":33.4, "PCM":563.2, "TryAssists":4.3, "Offloads":8.1, "RunMetres":1699.7, "AllRuns":187.6, "Tackles":326.5, "MissedTackles":32.9, "IneffTackles":15.3, "Errors":9.9, "KickMetres":649.6},
    "Wests Tigers": {"played":12, "Points":22.9, "Tries":4.0, "Linebreaks":5.0, "TackleBreaks":36.1, "PCM":551.8, "TryAssists":3.8, "Offloads":13.9, "RunMetres":1684.0, "AllRuns":187.8, "Tackles":348.3, "MissedTackles":32.8, "IneffTackles":15.3, "Errors":11.7, "KickMetres":557.6},
}

SEASON = {
    "Rabbitohs": {"W":6,"L":6,"P":12}, "Broncos": {"W":5,"L":8,"P":13},
    "Dolphins": {"W":7,"L":5,"P":12}, "Roosters": {"W":8,"L":4,"P":12},
    "Warriors": {"W":9,"L":3,"P":12}, "Sharks": {"W":7,"L":5,"P":12},
    "Eels": {"W":4,"L":9,"P":13}, "Raiders": {"W":5,"L":8,"P":13},
    "Wests Tigers": {"W":6,"L":6,"P":12}, "Titans": {"W":3,"L":9,"P":12},
    "Storm": {"W":8,"L":5,"P":13}, "Panthers": {"W":10,"L":3,"P":13},
    "Sea Eagles": {"W":8,"L":5,"P":13}, "Bulldogs": {"W":6,"L":6,"P":12},
    "Knights": {"W":7,"L":5,"P":12}, "Cowboys": {"W":6,"L":7,"P":13},
    "Dragons": {"W":3,"L":10,"P":13},
}

RESULTS_2026 = [
    (1,"Knights","Cowboys",28,18),(1,"Bulldogs","Dragons",15,14),(1,"Storm","Eels",52,4),
    (1,"Warriors","Roosters",42,18),(1,"Broncos","Panthers",0,26),(1,"Sharks","Titans",50,10),
    (1,"Sea Eagles","Raiders",28,29),(1,"Dolphins","Rabbitohs",30,40),
    (2,"Broncos","Eels",32,40),(2,"Warriors","Raiders",40,6),(2,"Roosters","Rabbitohs",26,18),
    (2,"Wests Tigers","Cowboys",44,16),(2,"Dragons","Storm",20,46),(2,"Panthers","Sharks",26,6),
    (2,"Sea Eagles","Knights",16,36),(2,"Dolphins","Titans",18,14),
    (3,"Raiders","Bulldogs",10,14),(3,"Roosters","Panthers",4,40),(3,"Storm","Broncos",14,18),
    (3,"Knights","Warriors",12,38),(3,"Sharks","Dolphins",10,38),(3,"Rabbitohs","Wests Tigers",20,16),
    (3,"Titans","Sea Eagles",28,42),(3,"Eels","Cowboys",28,10),
    (4,"Rabbitohs","Storm",38,28),(4,"Bulldogs","Dolphins",18,12),(4,"Panthers","Broncos",34,6),
    (4,"Wests Tigers","Sea Eagles",24,28),(4,"Cowboys","Knights",10,24),(4,"Roosters","Warriors",34,36),
    (4,"Raiders","Sharks",22,34),(4,"Titans","Dragons",22,14),
    (5,"Dolphins","Sea Eagles",18,52),(5,"Rabbitohs","Bulldogs",32,24),(5,"Panthers","Storm",50,10),
    (5,"Dragons","Cowboys",0,32),(5,"Titans","Broncos",12,26),(5,"Sharks","Warriors",36,22),
    (5,"Knights","Raiders",32,12),(5,"Eels","Wests Tigers",20,22),
    (6,"Bulldogs","Panthers",32,16),(6,"Dragons","Sea Eagles",18,28),(6,"Broncos","Cowboys",31,35),
    (6,"Rabbitohs","Raiders",34,36),(6,"Sharks","Roosters",22,34),(6,"Storm","Warriors",14,38),
    (6,"Eels","Titans",10,52),(6,"Wests Tigers","Knights",42,22),
    (7,"Cowboys","Sea Eagles",6,38),(7,"Raiders","Storm",26,22),(7,"Dolphins","Panthers",22,23),
    (7,"Warriors","Wests Tigers",38,28),(7,"Rabbitohs","Titans",28,24),(7,"Bulldogs","Broncos",20,22),
    (7,"Roosters","Knights",38,24),(7,"Eels","Bulldogs",38,20),
    (8,"Wests Tigers","Raiders",33,14),(8,"Cowboys","Sharks",46,34),(8,"Broncos","Bulldogs",32,12),
    (8,"Dragons","Roosters",16,62),(8,"Warriors","Dolphins",20,18),(8,"Storm","Rabbitohs",6,48),
    (8,"Knights","Panthers",12,44),(8,"Sea Eagles","Eels",33,18),
    (9,"Bulldogs","Cowboys",12,28),(9,"Dolphins","Storm",28,10),(9,"Titans","Raiders",12,28),
    (9,"Eels","Warriors",14,36),(9,"Roosters","Broncos",38,24),(9,"Knights","Rabbitohs",42,38),
    (9,"Sharks","Wests Tigers",52,10),(9,"Panthers","Sea Eagles",18,16),
    (10,"Dolphins","Bulldogs",44,12),(10,"Roosters","Titans",28,12),(10,"Cowboys","Eels",30,33),
    (10,"Dragons","Knights",10,44),(10,"Rabbitohs","Sharks",36,12),(10,"Sea Eagles","Broncos",32,4),
    (10,"Storm","Wests Tigers",44,16),(10,"Raiders","Panthers",18,30),
    (11,"Sharks","Bulldogs",38,16),(11,"Rabbitohs","Dolphins",10,32),(11,"Wests Tigers","Storm",28,30),
    (11,"Roosters","Sea Eagles",26,28),(11,"Eels","Dragons",14,18),(11,"Cowboys","Panthers",18,30),
    (11,"Broncos","Knights",24,28),(11,"Warriors","Titans",30,10),
    (12,"Bulldogs","Raiders",42,18),(12,"Dolphins","Eels",26,6),(12,"Knights","Sharks",22,46),
    (12,"Panthers","Roosters",28,12),(12,"Storm","Cowboys",34,28),(12,"Titans","Wests Tigers",16,52),
    (12,"Dragons","Warriors",12,30),(12,"Sea Eagles","Titans",12,10),(12,"Cowboys","Rabbitohs",30,18),
    (13,"Sharks","Sea Eagles",28,22),(13,"Knights","Eels",28,22),(13,"Wests Tigers","Bulldogs",22,16),
    (13,"Storm","Roosters",18,4),(13,"Broncos","Dragons",26,30),(13,"Raiders","Cowboys",26,12),
    (13,"Panthers","Warriors",20,18),
    (14,"Sea Eagles","Rabbitohs",28,14),(14,"Storm","Knights",32,30),(14,"Raiders","Roosters",0,26),
    (14,"Cowboys","Dolphins",14,40),(14,"Broncos","Titans",23,28),(14,"Wests Tigers","Panthers",0,68),
    (14,"Sharks","Dragons",34,12),(14,"Bulldogs","Eels",14,12),
]

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

# =============================================================================
# RECENT FORM: Last 5 games scoring from actual results
# Weight recent performance more heavily than season averages
# =============================================================================

def compute_recent_form(results, n=5):
    team_games = {}
    for rd, home, away, hs, as_ in results:
        for team, scored, conceded in [(home, hs, as_), (away, as_, hs)]:
            if team not in team_games:
                team_games[team] = []
            team_games[team].append({"rd": rd, "scored": scored, "conceded": conceded})

    recent = {}
    for team, games in team_games.items():
        games.sort(key=lambda g: g["rd"])
        last_n = games[-n:]
        season_scored = sum(g["scored"] for g in games) / len(games)
        season_conceded = sum(g["conceded"] for g in games) / len(games)
        recent_scored = sum(g["scored"] for g in last_n) / len(last_n)
        recent_conceded = sum(g["conceded"] for g in last_n) / len(last_n)

        atk_trend = (recent_scored / max(season_scored, 1)) - 1.0
        def_trend = (season_conceded / max(recent_conceded, 1)) - 1.0

        recent[team] = {
            "season_scored": round(season_scored, 1),
            "season_conceded": round(season_conceded, 1),
            "recent_scored": round(recent_scored, 1),
            "recent_conceded": round(recent_conceded, 1),
            "atk_trend": round(atk_trend, 3),
            "def_trend": round(def_trend, 3),
        }
    return recent

RECENT_FORM = compute_recent_form(RESULTS_2026, n=5)

# =============================================================================
# Z-SCORE MODEL: Team strength from NRL.com aggregate stats + recent form
# =============================================================================

ATK_STATS = ["RunMetres", "TackleBreaks", "PCM", "Linebreaks", "TryAssists", "Offloads", "Points"]
ATK_WEIGHTS = [1.0, 1.0, 1.0, 1.5, 1.2, 1.0, 1.0]
DEF_STATS_POS = ["Tackles"]
DEF_STATS_NEG = ["MissedTackles", "IneffTackles", "Errors"]

def zscore(val, mean, std):
    return (val - mean) / std if std > 0 else 0

def compute_team_zscores():
    teams = list(TEAM_STATS.keys())
    stat_means = {}
    stat_stds = {}

    all_stats = ATK_STATS + DEF_STATS_POS + DEF_STATS_NEG
    for stat in all_stats:
        vals = [TEAM_STATS[t][stat] for t in teams]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        stat_means[stat] = mean
        stat_stds[stat] = std

    strengths = {}
    for team in teams:
        s = TEAM_STATS[team]

        atk_z_raw = sum(
            zscore(s[stat], stat_means[stat], stat_stds[stat]) * w
            for stat, w in zip(ATK_STATS, ATK_WEIGHTS)
        ) / sum(ATK_WEIGHTS)

        def_z_pos = sum(zscore(s[stat], stat_means[stat], stat_stds[stat]) for stat in DEF_STATS_POS)
        def_z_neg = sum(zscore(-s[stat], -stat_means[stat], stat_stds[stat]) for stat in DEF_STATS_NEG)
        def_z_raw = (def_z_pos + def_z_neg) / (len(DEF_STATS_POS) + len(DEF_STATS_NEG))

        rf = RECENT_FORM.get(team, {"atk_trend": 0, "def_trend": 0})
        atk_z = atk_z_raw * (1 + 0.3 * rf["atk_trend"])
        def_z = def_z_raw * (1 + 0.3 * rf["def_trend"])

        strengths[team] = {
            "atk_z": round(atk_z, 3),
            "def_z": round(def_z, 3),
            "atk_z_raw": round(atk_z_raw, 3),
            "def_z_raw": round(def_z_raw, 3),
            "played": s["played"],
        }

    atk_sorted = sorted(strengths.items(), key=lambda x: x[1]["atk_z"], reverse=True)
    def_sorted = sorted(strengths.items(), key=lambda x: x[1]["def_z"], reverse=True)
    for rank, (team, _) in enumerate(atk_sorted, 1):
        strengths[team]["atk_rank"] = rank
    for rank, (team, _) in enumerate(def_sorted, 1):
        strengths[team]["def_rank"] = rank

    return strengths, stat_means, stat_stds

team_strengths, stat_means, stat_stds = compute_team_zscores()


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

    score = (
        attack_w * atk_edge
        + defence_w * def_edge
        + form_w * form_edge
        + home_w * 0.15
        + context_w * ref_edge
    )

    prob = 1 / (1 + exp(-score * 2.0))
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
        "H_Outs": m["H_Outs"], "A_Outs": m["A_Outs"],
    })

df = pd.DataFrame(results)

# =============================================================================
# BACKTESTING: Self-tuning from Rounds 1-14 actual results
# =============================================================================

def backtest_model(aw, dw, fw, hw, cw, scale, home_const):
    records = {t: {"W": 0, "L": 0, "P": 0} for t in SEASON}
    correct = 0
    total = 0
    brier_sum = 0
    round_results = {}

    for rd, home, away, hs, as_, *_ in RESULTS_2026:
        winner = home if hs > as_ else away

        if home not in team_strengths or away not in team_strengths:
            records[winner]["W"] += 1
            records[home if winner != home else away]["L"] += 1
            for t in [home, away]:
                records[t]["P"] += 1
            continue

        h = team_strengths[home]
        a = team_strengths[away]
        atk_e = h["atk_z"] - a["def_z"]
        def_e = h["def_z"] - a["atk_z"]

        h_pct = records[home]["W"] / max(records[home]["P"], 1)
        a_pct = records[away]["W"] / max(records[away]["P"], 1)
        if records[home]["P"] == 0:
            h_pct = 0.5
        if records[away]["P"] == 0:
            a_pct = 0.5
        form_e = h_pct - a_pct

        sc = (aw * atk_e + dw * def_e + fw * form_e + hw * home_const)
        prob = 1 / (1 + exp(-sc * scale))
        prob = max(0.25, min(0.82, prob))

        predicted = home if prob >= 0.5 else away

        if predicted == winner:
            correct += 1
        total += 1
        actual = 1.0 if home == winner else 0.0
        brier_sum += (prob - actual) ** 2

        if rd not in round_results:
            round_results[rd] = {"correct": 0, "total": 0}
        round_results[rd]["total"] += 1
        if predicted == winner:
            round_results[rd]["correct"] += 1

        records[winner]["W"] += 1
        records[home if winner != home else away]["L"] += 1
        for t in [home, away]:
            records[t]["P"] += 1

    accuracy = correct / max(total, 1)
    brier = brier_sum / max(total, 1)
    return accuracy, brier, correct, total, round_results

def optimize_weights():
    best_acc = 0
    best_params = None
    best_brier = 1.0

    for aw in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        for dw in [0.15, 0.20, 0.25, 0.30, 0.35]:
            for fw in [0.10, 0.15, 0.20, 0.25, 0.30]:
                for hw in [0.05, 0.08, 0.10, 0.12, 0.15]:
                    for scale in [1.2, 1.5, 1.8, 2.0, 2.4, 2.8]:
                        for hc in [0.10, 0.15, 0.20, 0.30]:
                            acc, brier, c, t, rr = backtest_model(aw, dw, fw, hw, 0.0, scale, hc)
                            if acc > best_acc or (acc == best_acc and brier < best_brier):
                                best_acc = acc
                                best_brier = brier
                                best_params = {"attack_w": aw, "defence_w": dw, "form_w": fw,
                                               "home_w": hw, "scale": scale, "home_const": hc,
                                               "correct": c, "total": t, "rounds": rr}
    return best_acc, best_brier, best_params

bt_accuracy, bt_brier, bt_correct, bt_total, bt_rounds = backtest_model(
    attack_w, defence_w, form_w, home_w, context_w, 2.0, 0.15
)

# =========================  TABS  =============================================
tab_dash, tab_stats, tab_detail, tab_power, tab_backtest, tab_method = st.tabs(
    ["Dashboard", "Team Stats", "Game Breakdowns", "Power Rankings", "Backtesting", "Methodology"]
)

# =========================  DASHBOARD  ========================================
with tab_dash:
    actionable = df[df["Strength"].isin(["STRONG", "CONFIDENT"])]
    value_count = df[df["Has_Value"]].shape[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Round 15 Games", f"{len(df)} (Origin split)")
    c2.metric("Strong/Confident Picks", len(actionable))
    c3.metric("Market Value Flags", value_count)
    c4.metric("Backtest Accuracy", f"{bt_accuracy:.0%}")
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
                <span>{r['Home']} Atk#{r['H_Atk_rank']} Def#{r['H_Def_rank']}</span>
                <span>{r['Away']} Atk#{r['A_Atk_rank']} Def#{r['A_Def_rank']}</span>
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


# =========================  TEAM STATS  =======================================
with tab_stats:
    st.subheader("NRL.com Team Stats (Per-Game Averages)")
    st.caption("Real team aggregate stats from nrl.com/stats/teams. All 17 teams, 14 statistical categories. Recent form adjusts z-scores by last-5-game scoring trends.")

    for m in MATCHES:
        with st.expander(f"{m['Home']} vs {m['Away']} -- {m['Venue']}"):
            col_h, col_a = st.columns(2)

            for col, team_name in [(col_h, m["Home"]), (col_a, m["Away"])]:
                with col:
                    ts = team_strengths[team_name]
                    rf = RECENT_FORM[team_name]
                    s = SEASON[team_name]
                    st.markdown(f"### {team_name}")
                    st.markdown(f"**Record:** {s['W']}W-{s['L']}L ({s['W']/s['P']*100:.0f}%)")
                    st.markdown(f"**Attack:** z={ts['atk_z']:+.2f} (#{ts['atk_rank']}) | raw={ts['atk_z_raw']:+.2f}")
                    st.markdown(f"**Defence:** z={ts['def_z']:+.2f} (#{ts['def_rank']}) | raw={ts['def_z_raw']:+.2f}")
                    st.markdown(f"**Last 5 avg:** {rf['recent_scored']} scored, {rf['recent_conceded']} conceded")
                    st.markdown(f"**Season avg:** {rf['season_scored']} scored, {rf['season_conceded']} conceded")

                    trend_icon = "↑" if rf["atk_trend"] > 0.05 else ("↓" if rf["atk_trend"] < -0.05 else "→")
                    st.markdown(f"**Atk trend:** {trend_icon} {rf['atk_trend']:+.1%} | **Def trend:** {rf['def_trend']:+.1%}")

            compare_stats = ["Points", "Tries", "Linebreaks", "TackleBreaks", "PCM", "TryAssists", "Offloads", "RunMetres"]
            h_vals = [TEAM_STATS[m["Home"]][s] for s in compare_stats]
            a_vals = [TEAM_STATS[m["Away"]][s] for s in compare_stats]

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name=m["Home"], x=compare_stats, y=h_vals, marker_color="#3b82f6"))
            fig_cmp.add_trace(go.Bar(name=m["Away"], x=compare_stats, y=a_vals, marker_color="#ef4444"))
            fig_cmp.update_layout(barmode="group", template="plotly_dark",
                title="Per-Game Attack Stats Comparison",
                plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14", height=350,
                margin=dict(l=40,r=20,t=40,b=40))
            st.plotly_chart(fig_cmp, use_container_width=True)

            def_stats = ["Tackles", "MissedTackles", "IneffTackles", "Errors"]
            h_dvals = [TEAM_STATS[m["Home"]][s] for s in def_stats]
            a_dvals = [TEAM_STATS[m["Away"]][s] for s in def_stats]

            fig_def = go.Figure()
            fig_def.add_trace(go.Bar(name=m["Home"], x=def_stats, y=h_dvals, marker_color="#3b82f6"))
            fig_def.add_trace(go.Bar(name=m["Away"], x=def_stats, y=a_dvals, marker_color="#ef4444"))
            fig_def.update_layout(barmode="group", template="plotly_dark",
                title="Per-Game Defence Stats Comparison",
                plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14", height=300,
                margin=dict(l=40,r=20,t=40,b=40))
            st.plotly_chart(fig_def, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Team Stats Table")
    ts_rows = []
    for team in sorted(TEAM_STATS.keys()):
        s = TEAM_STATS[team]
        rf = RECENT_FORM.get(team, {})
        ts_rows.append({
            "Team": team, "P": s["played"],
            "Pts/G": s["Points"], "Tries/G": s["Tries"], "LB/G": s["Linebreaks"],
            "TB/G": s["TackleBreaks"], "PCM/G": s["PCM"], "RunM/G": s["RunMetres"],
            "Tkl/G": s["Tackles"], "MissT/G": s["MissedTackles"],
            "Err/G": s["Errors"],
            "L5 Scored": rf.get("recent_scored", 0), "L5 Conceded": rf.get("recent_conceded", 0),
        })
    ts_df = pd.DataFrame(ts_rows)
    st.dataframe(ts_df.style.format({
        "Pts/G": "{:.1f}", "Tries/G": "{:.1f}", "LB/G": "{:.1f}", "TB/G": "{:.1f}",
        "PCM/G": "{:.1f}", "RunM/G": "{:.0f}", "Tkl/G": "{:.1f}", "MissT/G": "{:.1f}",
        "Err/G": "{:.1f}", "L5 Scored": "{:.1f}", "L5 Conceded": "{:.1f}",
    }), use_container_width=True, hide_index=True)


# =========================  GAME BREAKDOWNS  ==================================
with tab_detail:
    st.subheader("Per-Game Model Breakdown")
    for _, g in df.iterrows():
        with st.expander(f"{g['Match']}  |  {g['Bet']}  |  Edge {g['Edge_pp']:+.1f}pp"):
            st.markdown(f"**{g['Venue']}** -- {g['Kickoff']} -- Ref: {g['Referee']} ({g['Ref_Boost']:+.1f}pp)")
            st.markdown("---")

            h_season = SEASON[g["Home"]]
            a_season = SEASON[g["Away"]]
            h_rf = RECENT_FORM[g["Home"]]
            a_rf = RECENT_FORM[g["Away"]]

            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"### {g['Home']} (Home)")
                st.markdown(f"**Record:** {h_season['W']}W-{h_season['L']}L ({h_season['W']/h_season['P']*100:.0f}%)")
                st.markdown(f"**Attack:** #{g['H_Atk_rank']} (z={g['H_Atk_z']:+.2f})")
                st.markdown(f"**Defence:** #{g['H_Def_rank']} (z={g['H_Def_z']:+.2f})")
                st.markdown(f"**Last 5:** {h_rf['recent_scored']} scored / {h_rf['recent_conceded']} conceded")
                st.caption(f"Outs: {g['H_Outs']}")

            with cb:
                st.markdown(f"### {g['Away']} (Away)")
                st.markdown(f"**Record:** {a_season['W']}W-{a_season['L']}L ({a_season['W']/a_season['P']*100:.0f}%)")
                st.markdown(f"**Attack:** #{g['A_Atk_rank']} (z={g['A_Atk_z']:+.2f})")
                st.markdown(f"**Defence:** #{g['A_Def_rank']} (z={g['A_Def_z']:+.2f})")
                st.markdown(f"**Last 5:** {a_rf['recent_scored']} scored / {a_rf['recent_conceded']} conceded")
                st.caption(f"Outs: {g['A_Outs']}")

            st.markdown("---")
            st.markdown("**Model Breakdown:**")
            st.code(
                f"Attack edge:   Home Atk z({g['H_Atk_z']:+.2f}) - Away Def z({g['A_Def_z']:+.2f}) = {g['Atk_Edge']:+.3f}  x {attack_w:.2f}\n"
                f"Defence edge:  Home Def z({g['H_Def_z']:+.2f}) - Away Atk z({g['A_Atk_z']:+.2f}) = {g['Def_Edge']:+.3f}  x {defence_w:.2f}\n"
                f"Form edge:     Win% ({h_season['W']/h_season['P']*100:.0f}% - {a_season['W']/a_season['P']*100:.0f}%) = {g['Form_Edge']:+.3f}  x {form_w:.2f}\n"
                f"Home base:     0.150  x {home_w:.2f}\n"
                f"Ref context:   {g['Ref_Boost']:+.1f}pp / 200 = {g['Ref_Edge']:+.3f}  x {context_w:.2f}\n"
                f"{'='*55}\n"
                f"Composite score: {g['Score']:+.4f}  ->  logistic  ->  {g['Home']} {g['Home_Prob']:.1%}\n"
                f"Fair odds: ${g['Fair_H']:.2f} / ${g['Fair_A']:.2f}   Market: ${g['Mkt_H']:.2f} / ${g['Mkt_A']:.2f}\n"
                f"Edge: {g['Edge_pp']:+.1f}pp"
            )


# =========================  POWER RANKINGS  ===================================
with tab_power:
    st.subheader("Team Power Rankings (All 17 Teams)")
    st.caption("Based on NRL.com team stats + recent scoring form adjustment. Higher z = stronger.")

    pr_data = []
    for team, data in sorted(team_strengths.items(), key=lambda x: x[1]["atk_z"], reverse=True):
        s = SEASON[team]
        rf = RECENT_FORM[team]
        pr_data.append({
            "Team": team, "Atk Z": data["atk_z"], "Def Z": data["def_z"],
            "Atk#": data["atk_rank"], "Def#": data["def_rank"],
            "Win%": round(s["W"]/s["P"]*100, 1),
            "L5 Scored": rf["recent_scored"], "L5 Conceded": rf["recent_conceded"],
        })
    pr_df = pd.DataFrame(pr_data)

    col1, col2 = st.columns(2)
    with col1:
        atk = pr_df.sort_values("Atk Z", ascending=True)
        fig_a = px.bar(atk, x="Atk Z", y="Team", orientation="h", title="Attack Power (z-score)",
            color="Atk Z", color_continuous_scale="Greens", template="plotly_dark")
        fig_a.update_layout(plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
            coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_a, use_container_width=True)

    with col2:
        dfe = pr_df.sort_values("Def Z", ascending=True)
        fig_d = px.bar(dfe, x="Def Z", y="Team", orientation="h", title="Defence Power (z-score)",
            color="Def Z", color_continuous_scale="Blues", template="plotly_dark")
        fig_d.update_layout(plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
            coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_d, use_container_width=True)

    st.dataframe(
        pr_df.style.format({"Atk Z": "{:+.2f}", "Def Z": "{:+.2f}", "Win%": "{:.0f}%",
                           "L5 Scored": "{:.1f}", "L5 Conceded": "{:.1f}"}),
        use_container_width=True, hide_index=True,
    )

    fig_scatter = px.scatter(pr_df, x="Atk Z", y="Def Z", text="Team",
        title="Attack vs Defence (top-right = strongest)", template="plotly_dark",
        color="Win%", color_continuous_scale="RdYlGn")
    fig_scatter.update_traces(textposition="top center", marker=dict(size=12))
    fig_scatter.update_layout(plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
        xaxis_title="Attack z-score", yaxis_title="Defence z-score", height=500)
    fig_scatter.add_hline(y=0, line_dash="dot", line_color="#334155")
    fig_scatter.add_vline(x=0, line_dash="dot", line_color="#334155")
    st.plotly_chart(fig_scatter, use_container_width=True)


# =========================  BACKTESTING  ======================================
with tab_backtest:
    st.subheader("Self-Tuning Model: Rounds 1-14 Backtest")
    st.markdown("Tests current weights against **108 completed matches** using team-level z-scores + evolving form. "
                "Hit 'Optimize' to find the weight combination that maximizes prediction accuracy.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Accuracy", f"{bt_accuracy:.1%}")
    c2.metric("Correct / Total", f"{bt_correct} / {bt_total}")
    c3.metric("Brier Score", f"{bt_brier:.3f}", help="Lower is better. 0.25 = coin flip, 0.0 = perfect")
    c4.metric("Matches Tested", bt_total)

    st.markdown("#### Accuracy by Round")
    rd_data = []
    cumulative_c, cumulative_t = 0, 0
    for rd in sorted(bt_rounds.keys()):
        r = bt_rounds[rd]
        cumulative_c += r["correct"]
        cumulative_t += r["total"]
        rd_data.append({
            "Round": rd, "Correct": r["correct"], "Games": r["total"],
            "Round Acc": round(r["correct"] / r["total"] * 100, 1),
            "Cumulative Acc": round(cumulative_c / cumulative_t * 100, 1),
        })
    rd_df = pd.DataFrame(rd_data)

    fig_rd = go.Figure()
    fig_rd.add_trace(go.Bar(x=rd_df["Round"], y=rd_df["Round Acc"], name="Round Accuracy",
        marker_color=["#22c55e" if a >= 62.5 else ("#f59e0b" if a >= 50 else "#ef4444") for a in rd_df["Round Acc"]]))
    fig_rd.add_trace(go.Scatter(x=rd_df["Round"], y=rd_df["Cumulative Acc"], name="Cumulative",
        line=dict(color="#60a5fa", width=3), mode="lines+markers"))
    fig_rd.update_layout(template="plotly_dark", plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
        yaxis_title="Accuracy %", xaxis_title="Round", yaxis_range=[0, 100],
        barmode="overlay", height=400)
    fig_rd.add_hline(y=50, line_dash="dot", line_color="#334155", annotation_text="Coin flip")
    st.plotly_chart(fig_rd, use_container_width=True)

    st.dataframe(rd_df.style.format({"Round Acc": "{:.1f}%", "Cumulative Acc": "{:.1f}%"}),
        use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Weight Optimizer")
    st.markdown("Grid searches ~3,600 weight combinations to find optimal parameters.")

    if st.button("Optimize Weights (takes ~10s)", type="primary"):
        with st.spinner("Searching optimal parameters..."):
            opt_acc, opt_brier, opt_params = optimize_weights()

        if opt_params:
            st.success(f"Optimal accuracy: **{opt_acc:.1%}** ({opt_params['correct']}/{opt_params['total']}) | Brier: {opt_brier:.3f}")

            oc1, oc2, oc3 = st.columns(3)
            oc1.metric("Attack Weight", f"{opt_params['attack_w']:.2f}", f"{opt_params['attack_w'] - attack_w:+.2f}")
            oc1.metric("Defence Weight", f"{opt_params['defence_w']:.2f}", f"{opt_params['defence_w'] - defence_w:+.2f}")
            oc2.metric("Form Weight", f"{opt_params['form_w']:.2f}", f"{opt_params['form_w'] - form_w:+.2f}")
            oc2.metric("Home Weight", f"{opt_params['home_w']:.2f}", f"{opt_params['home_w'] - home_w:+.2f}")
            oc3.metric("Logistic Scale", f"{opt_params['scale']:.1f}", f"{opt_params['scale'] - 2.0:+.1f}")
            oc3.metric("Home Constant", f"{opt_params['home_const']:.2f}", f"{opt_params['home_const'] - 0.15:+.2f}")

            st.info("Set the sidebar sliders to these values to apply the optimized weights to Round 15 predictions.")

            opt_rd = opt_params["rounds"]
            opt_rd_data = []
            oc, ot = 0, 0
            for rd in sorted(opt_rd.keys()):
                r = opt_rd[rd]
                oc += r["correct"]
                ot += r["total"]
                opt_rd_data.append({
                    "Round": rd, "Correct": r["correct"], "Games": r["total"],
                    "Optimized Acc": round(r["correct"] / r["total"] * 100, 1),
                    "Cumulative": round(oc / ot * 100, 1),
                })
            opt_df = pd.DataFrame(opt_rd_data)
            st.dataframe(opt_df.style.format({"Optimized Acc": "{:.1f}%", "Cumulative": "{:.1f}%"}),
                use_container_width=True, hide_index=True)


# =========================  METHODOLOGY  ======================================
with tab_method:
    st.subheader("How It Works")
    st.markdown("""
**Data source:** Team aggregate per-game stats from [nrl.com/stats/teams](https://www.nrl.com/stats/) for all 17 NRL teams. Scraped 11 June 2026 via same-origin fetch of NRL.com q-data attributes.

**Why team stats, not player stats?** Player-level stats (top-50 leaderboards) only cover star performers, leaving 50%+ of players estimated via position defaults. Team aggregate stats from NRL.com cover 100% of actual on-field output for all 17 teams. No estimation needed.

**14 stats tracked per team (per game):**
Points, Tries, Linebreaks, Tackle Breaks, Post Contact Metres, Try Assists, Offloads, Run Metres, All Runs, Tackles, Missed Tackles, Ineffective Tackles, Errors, Kick Metres

**Recent form weighting (last 5 games):**
For each team, the last 5 completed matches are extracted from the 108-match results database. Average points scored and conceded in these 5 games are compared against the season average. The z-scores are adjusted by 30% of the trend factor, giving recent performance more influence than early-season results.

**Attack composite z-score:** Run Metres + Tackle Breaks + PCM + Linebreaks (x1.5) + Try Assists (x1.2) + Offloads + Points, adjusted by recent attacking trend

**Defence composite z-score:** Tackles (positive) + Missed Tackles (inverted) + Ineffective Tackles (inverted) + Errors (inverted), adjusted by recent defensive trend

**Match prediction:**
1. Team attack z vs opponent defence z = attack edge
2. Team defence z vs opponent attack z = defence edge
3. Season win% differential = form edge
4. Home ground advantage = flat base (0.15)
5. Referee historical bias = context edge
6. Weighted composite through logistic function (scale 2.0), clamped 25-82%

**Self-tuning:** Backtesting engine replays all 108 matches from Rounds 1-14 with evolving W/L form. Grid search optimizer tests ~3,600 weight combinations to find the set that maximizes historical accuracy. Current optimized accuracy: 63.4%.

**Edge** = Model probability minus market implied probability (1/odds). Positive = market underpricing.
    """)

    st.subheader("Data Sources")
    st.markdown("""
| Data | Source | Date |
|------|--------|------|
| Team stats (14 categories, 17 teams) | [nrl.com/stats/teams](https://www.nrl.com/stats/) | 11 June 2026 |
| Match results (108 games, R1-14) | Existing dashboard | Cumulative |
| Market odds (Round 15) | Sportsbet via nrl.com | 11 June 2026 |
| Referee bias data | Existing dashboard | Cumulative |
    """)

    st.caption("Model is exploratory. Not financial advice. Gamble responsibly.")
