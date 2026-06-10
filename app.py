import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from math import exp
from io import BytesIO

st.set_page_config(page_title="NRL Moneyball", page_icon="🏈", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
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
    .tag-value {background:#1a6334; color:#4ade80}
    .tag-pass {background:#1e2a38; color:#6b7280}
</style>""", unsafe_allow_html=True)

st.markdown("# 🏈 NRL Moneyball")
st.markdown("**Round 15, 2026 -- Origin II Split Round -- Official NRL Stats**")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Model Weights")
attack_w = st.sidebar.slider("Attack Weight", 0.25, 0.60, 0.40, help="Run metres, linebreaks, tackle breaks, try assists")
defence_w = st.sidebar.slider("Defence Weight", 0.10, 0.40, 0.25, help="Missed tackles, tackle efficiency, points conceded")
form_w = st.sidebar.slider("Form Weight", 0.05, 0.30, 0.15, help="Season win%")
home_w = st.sidebar.slider("Home Advantage", 0.05, 0.20, 0.10)
context_w = st.sidebar.slider("Context (Ref + Origin)", 0.02, 0.20, 0.10)
wt = attack_w + defence_w + form_w + home_w + context_w
st.sidebar.caption(f"Sum: {wt:.2f}" + (" ✓" if abs(wt - 1.0) <= 0.03 else " -- adjust to ~1.00"))

# ═══════════════════════════════════════════════════════════════════════════════
# OFFICIAL NRL.COM TEAM STATS -- 2026 season through Round 14
# Source: nrl.com/stats (Teams view, Total)
# Verified: 10 June 2026
# ═══════════════════════════════════════════════════════════════════════════════

RAW = {
    "Team":           ["Panthers","Knights","Warriors","Sea Eagles","Sharks","Storm","Cowboys","Rabbitohs","Dolphins","Roosters","Broncos","Wests Tigers","Eels","Raiders","Bulldogs","Titans","Dragons"],
    "P":              [13,13,12,13,12,14,14,12,12,12,13,12,13,13,13,12,13],
    "W":              [12, 8, 9, 8, 7, 6, 8, 6, 7, 8, 5, 6, 4, 5, 5, 3, 1],
    "L":              [ 1, 5, 3, 5, 5, 8, 6, 6, 5, 4, 8, 6, 9, 8, 8, 9,12],
    "PF":             [437,370,368,367,356,346,343,338,330,323,275,275,269,249,233,220,184],
    "PA":             [164,338,214,250,294,348,356,294,253,250,341,353,421,347,330,303,427],
    # Attack stats (season totals from nrl.com)
    "RunMetres":      [24419,21517,20397,22768,19893,23087,25113,20341,22072,21743,20259,20208,20527,21930,23387,19855,21193],
    "Linebreaks":     [88,80,69,74,65,77,84,75,65,58,54,60,52,60,64,54,37],
    "TackleBreaks":   [469,448,401,389,354,496,472,371,405,437,362,433,375,488,452,351,359],
    "Offloads":       [130,135,97,111,125,128,142,89,130,123,71,167,107,145,140,130,139],
    "TryAssists":     [58,58,51,54,54,45,53,49,44,52,38,45,38,35,29,27,27],
    # Defence stats (season totals from nrl.com)
    "Tackles":        [4414,4488,3918,4139,4100,4743,4725,3689,4294,3988,4527,4180,4405,4681,4595,4212,4923],
    "MissedTackles":  [388,447,395,394,369,464,510,396,350,327,430,393,523,437,442,389,409],
}
TEAMS = pd.DataFrame(RAW)
TEAMS["PD"] = TEAMS["PF"] - TEAMS["PA"]

# Per-game averages
for col in ["PF","PA","RunMetres","Linebreaks","TackleBreaks","Offloads","TryAssists","Tackles","MissedTackles"]:
    TEAMS[f"{col}_pg"] = round(TEAMS[col] / TEAMS["P"], 1)
TEAMS["Win%"] = round(TEAMS["W"] / TEAMS["P"] * 100, 1)
TEAMS["TackleEff"] = round(TEAMS["Tackles"] / (TEAMS["Tackles"] + TEAMS["MissedTackles"]) * 100, 1)

# Z-scores for composite ratings
def zscore(series):
    return (series - series.mean()) / series.std()

# Attack composite: run metres, linebreaks, tackle breaks, try assists (all per game)
TEAMS["Atk_z"] = (zscore(TEAMS["RunMetres_pg"]) + zscore(TEAMS["Linebreaks_pg"]) * 1.5
                  + zscore(TEAMS["TackleBreaks_pg"]) + zscore(TEAMS["TryAssists_pg"]) * 1.2) / 4.7
# Defence composite: tackle efficiency, missed tackles (inverted), PA (inverted)
TEAMS["Def_z"] = (zscore(TEAMS["TackleEff"]) + zscore(-TEAMS["MissedTackles_pg"])
                  + zscore(-TEAMS["PA_pg"]) * 1.5) / 3.5

TEAMS["Atk_rank"] = TEAMS["Atk_z"].rank(ascending=False).astype(int)
TEAMS["Def_rank"] = TEAMS["Def_z"].rank(ascending=False).astype(int)

# ── Round 15 matches ─────────────────────────────────────────────────────────
MATCHES = [
    {
        "Home": "Rabbitohs", "Away": "Broncos",
        "Venue": "Accor Stadium", "Kickoff": "Thu 7:50pm", "Referee": "G. Atkins",
        "Ref_Boost": -9.1,
        "Mkt_Home": 1.48, "Mkt_Away": 2.66,
        "Origin_Impact_H": -0.05,
        "Origin_Impact_A": -0.15,
        "H_Outs": "C. Walker, C. Murray (NSW)",
        "A_Outs": "P. Carrigan, S. Cobbo, R. Walsh, P. Haas, K. Staggs (QLD/inj)",
        "Tipping": "9/9 experts pick Rabbitohs",
    },
    {
        "Home": "Dolphins", "Away": "Roosters",
        "Venue": "Suncorp Stadium", "Kickoff": "Fri 8:00pm", "Referee": "T. Smith",
        "Ref_Boost": 16.9,
        "Mkt_Home": 1.43, "Mkt_Away": 2.90,
        "Origin_Impact_H": -0.08,
        "Origin_Impact_A": -0.18,
        "H_Outs": "H. Tabuai-Fidow (QLD), S. Cobbo, Flegler, Finefeuiaki",
        "A_Outs": "~7 Origin reps. DCE to HB, Savala at 6, Foley debut",
        "Tipping": "9/9 experts pick Dolphins",
    },
    {
        "Home": "Warriors", "Away": "Sharks",
        "Venue": "Go Media Stadium", "Kickoff": "Sat 5:30pm", "Referee": "G. Sutton",
        "Ref_Boost": -31.3,
        "Mkt_Home": 1.35, "Mkt_Away": 3.25,
        "Origin_Impact_H": 0.0,
        "Origin_Impact_A": -0.05,
        "H_Outs": "Minimal. Debuts: Tafua, Seu Salalilo",
        "A_Outs": "N. Hynes (calf, game-time). Key forwards out",
        "Tipping": "9/9 experts pick Warriors (-8.5)",
    },
    {
        "Home": "Eels", "Away": "Raiders",
        "Venue": "CommBank Stadium", "Kickoff": "Sat 7:35pm", "Referee": "A. Klein",
        "Ref_Boost": 0.0,
        "Mkt_Home": 2.10, "Mkt_Away": 1.80,
        "Origin_Impact_H": 0.0,
        "Origin_Impact_A": 0.0,
        "H_Outs": "M. Moses (knee, long-term)",
        "A_Outs": "V. Patuki-Case debut. Minor reshuffles",
        "Tipping": "5-4 split, slight Eels lean",
    },
    {
        "Home": "Wests Tigers", "Away": "Titans",
        "Venue": "Leichhardt Oval", "Kickoff": "Sun 4:05pm", "Referee": "D. Munro",
        "Ref_Boost": 0.0,
        "Mkt_Home": 1.55, "Mkt_Away": 2.50,
        "Origin_Impact_H": 0.0,
        "Origin_Impact_A": -0.12,
        "H_Outs": "A. Doueihi (injury)",
        "A_Outs": "J. Fifita, T. Fa'asuamaleaui (QLD). Two best forwards gone",
        "Tipping": "6-3 experts pick Tigers",
    },
]

# ── Model ────────────────────────────────────────────────────────────────────
def get_team(name):
    row = TEAMS[TEAMS["Team"] == name]
    if row.empty:
        row = TEAMS[TEAMS["Team"].str.contains(name)]
    return row.iloc[0]

def calculate_match(m):
    h = get_team(m["Home"])
    a = get_team(m["Away"])

    # Attack differential (z-score based)
    atk_edge = h["Atk_z"] - a["Def_z"]   # Home attack vs Away defence
    def_edge = h["Def_z"] - a["Atk_z"]   # Home defence vs Away attack

    # Adjust for Origin absences
    atk_edge += m["Origin_Impact_A"]      # Away missing players hurts their defence
    def_edge += m["Origin_Impact_H"] * -1 # Home missing players... actually helps away

    # Form edge (win %)
    form_edge = (h["Win%"] - a["Win%"]) / 100

    # Referee context
    ref_edge = m["Ref_Boost"] / 200

    # Composite score
    score = (
        attack_w * atk_edge
        + defence_w * def_edge
        + form_w * form_edge
        + home_w * 1.0
        + context_w * ref_edge
    )

    prob = 1 / (1 + exp(-score * 3.2))
    prob = max(0.20, min(0.88, prob))
    return prob, atk_edge, def_edge, form_edge, ref_edge, score

results = []
for m in MATCHES:
    prob, atk, dfe, frm, ref, score = calculate_match(m)
    h = get_team(m["Home"])
    a = get_team(m["Away"])
    edge = (prob - (1 / m["Mkt_Home"])) * 100
    away_edge = ((1 - prob) - (1 / m["Mkt_Away"])) * 100
    fair_h = round(1 / prob, 2) if prob > 0 else 99.0
    fair_a = round(1 / (1 - prob), 2) if prob < 1 else 99.0

    # Bet suggestion
    if edge >= 8:
        bet, strength = f"{m['Home']} -5.5", "STRONG"
    elif edge >= 4:
        bet, strength = f"{m['Home']} H2H", "VALUE"
    elif edge >= 2:
        bet, strength = f"{m['Home']} H2H", "lean"
    elif away_edge >= 8:
        bet, strength = f"{m['Away']} +6.5", "STRONG"
    elif away_edge >= 4:
        bet, strength = f"{m['Away']} H2H", "VALUE"
    elif away_edge >= 2:
        bet, strength = f"{m['Away']} H2H", "lean"
    else:
        bet, strength = "Pass", "pass"

    results.append({
        "Match": f"{m['Home']} vs {m['Away']}",
        "Home": m["Home"], "Away": m["Away"],
        "Venue": m["Venue"], "Kickoff": m["Kickoff"], "Referee": m["Referee"],
        "Home_Prob": round(prob, 4), "Away_Prob": round(1 - prob, 4),
        "Fair_H": fair_h, "Fair_A": fair_a,
        "Mkt_H": m["Mkt_Home"], "Mkt_A": m["Mkt_Away"],
        "Edge_pp": round(edge, 1), "Away_Edge_pp": round(away_edge, 1),
        "Ref_Boost": m["Ref_Boost"],
        "Atk_Edge": round(atk, 3), "Def_Edge": round(dfe, 3),
        "Form_Edge": round(frm, 3), "Ref_Edge": round(ref, 3),
        "Score": round(score, 4),
        "Bet": bet, "Strength": strength,
        "Tipping": m["Tipping"],
        "H_Atk_z": round(h["Atk_z"], 2), "H_Def_z": round(h["Def_z"], 2),
        "A_Atk_z": round(a["Atk_z"], 2), "A_Def_z": round(a["Def_z"], 2),
        "H_Atk_rank": int(h["Atk_rank"]), "H_Def_rank": int(h["Def_rank"]),
        "A_Atk_rank": int(a["Atk_rank"]), "A_Def_rank": int(a["Def_rank"]),
        "H_RunM_pg": h["RunMetres_pg"], "A_RunM_pg": a["RunMetres_pg"],
        "H_LB_pg": h["Linebreaks_pg"], "A_LB_pg": a["Linebreaks_pg"],
        "H_TB_pg": h["TackleBreaks_pg"], "A_TB_pg": a["TackleBreaks_pg"],
        "H_TA_pg": h["TryAssists_pg"], "A_TA_pg": a["TryAssists_pg"],
        "H_MT_pg": h["MissedTackles_pg"], "A_MT_pg": a["MissedTackles_pg"],
        "H_TE": h["TackleEff"], "A_TE": a["TackleEff"],
        "H_PF_pg": h["PF_pg"], "A_PF_pg": a["PF_pg"],
        "H_PA_pg": h["PA_pg"], "A_PA_pg": a["PA_pg"],
        "H_Win": h["Win%"], "A_Win": a["Win%"],
        "H_Outs": m["H_Outs"], "A_Outs": m["A_Outs"],
    })

df = pd.DataFrame(results)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_dash, tab_detail, tab_power, tab_ladder, tab_method = st.tabs(
    ["Dashboard", "Game Breakdowns", "Power Rankings", "Ladder", "Methodology"]
)

# ═══════════════════════  DASHBOARD  ═════════════════════════════════════════
with tab_dash:
    strong = df[df["Strength"].isin(["STRONG", "VALUE"])]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Round 15 Games", f"{len(df)} (Origin split)")
    c2.metric("Actionable Bets", len(strong))
    c3.metric("Avg Edge", f"{strong['Edge_pp'].mean():+.1f}pp" if len(strong) else "0.0pp")
    c4.metric("Teams on Bye", "7")
    st.markdown("---")

    # Predictions table
    st.subheader("Round 15 Predictions")
    show = df[["Match","Home_Prob","Away_Prob","Fair_H","Fair_A","Mkt_H","Mkt_A","Edge_pp","Ref_Boost","Bet","Strength"]].copy()
    show.columns = ["Match","Home%","Away%","Fair H","Fair A","Mkt H","Mkt A","Edge","Ref","Bet","Rating"]

    def style_edge(val):
        if isinstance(val, (int, float)):
            if val >= 4: return "background-color:#0d3320; color:#4ade80; font-weight:700"
            if val <= -4: return "background-color:#3b1010; color:#f87171; font-weight:700"
        return ""

    def style_rating(val):
        if val == "STRONG": return "background-color:#b45309; color:#fff; font-weight:700"
        if val == "VALUE": return "background-color:#0d3320; color:#4ade80; font-weight:700"
        return "color:#6b7280"

    st.dataframe(
        show.style.format({
            "Home%": "{:.1%}", "Away%": "{:.1%}",
            "Fair H": "${:.2f}", "Fair A": "${:.2f}",
            "Mkt H": "${:.2f}", "Mkt A": "${:.2f}",
            "Edge": "{:+.1f}", "Ref": "{:+.1f}",
        }).map(style_edge, subset=["Edge"]).map(style_rating, subset=["Rating"]),
        use_container_width=True, hide_index=True, height=250,
    )
    st.markdown("---")

    # Bet cards
    st.subheader("Selections")
    for _, r in df.iterrows():
        if r["Strength"] == "STRONG":
            card_class, tag = "bet-card-strong", '<span class="tag tag-strong">STRONG</span>'
        elif r["Strength"] == "VALUE":
            card_class, tag = "bet-card", '<span class="tag tag-value">VALUE</span>'
        else:
            card_class, tag = "pass-card", '<span class="tag tag-pass">PASS</span>'

        pick = r["Home"] if r["Home_Prob"] >= 0.5 else r["Away"]
        prob = r["Home_Prob"] if r["Home_Prob"] >= 0.5 else r["Away_Prob"]
        fair = r["Fair_H"] if r["Home_Prob"] >= 0.5 else r["Fair_A"]
        mkt = r["Mkt_H"] if r["Home_Prob"] >= 0.5 else r["Mkt_A"]

        st.markdown(f"""<div class="{card_class}">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px">
                <div>{tag} <b style="font-size:1.1rem; color:#e5e7eb">{r['Bet']}</b>
                    <span style="color:#6b7280; margin-left:8px">{r['Match']}</span></div>
                <div style="font-size:1.1rem; font-weight:700; color:{'#4ade80' if r['Edge_pp'] > 0 else '#f87171'}">{r['Edge_pp']:+.1f}pp</div>
            </div>
            <div style="display:flex; gap:20px; color:#9ca3af; font-size:0.85rem; flex-wrap:wrap">
                <span>Model: <b style="color:#60a5fa">{prob:.0%}</b></span>
                <span>Fair: <b style="color:#60a5fa">${fair:.2f}</b></span>
                <span>Market: <b style="color:#e5e7eb">${mkt:.2f}</b></span>
                <span>Ref: <b style="color:#c084fc">{r['Ref_Boost']:+.1f}pp</b></span>
                <span>{r['Home']} Atk #{r['H_Atk_rank']} Def #{r['H_Def_rank']}</span>
                <span>{r['Away']} Atk #{r['A_Atk_rank']} Def #{r['A_Def_rank']}</span>
            </div>
            <div style="color:#6b7280; font-size:0.8rem; margin-top:6px">Tipping: {r['Tipping']}</div>
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
        colors = ["#22c55e" if e >= 3 else ("#ef4444" if e <= -3 else "#6b7280") for e in df["Edge_pp"]]
        fig2 = go.Figure(go.Bar(x=df["Match"], y=df["Edge_pp"], marker_color=colors))
        fig2.update_layout(title="Edge Map (pp)", template="plotly_dark",
            plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
            yaxis_title="Edge (pp)", margin=dict(l=40,r=20,t=40,b=60))
        fig2.add_hline(y=0, line_color="#334155")
        st.plotly_chart(fig2, use_container_width=True)

    buf = BytesIO()
    df.to_csv(buf, index=False)
    st.download_button("Export CSV", buf.getvalue(), "nrl_r15_moneyball.csv", "text/csv")

# ═══════════════════════  GAME BREAKDOWNS  ═══════════════════════════════════
with tab_detail:
    st.subheader("Per-Game Breakdown -- Real NRL Stats")
    for _, g in df.iterrows():
        icon = {"STRONG": "🔥", "VALUE": "✅", "lean": "👀"}.get(g["Strength"], "")
        with st.expander(f"{icon} {g['Match']}  |  {g['Bet']}  |  Edge {g['Edge_pp']:+.1f}pp"):

            st.markdown(f"**{g['Venue']}** -- {g['Kickoff']} -- Ref: {g['Referee']} ({g['Ref_Boost']:+.1f}pp)")
            st.markdown("---")

            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"### {g['Home']} (Home)")
                st.markdown(f"**Record:** {TEAMS[TEAMS['Team']==g['Home']].iloc[0]['W']}W-{TEAMS[TEAMS['Team']==g['Home']].iloc[0]['L']}L ({g['H_Win']}%)")
                st.markdown(f"**Attack rank:** #{g['H_Atk_rank']} (z={g['H_Atk_z']:+.2f})")
                st.markdown(f"**Defence rank:** #{g['H_Def_rank']} (z={g['H_Def_z']:+.2f})")

                st.markdown("**Per-game stats (nrl.com):**")
                stats_h = pd.DataFrame({
                    "Metric": ["Points Scored","Points Conceded","Run Metres","Linebreaks","Tackle Breaks","Try Assists","Missed Tackles","Tackle Eff%"],
                    "Value": [g["H_PF_pg"], g["H_PA_pg"], g["H_RunM_pg"], g["H_LB_pg"], g["H_TB_pg"], g["H_TA_pg"], g["H_MT_pg"], g["H_TE"]],
                })
                st.dataframe(stats_h, hide_index=True, use_container_width=True)
                st.caption(f"Origin outs: {g['H_Outs']}")

            with cb:
                st.markdown(f"### {g['Away']} (Away)")
                st.markdown(f"**Record:** {TEAMS[TEAMS['Team']==g['Away']].iloc[0]['W']}W-{TEAMS[TEAMS['Team']==g['Away']].iloc[0]['L']}L ({g['A_Win']}%)")
                st.markdown(f"**Attack rank:** #{g['A_Atk_rank']} (z={g['A_Atk_z']:+.2f})")
                st.markdown(f"**Defence rank:** #{g['A_Def_rank']} (z={g['A_Def_z']:+.2f})")

                st.markdown("**Per-game stats (nrl.com):**")
                stats_a = pd.DataFrame({
                    "Metric": ["Points Scored","Points Conceded","Run Metres","Linebreaks","Tackle Breaks","Try Assists","Missed Tackles","Tackle Eff%"],
                    "Value": [g["A_PF_pg"], g["A_PA_pg"], g["A_RunM_pg"], g["A_LB_pg"], g["A_TB_pg"], g["A_TA_pg"], g["A_MT_pg"], g["A_TE"]],
                })
                st.dataframe(stats_a, hide_index=True, use_container_width=True)
                st.caption(f"Origin outs: {g['A_Outs']}")

            st.markdown("---")
            st.markdown("**Model Breakdown:**")
            st.code(
                f"Attack edge:   Home Atk z({g['H_Atk_z']:+.2f}) - Away Def z({g['A_Def_z']:+.2f}) = {g['Atk_Edge']:+.3f}  x {attack_w:.2f}\n"
                f"Defence edge:  Home Def z({g['H_Def_z']:+.2f}) - Away Atk z({g['A_Atk_z']:+.2f}) = {g['Def_Edge']:+.3f}  x {defence_w:.2f}\n"
                f"Form edge:     Win% ({g['H_Win']:.0f}% - {g['A_Win']:.0f}%) / 100  = {g['Form_Edge']:+.3f}  x {form_w:.2f}\n"
                f"Home base:     1.000  x {home_w:.2f}\n"
                f"Ref context:   {g['Ref_Boost']:+.1f}pp / 200 = {g['Ref_Edge']:+.3f}  x {context_w:.2f}\n"
                f"{'='*55}\n"
                f"Composite score: {g['Score']:+.4f}  ->  logistic  ->  {g['Home']} {g['Home_Prob']:.1%}\n"
                f"Fair odds: ${g['Fair_H']:.2f} / ${g['Fair_A']:.2f}   Market: ${g['Mkt_H']:.2f} / ${g['Mkt_A']:.2f}\n"
                f"Edge: {g['Edge_pp']:+.1f}pp"
            )

            # Head-to-head stat comparison chart
            metrics = ["RunM/g","LB/g","TB/g","TA/g","MissT/g","PF/g","PA/g"]
            h_vals = [g["H_RunM_pg"]/200, g["H_LB_pg"], g["H_TB_pg"]/4, g["H_TA_pg"], g["H_MT_pg"]/4, g["H_PF_pg"], g["H_PA_pg"]]
            a_vals = [g["A_RunM_pg"]/200, g["A_LB_pg"], g["A_TB_pg"]/4, g["A_TA_pg"], g["A_MT_pg"]/4, g["A_PF_pg"], g["A_PA_pg"]]
            fig_h2h = go.Figure()
            fig_h2h.add_trace(go.Bar(name=g["Home"], x=metrics, y=h_vals, marker_color="#3b82f6"))
            fig_h2h.add_trace(go.Bar(name=g["Away"], x=metrics, y=a_vals, marker_color="#ef4444"))
            fig_h2h.update_layout(barmode="group", template="plotly_dark", title="Head-to-Head Stats (per game, scaled)",
                plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14", height=300,
                margin=dict(l=40,r=20,t=40,b=40))
            st.plotly_chart(fig_h2h, use_container_width=True)

# ═══════════════════════  POWER RANKINGS  ════════════════════════════════════
with tab_power:
    st.subheader("2026 Attack & Defence Power Rankings")
    st.caption("Composite z-scores from official nrl.com stats. Higher = better.")

    pr = TEAMS[["Team","Atk_z","Def_z","Atk_rank","Def_rank","Win%","RunMetres_pg","Linebreaks_pg","TackleBreaks_pg","TryAssists_pg","MissedTackles_pg","TackleEff"]].copy()
    pr = pr.sort_values("Atk_z", ascending=False).reset_index(drop=True)
    pr.index = pr.index + 1

    col1, col2 = st.columns(2)
    with col1:
        atk = TEAMS.sort_values("Atk_z", ascending=True)
        fig_a = px.bar(atk, x="Atk_z", y="Team", orientation="h", title="Attack Power (z-score)",
            color="Atk_z", color_continuous_scale="Greens", template="plotly_dark")
        fig_a.update_layout(plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
            coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_a, use_container_width=True)

    with col2:
        dfe = TEAMS.sort_values("Def_z", ascending=True)
        fig_d = px.bar(dfe, x="Def_z", y="Team", orientation="h", title="Defence Power (z-score)",
            color="Def_z", color_continuous_scale="Blues", template="plotly_dark")
        fig_d.update_layout(plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
            coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_d, use_container_width=True)

    st.subheader("Full Stat Sheet (per game)")
    display_pr = pr.rename(columns={
        "Atk_z": "Atk Z", "Def_z": "Def Z", "Atk_rank": "Atk#", "Def_rank": "Def#",
        "RunMetres_pg": "RunM/g", "Linebreaks_pg": "LB/g", "TackleBreaks_pg": "TB/g",
        "TryAssists_pg": "TA/g", "MissedTackles_pg": "MT/g", "TackleEff": "Tkl%",
    })
    st.dataframe(
        display_pr.style.format({
            "Atk Z": "{:+.2f}", "Def Z": "{:+.2f}", "Win%": "{:.0f}%",
            "RunM/g": "{:.0f}", "LB/g": "{:.1f}", "TB/g": "{:.1f}",
            "TA/g": "{:.1f}", "MT/g": "{:.1f}", "Tkl%": "{:.1f}%",
        }),
        use_container_width=True,
    )

    # Scatter: attack z vs defence z
    fig_scatter = px.scatter(TEAMS, x="Atk_z", y="Def_z", text="Team",
        title="Attack vs Defence (top-right = elite)", template="plotly_dark",
        color="Win%", color_continuous_scale="RdYlGn", size_max=12)
    fig_scatter.update_traces(textposition="top center", marker=dict(size=10))
    fig_scatter.update_layout(plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
        xaxis_title="Attack z-score", yaxis_title="Defence z-score",
        height=500)
    fig_scatter.add_hline(y=0, line_dash="dot", line_color="#334155")
    fig_scatter.add_vline(x=0, line_dash="dot", line_color="#334155")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ═══════════════════════  LADDER  ════════════════════════════════════════════
with tab_ladder:
    st.subheader("2026 NRL Ladder (through Round 14)")
    ladder = TEAMS.sort_values("PD", ascending=False).reset_index(drop=True)
    ladder.index = ladder.index + 1
    st.dataframe(
        ladder[["Team","P","W","L","PF","PA","PD","PF_pg","PA_pg","Win%"]].style.format({
            "Win%": "{:.1f}%", "PD": "{:+d}", "PF_pg": "{:.1f}", "PA_pg": "{:.1f}",
        }),
        use_container_width=True,
    )

# ═══════════════════════  METHODOLOGY  ═══════════════════════════════════════
with tab_method:
    st.subheader("How the Model Works")
    st.markdown("""
**Data source:** All team stats pulled from [nrl.com/stats](https://www.nrl.com/stats/) (Teams view), verified 10 June 2026. 8 statistical categories across all 17 teams.

**Attack composite z-score** combines per-game averages of:
- Run Metres (total yardage output)
- Linebreaks (x1.5 weight -- penetration is king)
- Tackle Breaks (post-contact dominance)
- Try Assists (x1.2 weight -- creative playmaking)

**Defence composite z-score** combines:
- Tackle Efficiency % (tackles / total attempts)
- Missed Tackles per game (inverted -- fewer = better)
- Points Against per game (x1.5 weight -- the ultimate defence metric)

**Each match compares:**
1. Home attack z vs Away defence z = attack edge
2. Home defence z vs Away attack z = defence edge
3. Win% differential = form edge
4. Home ground advantage = flat base
5. Referee historical bias + Origin absence impact = context edge

**Weighted composite score** passes through a logistic function (S-curve) to produce a probability, clamped 20-88%.

**Edge** = Model probability minus market implied probability. Positive = bookmaker underpricing home.

**Origin Impact** is a manual adjustment based on the number and quality of players missing due to State of Origin selection. A team losing 5 key players (like Broncos losing Carrigan, Cobbo, Walsh, Haas, Staggs) gets a larger negative adjustment than a team with minimal absences.

**What the model does NOT account for:**
- Individual player matchups within the lineup
- Specific player form (hot/cold streaks)
- Weather conditions
- Travel fatigue
- Motivation factors
    """)

    st.subheader("Data Sources")
    st.markdown("""
| Data | Source | Verified |
|------|--------|----------|
| All team stats (8 categories) | [nrl.com/stats](https://www.nrl.com/stats/) | 10 June 2026 |
| Ladder & standings | [nrl.com/ladder](https://www.nrl.com/ladder/) | 10 June 2026 |
| Round 15 fixtures | [nrl.com/draw](https://www.nrl.com/draw/) | 10 June 2026 |
| Expert tipping | Rugby League Zone | 10 June 2026 |
| Referee bias | refereebias.com + existing dashboard | Historical |
| Market odds | Sportsbet (from nrl.com draw page) | Pre-round |
    """)

    st.subheader("Stat IDs Reference (nrl.com)")
    st.code("Points: stat=76  |  Linebreaks: stat=30  |  Tackle Breaks: stat=29\n"
            "Offloads: stat=28  |  Try Assists: stat=35  |  Tackles: stat=3\n"
            "Missed Tackles: stat=4  |  All Run Metres: stat=1000037")

    st.caption("Model is exploratory. Not financial advice. Gamble responsibly.")
