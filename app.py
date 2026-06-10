import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from math import exp
from io import BytesIO

st.set_page_config(page_title="NRL Moneyball", page_icon="🏈", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────────────────────
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
st.markdown("**Round 15, 2026 - Origin II Split Round - Real Stats - Real Edge**")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Model Weights")
attack_w = st.sidebar.slider("Attack (PF + scoring)", 0.30, 0.70, 0.45)
defence_w = st.sidebar.slider("Defence (PA + conceding)", 0.10, 0.40, 0.25)
home_w = st.sidebar.slider("Home Advantage", 0.05, 0.25, 0.12)
context_w = st.sidebar.slider("Context (Ref + Origin)", 0.05, 0.30, 0.18)
wt = attack_w + defence_w + home_w + context_w
st.sidebar.caption(f"Sum: {wt:.2f}" + (" ✓" if abs(wt - 1.0) <= 0.03 else " -- adjust to 1.00"))

st.sidebar.header("Adjustments")
home_yardage_adj = st.sidebar.slider("Home scoring reduction %", 0, 30, 0,
    help="Reduce home attack for key absences (e.g. halfback out = 10-15%)")
away_yardage_adj = st.sidebar.slider("Away scoring reduction %", 0, 30, 0)

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFIED 2026 DATA (Sources: Zero Tackle ladder, Aus Sports Tipping averages)
# All stats through Round 14 - verified 10 June 2026
# ═══════════════════════════════════════════════════════════════════════════════

# Full 2026 ladder + scoring averages (all 17 teams)
LADDER = pd.DataFrame([
    {"Team": "Panthers",    "P": 13, "W": 12, "L": 1, "PF": 437, "PA": 164, "PD": 273, "Avg_PF": 33.6, "Avg_PA": 12.6, "Form": "WWWWW"},
    {"Team": "Warriors",    "P": 12, "W": 9,  "L": 3, "PF": 368, "PA": 214, "PD": 154, "Avg_PF": 30.7, "Avg_PA": 17.8, "Form": "LWWWL"},
    {"Team": "Roosters",    "P": 12, "W": 8,  "L": 4, "PF": 323, "PA": 250, "PD": 73,  "Avg_PF": 26.9, "Avg_PA": 20.8, "Form": "WWWLW"},
    {"Team": "Sea Eagles",  "P": 13, "W": 8,  "L": 5, "PF": 367, "PA": 250, "PD": 117, "Avg_PF": 28.2, "Avg_PA": 19.2, "Form": "WWLWW"},
    {"Team": "Dolphins",    "P": 12, "W": 7,  "L": 5, "PF": 330, "PA": 253, "PD": 77,  "Avg_PF": 27.5, "Avg_PA": 21.1, "Form": "WLWLW"},
    {"Team": "Sharks",      "P": 12, "W": 7,  "L": 5, "PF": 356, "PA": 294, "PD": 62,  "Avg_PF": 29.7, "Avg_PA": 24.5, "Form": "WLWWW"},
    {"Team": "Knights",     "P": 13, "W": 8,  "L": 5, "PF": 370, "PA": 338, "PD": 32,  "Avg_PF": 28.5, "Avg_PA": 26.0, "Form": "LWLWL"},
    {"Team": "Rabbitohs",   "P": 12, "W": 6,  "L": 6, "PF": 338, "PA": 294, "PD": 44,  "Avg_PF": 28.2, "Avg_PA": 24.5, "Form": "WWLWL"},
    {"Team": "Cowboys",     "P": 14, "W": 8,  "L": 6, "PF": 343, "PA": 356, "PD": -13, "Avg_PF": 24.5, "Avg_PA": 25.4, "Form": "WLWLW"},
    {"Team": "Tigers",      "P": 12, "W": 6,  "L": 6, "PF": 275, "PA": 353, "PD": -78, "Avg_PF": 22.9, "Avg_PA": 29.4, "Form": "LLLWL"},
    {"Team": "Storm",       "P": 14, "W": 6,  "L": 8, "PF": 346, "PA": 348, "PD": -2,  "Avg_PF": 24.7, "Avg_PA": 24.9, "Form": "LWLLW"},
    {"Team": "Broncos",     "P": 13, "W": 5,  "L": 8, "PF": 275, "PA": 341, "PD": -66, "Avg_PF": 21.2, "Avg_PA": 26.2, "Form": "LLWLL"},
    {"Team": "Bulldogs",    "P": 13, "W": 5,  "L": 8, "PF": 233, "PA": 330, "PD": -97, "Avg_PF": 17.9, "Avg_PA": 25.4, "Form": "WLWLL"},
    {"Team": "Raiders",     "P": 13, "W": 5,  "L": 8, "PF": 249, "PA": 347, "PD": -98, "Avg_PF": 19.2, "Avg_PA": 26.7, "Form": "WLWLW"},
    {"Team": "Titans",      "P": 12, "W": 3,  "L": 9, "PF": 220, "PA": 303, "PD": -83, "Avg_PF": 18.3, "Avg_PA": 25.3, "Form": "LWLLL"},
    {"Team": "Eels",        "P": 13, "W": 4,  "L": 9, "PF": 269, "PA": 421, "PD": -152,"Avg_PF": 20.7, "Avg_PA": 32.4, "Form": "LLWLL"},
    {"Team": "Dragons",     "P": 13, "W": 1,  "L": 12,"PF": 184, "PA": 427, "PD": -243,"Avg_PF": 14.2, "Avg_PA": 32.8, "Form": "LLLLL"},
])
LADDER["Win%"] = round(LADDER["W"] / LADDER["P"] * 100, 1)
LADDER["Avg_Diff"] = round(LADDER["Avg_PF"] - LADDER["Avg_PA"], 1)

# ── Round 15 matches (Origin II split round - 5 games, 7 teams on bye) ──────
ROUND_DATA = pd.DataFrame([
    {
        "Match": "Rabbitohs vs Broncos",
        "Home": "Rabbitohs", "Away": "Broncos",
        "Venue": "Accor Stadium", "Kickoff": "Thu 7:50pm", "Referee": "G. Atkins",
        "H_PF": 28.2, "H_PA": 24.5, "H_PD": 44,  "H_W": 6,  "H_L": 6,  "H_P": 12,
        "A_PF": 21.2, "A_PA": 26.2, "A_PD": -66, "A_W": 5,  "A_L": 8,  "A_P": 13,
        "Ref_Boost_Home": -9.1,
        "Mkt_Home": 1.48, "Mkt_Away": 2.75,
        "H_Origin_Outs": "Cody Walker (NSW), Cameron Murray (NSW)",
        "A_Origin_Outs": "Pat Carrigan (QLD), Selwyn Cobbo (QLD), Reece Walsh (QLD injured). Also missing Haas, Staggs",
        "H_Notes": "Strong forward pack (Tatola, Fifita, Hubner). 6th in attack at 28.2ppg.",
        "A_Notes": "Heavily depleted. 5 key players out. Season avg already poor at 21.2ppg.",
        "Tipping": "9/9 experts pick Rabbitohs",
    },
    {
        "Match": "Dolphins vs Roosters",
        "Home": "Dolphins", "Away": "Roosters",
        "Venue": "Suncorp Stadium", "Kickoff": "Fri 8:00pm", "Referee": "T. Smith",
        "H_PF": 27.5, "H_PA": 21.1, "H_PD": 77,  "H_W": 7,  "H_L": 5,  "H_P": 12,
        "A_PF": 26.9, "A_PA": 20.8, "A_PD": 73,  "A_W": 8,  "A_L": 4,  "A_P": 12,
        "Ref_Boost_Home": 16.9,
        "Mkt_Home": 1.43, "Mkt_Away": 2.90,
        "H_Origin_Outs": "Hammer Tabuai-Fidow (QLD), Selwyn Cobbo (QLD), Flegler, Finefeuiaki, Plath",
        "A_Origin_Outs": "~7 Origin reps out. DCE shifts to halfback, Hugo Savala at 6. Reece Foley debut.",
        "H_Notes": "5th in attack, 4th in defence. Strong at Suncorp. Ref boost +16.9pp.",
        "A_Notes": "3rd on ladder but decimated by Origin. Completely reshuffled spine.",
        "Tipping": "9/9 experts pick Dolphins",
    },
    {
        "Match": "Warriors vs Sharks",
        "Home": "Warriors", "Away": "Sharks",
        "Venue": "Go Media Stadium", "Kickoff": "Sat 5:30pm", "Referee": "G. Sutton",
        "H_PF": 30.7, "H_PA": 17.8, "H_PD": 154, "H_W": 9,  "H_L": 3,  "H_P": 12,
        "A_PF": 29.7, "A_PA": 24.5, "A_PD": 62,  "A_W": 7,  "A_L": 5,  "A_P": 12,
        "Ref_Boost_Home": -31.3,
        "Mkt_Home": 1.35, "Mkt_Away": 3.25,
        "H_Origin_Outs": "Debuts: Makaia Tafua, Jason Seu Salalilo",
        "A_Origin_Outs": "N. Hynes (calf, game-time decision). Key forwards missing.",
        "H_Notes": "2nd on ladder. 2nd best attack (30.7ppg), 2nd best defence (17.8ppg). Dominant at home.",
        "A_Notes": "6th. Good attack but concede 24.5ppg. Missing forwards hurts.",
        "Tipping": "9/9 experts pick Warriors (spread -8.5)",
    },
    {
        "Match": "Eels vs Raiders",
        "Home": "Eels", "Away": "Raiders",
        "Venue": "CommBank Stadium", "Kickoff": "Sat 7:35pm", "Referee": "A. Klein",
        "H_PF": 20.7, "H_PA": 32.4, "H_PD": -152,"H_W": 4,  "H_L": 9,  "H_P": 13,
        "A_PF": 19.2, "A_PA": 26.7, "A_PD": -98, "A_W": 5,  "A_L": 8,  "A_P": 13,
        "Ref_Boost_Home": 0.0,
        "Mkt_Home": 2.10, "Mkt_Away": 1.80,
        "H_Origin_Outs": "M. Moses (knee, out long-term). No major Origin losses.",
        "A_Origin_Outs": "Debut: Vena Patuki-Case. Minor reshuffles.",
        "H_Notes": "16th. Worst defence in the comp at 32.4ppg conceded. No Moses.",
        "A_Notes": "14th. Poor season but slightly better form. 5W-8L.",
        "Tipping": "Split 5-4, slight Eels lean (home advantage)",
    },
    {
        "Match": "Tigers vs Titans",
        "Home": "Tigers", "Away": "Titans",
        "Venue": "Leichhardt Oval", "Kickoff": "Sun 4:05pm", "Referee": "D. Munro",
        "H_PF": 22.9, "H_PA": 29.4, "H_PD": -78, "H_W": 6,  "H_L": 6,  "H_P": 12,
        "A_PF": 18.3, "A_PA": 25.3, "A_PD": -83, "A_W": 3,  "A_L": 9,  "A_P": 12,
        "Ref_Boost_Home": 0.0,
        "Mkt_Home": 1.55, "Mkt_Away": 2.50,
        "H_Origin_Outs": "A. Doueihi (injury). No major Origin losses.",
        "A_Origin_Outs": "Jojo Fifita (QLD), Tino Fa'asuamaleaui (QLD). Two best forwards gone.",
        "H_Notes": "10th. Improved this season (6-6). Home at Leichhardt is a fortress factor.",
        "A_Notes": "15th. Worst attack in comp at 18.3ppg. Lose 2 best forwards to Origin.",
        "Tipping": "6-3 experts pick Tigers",
    },
])

# ── Model ────────────────────────────────────────────────────────────────────
def calculate_prob(row):
    h_pf = row["H_PF"] * (1 - home_yardage_adj / 100)
    a_pf = row["A_PF"] * (1 - away_yardage_adj / 100)

    # Attack differential (who scores more vs who concedes more)
    attack_edge = (h_pf - row["A_PA"]) / 10   # Home attack vs Away defence
    defence_edge = (row["H_PA"] - a_pf) / 10  # Home defence vs Away attack (negative = home better)
    defence_edge = -defence_edge               # Flip so positive = home advantage

    # Season form: win % differential
    h_winpct = row["H_W"] / row["H_P"] if row["H_P"] > 0 else 0.5
    a_winpct = row["A_W"] / row["A_P"] if row["A_P"] > 0 else 0.5
    form_edge = (h_winpct - a_winpct) * 2

    # Referee context
    ref_adj = row["Ref_Boost_Home"] / 150

    score = (
        attack_w * attack_edge
        + defence_w * (defence_edge + form_edge * 0.5)
        + home_w * 1.0
        + context_w * ref_adj
    )

    prob = 1 / (1 + exp(-score * 0.55))
    return round(max(0.20, min(0.88, prob)), 4)


def get_bet_suggestion(prob, edge, home, away, h_notes, a_notes):
    if edge >= 8:
        return f"{home} -5.5", "STRONG", f"Big model edge. {h_notes}"
    elif edge >= 5:
        return f"{home} H2H", "value", f"Solid edge. {h_notes}"
    elif edge >= 3:
        return f"{home} H2H", "lean", f"Slight edge. Watch line movement."
    elif edge <= -8:
        return f"{away} +6.5", "STRONG", f"Away value. {a_notes}"
    elif edge <= -5:
        return f"{away} H2H", "value", f"Away edge. {a_notes}"
    elif edge <= -3:
        return f"{away} H2H", "lean", f"Slight away lean."
    else:
        return "Pass", "pass", "No meaningful edge."


# Run model
df = ROUND_DATA.copy()
df["Home_Prob"] = df.apply(calculate_prob, axis=1)
df["Away_Prob"] = round(1 - df["Home_Prob"], 4)
df["Fair_Home"] = round(1 / df["Home_Prob"], 2)
df["Fair_Away"] = round(1 / df["Away_Prob"], 2)
df["Edge_pp"] = round((df["Home_Prob"] - (1 / df["Mkt_Home"])) * 100, 1)
df["Away_Edge_pp"] = round((df["Away_Prob"] - (1 / df["Mkt_Away"])) * 100, 1)

bets = df.apply(
    lambda r: get_bet_suggestion(
        r["Home_Prob"], r["Edge_pp"], r["Home"], r["Away"], r["H_Notes"], r["A_Notes"]
    ), axis=1, result_type="expand",
)
df["Bet"], df["Strength"], df["Reason"] = bets[0], bets[1], bets[2]

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_dash, tab_detail, tab_ladder, tab_method = st.tabs(
    ["Dashboard", "Game Breakdowns", "2026 Ladder", "Methodology"]
)

# ═══════════════════════  DASHBOARD  ═════════════════════════════════════════
with tab_dash:
    # Metrics
    strong_bets = df[df["Strength"].isin(["STRONG", "value"])]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Round 15 Games", f"{len(df)} (Origin split)")
    c2.metric("Actionable Bets", len(strong_bets))
    c3.metric("Avg Edge", f"{strong_bets['Edge_pp'].mean():+.1f}pp" if len(strong_bets) else "0.0pp")
    c4.metric("Teams on Bye", "7")

    st.markdown("---")

    # Main predictions table
    st.subheader("Round 15 Predictions")
    show = df[[
        "Match", "Home_Prob", "Away_Prob", "Fair_Home", "Fair_Away",
        "Mkt_Home", "Mkt_Away", "Edge_pp", "Ref_Boost_Home", "Bet", "Strength",
    ]].copy()
    show.columns = [
        "Match", "Home%", "Away%", "Fair H", "Fair A",
        "Mkt H", "Mkt A", "Edge", "Ref", "Bet", "Rating",
    ]

    def style_edge(val):
        if isinstance(val, (int, float)):
            if val >= 5: return "background-color:#0d3320; color:#4ade80; font-weight:700"
            if val <= -5: return "background-color:#3b1010; color:#f87171; font-weight:700"
        return ""

    def style_rating(val):
        if val == "STRONG": return "background-color:#b45309; color:#fff; font-weight:700"
        if val == "value": return "background-color:#0d3320; color:#4ade80; font-weight:700"
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
            card_class = "bet-card-strong"
            tag = '<span class="tag tag-strong">STRONG</span>'
        elif r["Strength"] == "value":
            card_class = "bet-card"
            tag = '<span class="tag tag-value">VALUE</span>'
        else:
            card_class = "pass-card"
            tag = '<span class="tag tag-pass">PASS</span>'

        pick_team = r["Home"] if r["Home_Prob"] >= 0.5 else r["Away"]
        pick_prob = r["Home_Prob"] if r["Home_Prob"] >= 0.5 else r["Away_Prob"]
        pick_fair = r["Fair_Home"] if r["Home_Prob"] >= 0.5 else r["Fair_Away"]
        pick_mkt = r["Mkt_Home"] if r["Home_Prob"] >= 0.5 else r["Mkt_Away"]

        st.markdown(f"""<div class="{card_class}">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px">
                <div>{tag} <b style="font-size:1.1rem; color:#e5e7eb">{r['Bet']}</b>
                    <span style="color:#6b7280; margin-left:8px">{r['Match']}</span></div>
                <div style="font-size:1.1rem; font-weight:700; color:{'#4ade80' if r['Edge_pp'] > 0 else '#f87171'}">{r['Edge_pp']:+.1f}pp</div>
            </div>
            <div style="display:flex; gap:24px; color:#9ca3af; font-size:0.85rem">
                <span>Model: <b style="color:#60a5fa">{pick_prob:.0%}</b></span>
                <span>Fair: <b style="color:#60a5fa">${pick_fair:.2f}</b></span>
                <span>Market: <b style="color:#e5e7eb">${pick_mkt:.2f}</b></span>
                <span>Ref: <b style="color:#c084fc">{r['Ref_Boost_Home']:+.1f}pp</b></span>
                <span>Tipping: <b style="color:#9ca3af">{r['Tipping']}</b></span>
            </div>
            <div style="color:#6b7280; font-size:0.8rem; margin-top:6px">{r['Reason']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df, x="Match", y="Home_Prob", title="Home Win Probability",
            color_discrete_sequence=["#3b82f6"], template="plotly_dark",
        )
        fig.update_layout(
            yaxis_tickformat=".0%", yaxis_range=[0, 1],
            plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
            margin=dict(l=40, r=20, t=40, b=60),
        )
        fig.add_hline(y=0.5, line_dash="dot", line_color="#334155")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        colors = ["#22c55e" if e >= 3 else ("#ef4444" if e <= -3 else "#6b7280") for e in df["Edge_pp"]]
        fig2 = go.Figure(go.Bar(x=df["Match"], y=df["Edge_pp"], marker_color=colors))
        fig2.update_layout(
            title="Edge Map (pp)", template="plotly_dark",
            plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
            yaxis_title="Edge (pp)", margin=dict(l=40, r=20, t=40, b=60),
        )
        fig2.add_hline(y=0, line_color="#334155")
        st.plotly_chart(fig2, use_container_width=True)

    # Export
    export = df[[
        "Match", "Home", "Away", "Home_Prob", "Away_Prob", "Fair_Home", "Fair_Away",
        "Mkt_Home", "Mkt_Away", "Edge_pp", "Ref_Boost_Home", "Bet", "Strength", "Reason",
    ]]
    buf = BytesIO()
    export.to_csv(buf, index=False)
    st.download_button("Export Round 15 CSV", buf.getvalue(), "nrl_r15_moneyball.csv", "text/csv")

# ═══════════════════════  GAME BREAKDOWNS  ═══════════════════════════════════
with tab_detail:
    st.subheader("Per-Game Breakdown")
    for _, g in df.iterrows():
        strength_icon = {"STRONG": "🔥", "value": "✅", "lean": "👀"}.get(g["Strength"], "")
        with st.expander(f"{strength_icon} {g['Match']}  |  {g['Bet']}  |  Edge {g['Edge_pp']:+.1f}pp"):
            ca, cb, cc = st.columns(3)
            with ca:
                st.markdown("**Venue / Schedule**")
                st.write(f"{g['Venue']} - {g['Kickoff']}")
                st.write(f"Referee: {g['Referee']}")
                st.write(f"Ref Boost: {g['Ref_Boost_Home']:+.1f}pp")
                st.write(f"Expert tipping: {g['Tipping']}")
            with cb:
                st.markdown(f"**{g['Home']}** (Home) - {g['H_W']}W-{g['H_L']}L")
                st.write(f"Attack: {g['H_PF']:.1f} ppg (scored)")
                st.write(f"Defence: {g['H_PA']:.1f} ppg (conceded)")
                st.write(f"Season PD: {g['H_PD']:+d}")
                st.write(f"Origin outs: {g['H_Origin_Outs']}")
                st.caption(g["H_Notes"])
            with cc:
                st.markdown(f"**{g['Away']}** (Away) - {g['A_W']}W-{g['A_L']}L")
                st.write(f"Attack: {g['A_PF']:.1f} ppg (scored)")
                st.write(f"Defence: {g['A_PA']:.1f} ppg (conceded)")
                st.write(f"Season PD: {g['A_PD']:+d}")
                st.write(f"Origin outs: {g['A_Origin_Outs']}")
                st.caption(g["A_Notes"])

            st.markdown("---")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Home Win%", f"{g['Home_Prob']:.0%}")
            m2.metric("Fair Home", f"${g['Fair_Home']:.2f}")
            m3.metric("Fair Away", f"${g['Fair_Away']:.2f}")
            m4.metric("Market Home", f"${g['Mkt_Home']:.2f}")
            m5.metric("Edge", f"{g['Edge_pp']:+.1f}pp")

            # Show the math
            h_pf = g["H_PF"] * (1 - home_yardage_adj / 100)
            a_pf = g["A_PF"] * (1 - away_yardage_adj / 100)
            atk = (h_pf - g["A_PA"]) / 10
            dfe = -(g["H_PA"] - a_pf) / 10
            h_wp = g["H_W"] / g["H_P"]
            a_wp = g["A_W"] / g["A_P"]
            frm = (h_wp - a_wp) * 2
            ref = g["Ref_Boost_Home"] / 150
            st.code(
                f"Attack edge:  ({h_pf:.1f} - {g['A_PA']:.1f}) / 10 = {atk:+.3f}  x {attack_w:.2f} = {atk*attack_w:+.4f}\n"
                f"Defence edge: -({g['H_PA']:.1f} - {a_pf:.1f}) / 10 = {dfe:+.3f}  x {defence_w:.2f} = {dfe*defence_w:+.4f}\n"
                f"Form edge:    ({h_wp:.3f} - {a_wp:.3f}) x 2  = {frm:+.3f}  x {defence_w*0.5:.2f} = {frm*defence_w*0.5:+.4f}\n"
                f"Home base:    1.000  x {home_w:.2f} = {home_w:+.4f}\n"
                f"Ref context:  {g['Ref_Boost_Home']:+.1f}/150 = {ref:+.4f}  x {context_w:.2f} = {ref*context_w:+.5f}\n"
                f"{'='*55}\n"
                f"Raw score: {atk*attack_w + dfe*defence_w + frm*defence_w*0.5 + home_w + ref*context_w:+.4f}  ->  Home {g['Home_Prob']:.1%}"
            )

# ═══════════════════════  LADDER  ════════════════════════════════════════════
with tab_ladder:
    st.subheader("2026 NRL Ladder (through Round 14)")
    st.caption("Source: Zero Tackle. Verified 10 June 2026.")

    ladder = LADDER.sort_values("PD", ascending=False).reset_index(drop=True)
    ladder.index = ladder.index + 1
    st.dataframe(
        ladder[["Team", "P", "W", "L", "PF", "PA", "PD", "Avg_PF", "Avg_PA", "Avg_Diff", "Win%", "Form"]].style.format({
            "Win%": "{:.1f}%", "PD": "{:+d}", "Avg_PF": "{:.1f}", "Avg_PA": "{:.1f}", "Avg_Diff": "{:+.1f}",
        }),
        use_container_width=True,
    )

    st.subheader("Scoring Power Rankings")
    attack_rank = LADDER.sort_values("Avg_PF", ascending=True)
    fig_atk = px.bar(
        attack_rank, x="Avg_PF", y="Team", orientation="h",
        title="Points Scored Per Game", color="Avg_PF",
        color_continuous_scale="Greens", template="plotly_dark",
    )
    fig_atk.update_layout(
        plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
        coloraxis_showscale=False, yaxis_title="",
    )
    st.plotly_chart(fig_atk, use_container_width=True)

    defence_rank = LADDER.sort_values("Avg_PA", ascending=False)
    fig_def = px.bar(
        defence_rank, x="Avg_PA", y="Team", orientation="h",
        title="Points Conceded Per Game (lower = better)", color="Avg_PA",
        color_continuous_scale="Reds_r", template="plotly_dark",
    )
    fig_def.update_layout(
        plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
        coloraxis_showscale=False, yaxis_title="",
    )
    st.plotly_chart(fig_def, use_container_width=True)

# ═══════════════════════  METHODOLOGY  ═══════════════════════════════════════
with tab_method:
    st.subheader("How It Works")
    st.markdown("""
    **The model compares four things to estimate win probability:**

    1. **Attack Edge** -- How many points does the home team score per game vs how many
       the away team concedes? If Rabbitohs score 28.2 and Broncos concede 26.2,
       that's a positive attack edge for home.

    2. **Defence Edge** -- Same in reverse. If the home team concedes fewer points
       than the away team scores, that's a defensive edge for home.

    3. **Form Edge** -- Season win percentage differential. A 6-6 team vs a 5-8
       team gets a small form boost.

    4. **Context** -- Referee historical bias (from refereebias.com data) and
       Origin absences. A ref boost of -31.3pp (Warriors vs Sharks) means
       Sutton historically disadvantages the home team by 31pp.

    **The score gets converted to a probability using a logistic function**
    (S-curve), clamped between 20% and 88% to prevent absurd outputs.

    **Edge = Model probability minus the market's implied probability.**
    If the model says 65% but the bookie's $1.48 implies 67.6%, the edge
    is -2.6pp (no bet). If fair price is $1.35 but market is $1.48,
    there's value.

    **Suggested bets are rules-based:**
    - Edge >= 8pp: spread bet (e.g. -5.5), rated STRONG
    - Edge 5-8pp: H2H, rated VALUE
    - Edge 3-5pp: lean only, not a bet
    - Under 3pp: pass
    """)

    st.subheader("Data Sources")
    st.markdown("""
    | Data | Source | Verified |
    |------|--------|----------|
    | Ladder & standings | [Zero Tackle](https://zerotackle.com/nrl/nrl-ladder/) | 10 June 2026 |
    | Scoring averages | [Aus Sports Tipping](https://aussportstipping.com/sports/nrl/score_statistics/) | 10 June 2026 |
    | Round 15 teams | [Zero Tackle team lists](https://zerotackle.com/round-15-team-lists-2026-234642/) | 10 June 2026 |
    | Expert tipping | [Rugby League Zone](https://rugbyleaguezone.com/nrl-round-15-predictions-2026-430537/) | 10 June 2026 |
    | Referee bias | refereebias.com + existing dashboard | Historical |
    | Market odds | Bookmaker consensus | Pre-round |
    """)

    st.caption("Model is exploratory. Not financial advice. Gamble responsibly.")
