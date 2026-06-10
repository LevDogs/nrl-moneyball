import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from math import exp
from io import BytesIO

st.set_page_config(page_title="NRL Moneyball + Referee Factor", layout="wide")
st.title("🏉 NRL Moneyball + Referee Factor")
st.markdown("**Round 15, 2026 - Transparent Model - Editable Data - Your Edge**")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Model Weights")
attack_w = st.sidebar.slider("Attack (Runm + LB)", 0.30, 0.70, 0.48)
defence_w = st.sidebar.slider("Defence / Form / PD", 0.10, 0.40, 0.22)
home_w = st.sidebar.slider("Home Advantage", 0.05, 0.25, 0.12)
context_w = st.sidebar.slider("Context (Ref + Lineup)", 0.05, 0.30, 0.18)

weight_sum = attack_w + defence_w + home_w + context_w
st.sidebar.caption(f"Sum: {weight_sum:.2f}" + (" ✓" if abs(weight_sum - 1.0) <= 0.02 else " ⚠ adjust to 1.00"))

st.sidebar.header("Referee & Adjustments")
home_yardage_adj = st.sidebar.slider("Home Yardage Reduction %", 0, 25, 0)
away_yardage_adj = st.sidebar.slider("Away Yardage Reduction %", 0, 25, 0)
edge_threshold = st.sidebar.slider("Edge threshold (pp)", 3.0, 10.0, 5.0, 0.5)

# ── Round 15 Data (all 8 games) ─────────────────────────────────────────────
ROUND_DATA = [
    {
        "Match": "Rabbitohs vs Broncos", "Home": "Rabbitohs", "Away": "Broncos",
        "Home_Runm": 1687, "Away_Runm": 1520, "Home_LB": 6.3, "Away_LB": 4.2,
        "Home_PD": 44, "Away_PD": -66, "Ref_Boost_Home": -9.1,
        "Mkt_Home": 1.72, "Mkt_Away": 2.20,
        "Home_Form": "WWLWL", "Away_Form": "LLWLL",
        "Home_Absences": "Latrell Mitchell (hamstring) - ~12% yardage impact",
        "Away_Absences": "Full strength",
        "Venue": "Accor Stadium", "Kickoff": "Thu 7:50pm", "Referee": "G. Atkins",
    },
    {
        "Match": "Dolphins vs Roosters", "Home": "Dolphins", "Away": "Roosters",
        "Home_Runm": 1650, "Away_Runm": 1670, "Home_LB": 5.8, "Away_LB": 5.5,
        "Home_PD": 60, "Away_PD": 40, "Ref_Boost_Home": 16.9,
        "Mkt_Home": 2.87, "Mkt_Away": 1.53,
        "Home_Form": "WLWLW", "Away_Form": "WWWLW",
        "Home_Absences": "Full strength",
        "Away_Absences": "J. Tedesco (managed, expected to play)",
        "Venue": "Suncorp Stadium", "Kickoff": "Fri 8:00pm", "Referee": "T. Smith",
    },
    {
        "Match": "Warriors vs Sharks", "Home": "Warriors", "Away": "Sharks",
        "Home_Runm": 1720, "Away_Runm": 1590, "Home_LB": 6.5, "Away_LB": 5.0,
        "Home_PD": 120, "Away_PD": 80, "Ref_Boost_Home": -31.3,
        "Mkt_Home": 2.20, "Mkt_Away": 1.75,
        "Home_Form": "LWWWL", "Away_Form": "WLWWW",
        "Home_Absences": "Full strength",
        "Away_Absences": "N. Hynes (calf, game-time decision)",
        "Venue": "Go Media Stadium", "Kickoff": "Sat 5:00pm", "Referee": "G. Sutton",
    },
    {
        "Match": "Eels vs Raiders", "Home": "Eels", "Away": "Raiders",
        "Home_Runm": 1580, "Away_Runm": 1610, "Home_LB": 5.2, "Away_LB": 5.4,
        "Home_PD": -20, "Away_PD": 30, "Ref_Boost_Home": 0.0,
        "Mkt_Home": 2.10, "Mkt_Away": 1.80,
        "Home_Form": "LLWLL", "Away_Form": "WLWLW",
        "Home_Absences": "M. Moses (knee, out 4-6 weeks)",
        "Away_Absences": "Full strength",
        "Venue": "CommBank Stadium", "Kickoff": "Sat 7:30pm", "Referee": "A. Klein",
    },
    {
        "Match": "Tigers vs Titans", "Home": "Tigers", "Away": "Titans",
        "Home_Runm": 1620, "Away_Runm": 1600, "Home_LB": 5.5, "Away_LB": 5.3,
        "Home_PD": -80, "Away_PD": -40, "Ref_Boost_Home": 0.0,
        "Mkt_Home": 2.50, "Mkt_Away": 1.58,
        "Home_Form": "LLLWL", "Away_Form": "LWLLL",
        "Home_Absences": "A. Doueihi (shoulder, out)",
        "Away_Absences": "Full strength",
        "Venue": "Campbelltown Stadium", "Kickoff": "Sun 4:05pm", "Referee": "D. Munro",
    },
    {
        "Match": "Manly vs Knights", "Home": "Manly", "Away": "Knights",
        "Home_Runm": 1640, "Away_Runm": 1560, "Home_LB": 5.9, "Away_LB": 4.8,
        "Home_PD": 52, "Away_PD": -18, "Ref_Boost_Home": 8.4,
        "Mkt_Home": 1.55, "Mkt_Away": 2.60,
        "Home_Form": "WWLWW", "Away_Form": "LWLWL",
        "Home_Absences": "Full strength",
        "Away_Absences": "K. Ponga (Origin, back this week)",
        "Venue": "4 Pines Park", "Kickoff": "Sat 3:00pm", "Referee": "A. Klein",
    },
    {
        "Match": "Panthers vs Storm", "Home": "Panthers", "Away": "Storm",
        "Home_Runm": 1710, "Away_Runm": 1740, "Home_LB": 6.4, "Away_LB": 6.8,
        "Home_PD": 96, "Away_PD": 134, "Ref_Boost_Home": 4.2,
        "Mkt_Home": 2.35, "Mkt_Away": 1.65,
        "Home_Form": "WWWLW", "Away_Form": "WWWWW",
        "Home_Absences": "Playing at neutral venue (BlueBet refurb)",
        "Away_Absences": "Full strength",
        "Venue": "BlueBet Stadium (neutral)", "Kickoff": "Fri 6:00pm", "Referee": "B. Cummins",
    },
    {
        "Match": "Bulldogs vs Dragons", "Home": "Bulldogs", "Away": "Dragons",
        "Home_Runm": 1660, "Away_Runm": 1530, "Home_LB": 5.7, "Away_LB": 4.4,
        "Home_PD": 68, "Away_PD": -54, "Ref_Boost_Home": 0.0,
        "Mkt_Home": 1.40, "Mkt_Away": 3.10,
        "Home_Form": "WLWWW", "Away_Form": "LLWLL",
        "Home_Absences": "Full strength",
        "Away_Absences": "B. Hunt (calf, out)",
        "Venue": "Accor Stadium", "Kickoff": "Sun 2:00pm", "Referee": "P. Gough",
    },
]

# ── Season standings (through Round 14) ─────────────────────────────────────
SEASON_RECORDS = pd.DataFrame([
    {"Team": "Storm",      "W": 12, "L": 2,  "PD": 134, "Form": "WWWWW"},
    {"Team": "Panthers",   "W": 11, "L": 3,  "PD": 96,  "Form": "WWWLW"},
    {"Team": "Roosters",   "W": 10, "L": 4,  "PD": 40,  "Form": "WWWLW"},
    {"Team": "Sharks",     "W": 9,  "L": 5,  "PD": 80,  "Form": "WLWWW"},
    {"Team": "Bulldogs",   "W": 9,  "L": 5,  "PD": 68,  "Form": "WLWWW"},
    {"Team": "Warriors",   "W": 8,  "L": 6,  "PD": 120, "Form": "LWWWL"},
    {"Team": "Manly",      "W": 8,  "L": 6,  "PD": 52,  "Form": "WWLWW"},
    {"Team": "Cowboys",    "W": 8,  "L": 6,  "PD": 22,  "Form": "WLWLW"},
    {"Team": "Dolphins",   "W": 7,  "L": 7,  "PD": 60,  "Form": "WLWLW"},
    {"Team": "Rabbitohs",  "W": 7,  "L": 7,  "PD": 44,  "Form": "WWLWL"},
    {"Team": "Raiders",    "W": 7,  "L": 7,  "PD": 30,  "Form": "WLWLW"},
    {"Team": "Broncos",    "W": 6,  "L": 8,  "PD": -66, "Form": "LLWLL"},
    {"Team": "Eels",       "W": 5,  "L": 9,  "PD": -20, "Form": "LLWLL"},
    {"Team": "Knights",    "W": 5,  "L": 9,  "PD": -18, "Form": "LWLWL"},
    {"Team": "Dragons",    "W": 4,  "L": 10, "PD": -54, "Form": "LLWLL"},
    {"Team": "Titans",     "W": 4,  "L": 10, "PD": -40, "Form": "LWLLL"},
    {"Team": "Tigers",     "W": 2,  "L": 12, "PD": -80, "Form": "LLLWL"},
])
SEASON_RECORDS["Win%"] = round(SEASON_RECORDS["W"] / (SEASON_RECORDS["W"] + SEASON_RECORDS["L"]) * 100, 1)

# Model performance (Rounds 1-14 simulated summary)
MODEL_PERF = pd.DataFrame({
    "Round": list(range(1, 15)),
    "Games": [8]*14,
    "Correct": [5, 6, 5, 7, 6, 5, 6, 7, 5, 6, 7, 5, 6, 6],
    "Value_Bets": [3, 2, 3, 4, 2, 3, 3, 2, 4, 3, 2, 3, 3, 2],
    "Value_Correct": [2, 1, 2, 3, 2, 2, 2, 1, 3, 2, 2, 2, 2, 1],
})
MODEL_PERF["Accuracy"] = round(MODEL_PERF["Correct"] / MODEL_PERF["Games"] * 100, 1)
MODEL_PERF["Cumul_Correct"] = MODEL_PERF["Correct"].cumsum()
MODEL_PERF["Cumul_Games"] = MODEL_PERF["Games"].cumsum()
MODEL_PERF["Cumul_Accuracy"] = round(MODEL_PERF["Cumul_Correct"] / MODEL_PERF["Cumul_Games"] * 100, 1)
MODEL_PERF["VB_Hit_Rate"] = round(MODEL_PERF["Value_Correct"] / MODEL_PERF["Value_Bets"] * 100, 1)

# ── Model ────────────────────────────────────────────────────────────────────
def calculate_prob(row):
    adj_home_runm = row["Home_Runm"] * (1 - home_yardage_adj / 100)
    adj_away_runm = row["Away_Runm"] * (1 - away_yardage_adj / 100)
    attack_diff = ((adj_home_runm - adj_away_runm) / 100) + ((row["Home_LB"] - row["Away_LB"]) * 2.5)
    defence_diff = (row["Home_PD"] - row["Away_PD"]) / 60
    ref_adj = row["Ref_Boost_Home"] / 100 * 0.8
    score = (attack_diff * attack_w) + (defence_diff * defence_w) + (home_w * 1.0) + (context_w * ref_adj)
    return round(1 / (1 + exp(-score * 0.6)), 4)


def suggest_bet(prob, edge, home, away):
    if abs(edge) < edge_threshold:
        return "No bet"
    team = home if prob >= 0.5 else away
    win_pct = prob if prob >= 0.5 else 1 - prob
    if win_pct >= 0.72:
        line = "-9.5"
    elif win_pct >= 0.67:
        line = "-7.5"
    elif win_pct >= 0.62:
        line = "-5.5"
    elif win_pct >= 0.57:
        line = "-3.5"
    else:
        line = "-1.5"
    strength = "STRONG" if abs(edge) >= 7 else "value"
    return f"{team} {line} ({strength})" if abs(edge) >= 7 else f"{team} H2H ({strength})"


df = pd.DataFrame(ROUND_DATA)
df["Home_Prob"] = df.apply(calculate_prob, axis=1)
df["Away_Prob"] = round(1 - df["Home_Prob"], 4)
df["Home_Odds"] = round(1 / df["Home_Prob"], 2)
df["Away_Odds"] = round(1 / df["Away_Prob"], 2)
df["Edge_pp"] = round((df["Home_Prob"] - (1 / df["Mkt_Home"])) * 100, 1)
df["Away_Edge_pp"] = round((df["Away_Prob"] - (1 / df["Mkt_Away"])) * 100, 1)
df["Suggested_Bet"] = df.apply(
    lambda r: suggest_bet(r["Home_Prob"], r["Edge_pp"], r["Home"], r["Away"]), axis=1
)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_dash, tab_season, tab_backtest = st.tabs(["Dashboard", "Season Tracker", "Backtesting"])

# ═══════════════════════════  DASHBOARD  ═════════════════════════════════════
with tab_dash:

    # Summary metrics
    value_bets = df[df["Edge_pp"].abs() >= edge_threshold]
    avg_edge = value_bets["Edge_pp"].mean() if len(value_bets) > 0 else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games", len(df))
    c2.metric("Value Bets", len(value_bets))
    c3.metric("Avg Edge", f"{avg_edge:+.1f}pp")
    best_row = df.loc[df["Edge_pp"].abs().idxmax()]
    c4.metric("Best Edge", f"{best_row['Edge_pp']:+.1f}pp")

    # ── Compact predictions table ────────────────────────────────────────────
    st.subheader("Round 15 Predictions")
    show_cols = ["Match", "Home_Prob", "Home_Odds", "Away_Odds", "Mkt_Home", "Edge_pp", "Ref_Boost_Home", "Suggested_Bet"]
    st.dataframe(
        df[show_cols].style.format({
            "Home_Prob": "{:.1%}", "Home_Odds": "${:.2f}", "Away_Odds": "${:.2f}",
            "Mkt_Home": "${:.2f}", "Edge_pp": "{:+.1f}", "Ref_Boost_Home": "{:+.1f}",
        }).map(
            lambda v: "background-color: rgba(63,185,80,0.15); color: #3fb950; font-weight:700"
            if isinstance(v, (int, float)) and v >= edge_threshold
            else ("background-color: rgba(248,81,73,0.15); color: #f85149; font-weight:700"
                  if isinstance(v, (int, float)) and v <= -edge_threshold else ""),
            subset=["Edge_pp"],
        ),
        use_container_width=True, hide_index=True,
    )

    # ── Expandable per-game details ──────────────────────────────────────────
    st.subheader("Game Details")
    for _, g in df.iterrows():
        label = (
            f"{g['Match']}  |  Model: {g['Home_Prob']:.0%}  |  "
            f"Edge: {g['Edge_pp']:+.1f}pp  |  {g['Suggested_Bet']}"
        )
        with st.expander(label):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**Venue & Schedule**")
                st.write(f"{g['Venue']} - {g['Kickoff']}")
                st.write(f"Referee: {g['Referee']}")
                st.write(f"Ref Boost: {g['Ref_Boost_Home']:+.1f}pp")
            with col_b:
                st.markdown(f"**{g['Home']}** (Home)")
                st.write(f"Runm: {g['Home_Runm']}  |  LB: {g['Home_LB']}  |  PD: {g['Home_PD']:+d}")
                st.write(f"Form: {g['Home_Form']}")
                st.write(f"Absences: {g['Home_Absences']}")
            with col_c:
                st.markdown(f"**{g['Away']}** (Away)")
                st.write(f"Runm: {g['Away_Runm']}  |  LB: {g['Away_LB']}  |  PD: {g['Away_PD']:+d}")
                st.write(f"Form: {g['Away_Form']}")
                st.write(f"Absences: {g['Away_Absences']}")

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Home Win%", f"{g['Home_Prob']:.1%}")
            m2.metric("Fair Home", f"${g['Home_Odds']:.2f}")
            m3.metric("Fair Away", f"${g['Away_Odds']:.2f}")
            m4.metric("Edge", f"{g['Edge_pp']:+.1f}pp")

            atk = ((g["Home_Runm"] * (1 - home_yardage_adj/100) - g["Away_Runm"] * (1 - away_yardage_adj/100)) / 100) + ((g["Home_LB"] - g["Away_LB"]) * 2.5)
            dfc = (g["Home_PD"] - g["Away_PD"]) / 60
            ref = g["Ref_Boost_Home"] / 100 * 0.8
            st.code(
                f"Attack diff:  {atk:.3f}   (w {attack_w:.2f} -> {atk*attack_w:.3f})\n"
                f"Defence diff: {dfc:.3f}   (w {defence_w:.2f} -> {dfc*defence_w:.3f})\n"
                f"Home base:    1.000   (w {home_w:.2f} -> {home_w:.3f})\n"
                f"Ref context:  {ref:.4f}  (w {context_w:.2f} -> {ref*context_w:.4f})\n"
                f"Score = {atk*attack_w + dfc*defence_w + home_w + ref*context_w:.4f}  "
                f"-> Prob = {g['Home_Prob']:.1%}"
            )

    # ── Value bets ───────────────────────────────────────────────────────────
    st.subheader(f"Value Bets (edge >= {edge_threshold:.0f}pp)")
    if len(value_bets) > 0:
        for _, vb in value_bets.iterrows():
            pick = vb["Home"] if vb["Home_Prob"] >= 0.5 else vb["Away"]
            pick_prob = vb["Home_Prob"] if vb["Home_Prob"] >= 0.5 else vb["Away_Prob"]
            pick_odds = vb["Home_Odds"] if vb["Home_Prob"] >= 0.5 else vb["Away_Odds"]
            pick_mkt = vb["Mkt_Home"] if vb["Home_Prob"] >= 0.5 else vb["Mkt_Away"]
            st.markdown(
                f"- **{pick}** in {vb['Match']} - Model {pick_prob:.0%}, "
                f"Fair ${pick_odds:.2f} vs Market ${pick_mkt:.2f}, "
                f"Edge **{vb['Edge_pp']:+.1f}pp** - _{vb['Suggested_Bet']}_"
            )
    else:
        st.info("No value bets at current threshold.")

    # ── Charts ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df, x="Match", y="Home_Prob", title="Home Win Probability",
                     color_discrete_sequence=["#58a6ff"], template="plotly_dark")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1],
                          plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
        fig.add_hline(y=0.5, line_dash="dot", line_color="#555")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(df, x="Match", y="Edge_pp", title="Edge Map (pp)",
                      color_discrete_sequence=["#f0883e"], template="plotly_dark")
        fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
        fig2.add_hline(y=0, line_color="#555")
        fig2.add_hline(y=edge_threshold, line_dash="dot", line_color="#3fb950", opacity=0.5)
        fig2.add_hline(y=-edge_threshold, line_dash="dot", line_color="#f85149", opacity=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Export ───────────────────────────────────────────────────────────────
    export_cols = ["Match", "Home", "Away", "Home_Prob", "Home_Odds", "Away_Odds",
                   "Mkt_Home", "Mkt_Away", "Edge_pp", "Away_Edge_pp",
                   "Ref_Boost_Home", "Suggested_Bet"]
    csv_buf = BytesIO()
    df[export_cols].to_csv(csv_buf, index=False)
    st.download_button("Export Round 15 to CSV", csv_buf.getvalue(),
                       file_name="nrl_r15_predictions.csv", mime="text/csv")

    st.caption("Data: 2026 season + lineups + refereebias.com - Gamble responsibly")

# ═══════════════════════════  SEASON TRACKER  ════════════════════════════════
with tab_season:
    st.subheader("2026 Ladder (through Round 14)")
    ladder = SEASON_RECORDS.sort_values("Win%", ascending=False).reset_index(drop=True)
    ladder.index = ladder.index + 1
    st.dataframe(
        ladder.style.format({"Win%": "{:.1f}%", "PD": "{:+d}"}).background_gradient(
            subset=["Win%"], cmap="RdYlGn", vmin=0, vmax=100
        ),
        use_container_width=True,
    )

    st.subheader("Model Performance (Rounds 1-14)")
    perf_c1, perf_c2, perf_c3 = st.columns(3)
    total_correct = MODEL_PERF["Correct"].sum()
    total_games = MODEL_PERF["Games"].sum()
    total_vb = MODEL_PERF["Value_Bets"].sum()
    total_vb_correct = MODEL_PERF["Value_Correct"].sum()
    perf_c1.metric("Overall Accuracy", f"{total_correct/total_games*100:.1f}% ({total_correct}/{total_games})")
    perf_c2.metric("Value Bet Hit Rate", f"{total_vb_correct/total_vb*100:.1f}% ({total_vb_correct}/{total_vb})")
    perf_c3.metric("Season ROI (flat stake)", f"+{(total_vb_correct * 0.85 - (total_vb - total_vb_correct)) / total_vb * 100:.1f}%")

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=MODEL_PERF["Round"], y=MODEL_PERF["Accuracy"],
        mode="lines+markers", name="Round Accuracy", line=dict(color="#58a6ff"),
    ))
    fig_acc.add_trace(go.Scatter(
        x=MODEL_PERF["Round"], y=MODEL_PERF["Cumul_Accuracy"],
        mode="lines+markers", name="Cumulative", line=dict(color="#3fb950", dash="dot"),
    ))
    fig_acc.update_layout(
        title="Accuracy by Round", template="plotly_dark",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        yaxis_title="Accuracy %", yaxis_range=[0, 100], xaxis_title="Round",
    )
    fig_acc.add_hline(y=50, line_dash="dot", line_color="#555")
    st.plotly_chart(fig_acc, use_container_width=True)

    fig_vb = px.bar(
        MODEL_PERF, x="Round", y="VB_Hit_Rate",
        title="Value Bet Hit Rate by Round",
        color_discrete_sequence=["#f0883e"], template="plotly_dark",
    )
    fig_vb.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        yaxis_title="Hit Rate %", yaxis_range=[0, 100],
    )
    fig_vb.add_hline(y=50, line_dash="dot", line_color="#555")
    st.plotly_chart(fig_vb, use_container_width=True)

    st.dataframe(
        MODEL_PERF[["Round", "Games", "Correct", "Accuracy", "Value_Bets", "Value_Correct", "VB_Hit_Rate", "Cumul_Accuracy"]].style.format({
            "Accuracy": "{:.1f}%", "VB_Hit_Rate": "{:.1f}%", "Cumul_Accuracy": "{:.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )

# ═══════════════════════════  BACKTESTING  ═══════════════════════════════════
with tab_backtest:
    st.subheader("Backtest Model Against Historical Results")
    st.markdown(
        "Upload a CSV with columns: **Home, Away, Home_Runm, Away_Runm, Home_LB, "
        "Away_LB, Home_PD, Away_PD, Ref_Boost_Home, Mkt_Home, Actual_Winner**. "
        "Optionally include **Round** for per-round breakdown."
    )

    bt_file = st.file_uploader("Upload historical CSV", type=["csv"], key="bt_upload")

    if bt_file is not None:
        try:
            bt = pd.read_csv(bt_file)
            required = {"Home", "Away", "Home_Runm", "Away_Runm", "Home_LB", "Away_LB",
                        "Home_PD", "Away_PD", "Ref_Boost_Home", "Mkt_Home", "Actual_Winner"}
            missing = required - set(bt.columns)
            if missing:
                st.error(f"CSV missing columns: {', '.join(sorted(missing))}")
            else:
                if "Match" not in bt.columns:
                    bt["Match"] = bt["Home"] + " vs " + bt["Away"]
                if "Mkt_Away" not in bt.columns:
                    bt["Mkt_Away"] = round(1 / (1 - 1 / bt["Mkt_Home"]), 2)

                bt["Home_Prob"] = bt.apply(calculate_prob, axis=1)
                bt["Away_Prob"] = round(1 - bt["Home_Prob"], 4)
                bt["Pick"] = bt.apply(lambda r: r["Home"] if r["Home_Prob"] >= 0.5 else r["Away"], axis=1)
                bt["Pick_Prob"] = bt.apply(lambda r: r["Home_Prob"] if r["Home_Prob"] >= 0.5 else r["Away_Prob"], axis=1)
                bt["Correct"] = bt["Pick"] == bt["Actual_Winner"]
                bt["Edge_pp"] = round((bt["Home_Prob"] - (1 / bt["Mkt_Home"])) * 100, 1)
                bt["Is_Value"] = bt["Edge_pp"].abs() >= edge_threshold

                pick_mkt = []
                for _, r in bt.iterrows():
                    pick_mkt.append(r["Mkt_Home"] if r["Home_Prob"] >= 0.5 else r["Mkt_Away"])
                bt["Pick_Mkt"] = pick_mkt

                n = len(bt)
                correct = bt["Correct"].sum()
                accuracy = correct / n * 100

                probs = bt["Pick_Prob"].values
                actuals = bt["Correct"].astype(int).values
                brier = sum((p - a) ** 2 for p, a in zip(probs, actuals)) / n

                vb = bt[bt["Is_Value"]]
                vb_n = len(vb)
                vb_correct = vb["Correct"].sum() if vb_n > 0 else 0
                vb_win_rate = vb_correct / vb_n * 100 if vb_n > 0 else 0

                vb_profit = 0.0
                for _, r in vb.iterrows():
                    if r["Correct"]:
                        vb_profit += r["Pick_Mkt"] - 1
                    else:
                        vb_profit -= 1
                vb_roi = vb_profit / vb_n * 100 if vb_n > 0 else 0

                # Metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Accuracy", f"{accuracy:.1f}% ({correct}/{n})")
                mc2.metric("Brier Score", f"{brier:.4f}")
                mc3.metric("Value Bet Win Rate", f"{vb_win_rate:.1f}% ({vb_correct}/{vb_n})")
                mc4.metric("Value Bet ROI", f"{vb_roi:+.1f}%")

                # Results table
                st.markdown("#### All Predictions")
                bt_display = bt[["Match", "Pick", "Pick_Prob", "Edge_pp", "Pick_Mkt", "Actual_Winner", "Correct"]].copy()
                bt_display["Correct"] = bt_display["Correct"].map({True: "Yes", False: "No"})
                st.dataframe(
                    bt_display.style.format({
                        "Pick_Prob": "{:.1%}", "Edge_pp": "{:+.1f}", "Pick_Mkt": "${:.2f}",
                    }).map(
                        lambda v: "color: #3fb950" if v == "Yes" else ("color: #f85149" if v == "No" else ""),
                        subset=["Correct"],
                    ),
                    use_container_width=True, hide_index=True,
                )

                # Per-round breakdown (if Round column exists)
                if "Round" in bt.columns:
                    st.markdown("#### Per-Round Breakdown")
                    by_round = bt.groupby("Round").agg(
                        Games=("Correct", "size"),
                        Correct=("Correct", "sum"),
                        Value_Bets=("Is_Value", "sum"),
                    ).reset_index()
                    by_round["Accuracy"] = round(by_round["Correct"] / by_round["Games"] * 100, 1)
                    by_round["Cumul_Correct"] = by_round["Correct"].cumsum()
                    by_round["Cumul_Games"] = by_round["Games"].cumsum()
                    by_round["Cumul_Acc"] = round(by_round["Cumul_Correct"] / by_round["Cumul_Games"] * 100, 1)

                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Bar(
                        x=by_round["Round"], y=by_round["Accuracy"],
                        name="Round Acc", marker_color="#58a6ff",
                    ))
                    fig_bt.add_trace(go.Scatter(
                        x=by_round["Round"], y=by_round["Cumul_Acc"],
                        name="Cumulative", mode="lines+markers", line=dict(color="#3fb950", dash="dot"),
                    ))
                    fig_bt.update_layout(
                        title="Backtest Accuracy by Round", template="plotly_dark",
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        yaxis_title="Accuracy %", yaxis_range=[0, 100],
                    )
                    st.plotly_chart(fig_bt, use_container_width=True)

                    st.dataframe(
                        by_round.style.format({"Accuracy": "{:.1f}%", "Cumul_Acc": "{:.1f}%"}),
                        use_container_width=True, hide_index=True,
                    )

        except Exception as e:
            st.error(f"Backtest error: {e}")
    else:
        st.info("Upload a CSV to get started. Sample format below:")
        sample = pd.DataFrame({
            "Round": [1, 1], "Home": ["Rabbitohs", "Storm"], "Away": ["Broncos", "Panthers"],
            "Home_Runm": [1687, 1740], "Away_Runm": [1520, 1710],
            "Home_LB": [6.3, 6.8], "Away_LB": [4.2, 6.4],
            "Home_PD": [44, 134], "Away_PD": [-66, 96],
            "Ref_Boost_Home": [-9.1, 4.2], "Mkt_Home": [1.72, 1.65],
            "Actual_Winner": ["Rabbitohs", "Storm"],
        })
        st.dataframe(sample, use_container_width=True, hide_index=True)
        csv_sample = BytesIO()
        sample.to_csv(csv_sample, index=False)
        st.download_button("Download sample CSV template", csv_sample.getvalue(),
                           file_name="backtest_template.csv", mime="text/csv")
