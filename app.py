import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from math import exp
from io import BytesIO

st.set_page_config(page_title="NRL Moneyball + Referee Factor", page_icon="🏈", layout="wide")
st.title("🏉 NRL Moneyball + Referee Factor")
st.markdown("**Round 15, 2026 - Transparent - Lineup & Ref Adjusted**")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Model Weights")
attack_w = st.sidebar.slider("Attack (Runm + LB)", 0.30, 0.70, 0.48)
defence_w = st.sidebar.slider("Defence / Form / PD", 0.10, 0.40, 0.22)
home_w = st.sidebar.slider("Home Advantage", 0.05, 0.25, 0.12)
context_w = st.sidebar.slider("Context (Ref + Lineup)", 0.05, 0.30, 0.18)

wt = attack_w + defence_w + home_w + context_w
st.sidebar.caption(f"Sum: {wt:.2f}" + (" ✓" if abs(wt - 1.0) <= 0.02 else " -- adjust to 1.00"))

st.sidebar.header("Adjustments")
ref_boost = st.sidebar.number_input("Ref Boost for Home (pp)", value=0.0, step=0.1)
home_yardage_adj = st.sidebar.slider("Home Yardage Reduction %", 0, 25, 0)
away_yardage_adj = st.sidebar.slider("Away Yardage Reduction %", 0, 25, 0)
edge_threshold = st.sidebar.slider("Edge threshold (pp)", 3.0, 10.0, 5.0, 0.5)

# ── Round 15 data ────────────────────────────────────────────────────────────
DEFAULT_DATA = pd.DataFrame({
    "Match": [
        "Rabbitohs vs Broncos", "Dolphins vs Roosters",
        "Warriors vs Sharks", "Eels vs Raiders", "Tigers vs Titans",
    ],
    "Home": ["Rabbitohs", "Dolphins", "Warriors", "Eels", "Tigers"],
    "Away": ["Broncos", "Roosters", "Sharks", "Raiders", "Titans"],
    "Lineup_Notes": [
        "Rabbitohs: Strong forwards (Tatola, Fifita, Hubner). Broncos: Heavily depleted - no Haas, Walsh, Staggs",
        "Both sides depleted by Origin",
        "Warriors home core strong. Sharks missing key forwards",
        "Eels without Moses. Raiders reshuffled",
        "Tigers at home. Titans competitive",
    ],
    "Home_Runm": [1687, 1650, 1720, 1580, 1620],
    "Away_Runm": [1520, 1670, 1590, 1610, 1600],
    "Home_LB": [6.3, 5.8, 6.5, 5.2, 5.5],
    "Away_LB": [4.2, 5.5, 5.0, 5.4, 5.3],
    "Home_PD": [44, 60, 120, -20, -80],
    "Away_PD": [-66, 40, 80, 30, -40],
    "Ref_Boost_Home": [-9.1, 0.0, -31.3, 0.0, 0.0],
    "Mkt_Home": [1.48, 1.43, 1.35, 2.10, 2.50],
})

if "match_data" not in st.session_state:
    st.session_state.match_data = DEFAULT_DATA.copy()

# ── Season records (through Round 14) ────────────────────────────────────────
SEASON = pd.DataFrame({
    "Team": [
        "Storm", "Panthers", "Roosters", "Sharks", "Bulldogs", "Warriors",
        "Manly", "Cowboys", "Dolphins", "Rabbitohs", "Raiders", "Broncos",
        "Eels", "Knights", "Dragons", "Titans", "Tigers",
    ],
    "W": [12, 11, 10, 9, 9, 8, 8, 8, 7, 7, 7, 6, 5, 5, 4, 4, 2],
    "L": [2, 3, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 9, 9, 10, 10, 12],
    "PD": [134, 96, 40, 80, 68, 120, 52, 22, 60, 44, 30, -66, -20, -18, -54, -40, -80],
    "Form": [
        "WWWWW", "WWWLW", "WWWLW", "WLWWW", "WLWWW", "LWWWL",
        "WWLWW", "WLWLW", "WLWLW", "WWLWL", "WLWLW", "LLWLL",
        "LLWLL", "LWLWL", "LLWLL", "LWLLL", "LLLWL",
    ],
})
SEASON["Win%"] = round(SEASON["W"] / (SEASON["W"] + SEASON["L"]) * 100, 1)


# ── Model ────────────────────────────────────────────────────────────────────
def calculate_prob(row):
    adj_home_runm = row["Home_Runm"] * (1 - home_yardage_adj / 100)
    adj_away_runm = row["Away_Runm"] * (1 - away_yardage_adj / 100)
    attack_diff = ((adj_home_runm - adj_away_runm) / 100) + (
        (row["Home_LB"] - row["Away_LB"]) * 2.5
    )
    defence_diff = (row["Home_PD"] - row["Away_PD"]) / 60
    ref_adj = row["Ref_Boost_Home"] / 100 * 0.8
    score = (
        (attack_diff * attack_w)
        + (defence_diff * defence_w)
        + (home_w * 1.0)
        + (context_w * ref_adj)
    )
    return round(1 / (1 + exp(-score * 0.6)), 4)


def suggest_bet(prob_home, edge, home, away, threshold):
    if abs(edge) < threshold:
        return "No bet"
    if prob_home >= 0.5:
        team, pct = home, prob_home
    else:
        team, pct = away, 1 - prob_home
    if pct >= 0.72:
        line = "-9.5"
    elif pct >= 0.67:
        line = "-7.5"
    elif pct >= 0.62:
        line = "-5.5"
    elif pct >= 0.57:
        line = "-3.5"
    else:
        line = "H2H"
    tag = "STRONG" if abs(edge) >= 7 else "value"
    if line == "H2H":
        return f"{team} H2H ({tag})"
    return f"{team} {line} ({tag})"


def run_predictions(df):
    out = df.copy()
    out["Home_Win%"] = out.apply(calculate_prob, axis=1)
    out["Away_Win%"] = round(1 - out["Home_Win%"], 4)
    out["Fair_Home"] = round(1 / out["Home_Win%"], 2)
    out["Fair_Away"] = round(1 / out["Away_Win%"], 2)
    out["Edge_pp"] = round((out["Home_Win%"] - (1 / out["Mkt_Home"])) * 100, 1)
    out["Suggested_Bet"] = out.apply(
        lambda r: suggest_bet(
            r["Home_Win%"], r["Edge_pp"], r["Home"], r["Away"], edge_threshold
        ),
        axis=1,
    )
    return out


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_dash, tab_edit, tab_season, tab_backtest = st.tabs(
    ["Dashboard", "Edit Data", "Season Tracking", "Backtesting"]
)

# ═══════════════════════════  DASHBOARD  ═════════════════════════════════════
with tab_dash:
    results = run_predictions(st.session_state.match_data)

    # Summary row
    vb = results[results["Edge_pp"].abs() >= edge_threshold]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games", len(results))
    c2.metric("Value Bets", len(vb))
    c3.metric("Avg Edge", f"{vb['Edge_pp'].mean():+.1f}pp" if len(vb) else "0.0pp")
    best = results.loc[results["Edge_pp"].abs().idxmax()]
    c4.metric("Best Edge", f"{best['Edge_pp']:+.1f}pp")

    # Main table
    st.subheader("Round 15 - Full Analysis")
    show = results[
        [
            "Match", "Lineup_Notes", "Home_Win%", "Away_Win%",
            "Fair_Home", "Fair_Away", "Mkt_Home", "Edge_pp",
            "Ref_Boost_Home", "Suggested_Bet",
        ]
    ].copy()
    show.columns = [
        "Match", "Lineup / Notes", "Home Win%", "Away Win%",
        "Fair Home", "Fair Away", "Mkt Home", "Edge (pp)",
        "Ref Boost", "Suggested Bet",
    ]
    st.dataframe(
        show.style.format({
            "Home Win%": "{:.1%}", "Away Win%": "{:.1%}",
            "Fair Home": "${:.2f}", "Fair Away": "${:.2f}",
            "Mkt Home": "${:.2f}", "Edge (pp)": "{:+.1f}",
            "Ref Boost": "{:+.1f}",
        }).map(
            lambda v: (
                "background-color: rgba(63,185,80,0.15); color: #3fb950; font-weight:700"
                if isinstance(v, (int, float)) and v >= edge_threshold
                else (
                    "background-color: rgba(248,81,73,0.15); color: #f85149; font-weight:700"
                    if isinstance(v, (int, float)) and v <= -edge_threshold
                    else ""
                )
            ),
            subset=["Edge (pp)"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Value bets
    st.subheader(f"Value Bets (edge >= {edge_threshold:.0f}pp)")
    if len(vb):
        for _, r in vb.iterrows():
            pick = r["Home"] if r["Home_Win%"] >= 0.5 else r["Away"]
            pct = r["Home_Win%"] if r["Home_Win%"] >= 0.5 else r["Away_Win%"]
            fair = r["Fair_Home"] if r["Home_Win%"] >= 0.5 else r["Fair_Away"]
            st.markdown(
                f"- **{pick}** in {r['Match']} -- Model {pct:.0%}, "
                f"Fair ${fair:.2f} vs Market ${r['Mkt_Home']:.2f}, "
                f"Edge **{r['Edge_pp']:+.1f}pp** -- _{r['Suggested_Bet']}_"
            )
    else:
        st.info("No value bets at current threshold.")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            results, x="Match", y="Home_Win%",
            title="Home Win Probability",
            color_discrete_sequence=["#58a6ff"], template="plotly_dark",
        )
        fig.update_layout(
            yaxis_tickformat=".0%", yaxis_range=[0, 1],
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        )
        fig.add_hline(y=0.5, line_dash="dot", line_color="#555")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(
            results, x="Match", y="Edge_pp",
            title="Edge Map (pp)",
            color_discrete_sequence=["#f0883e"], template="plotly_dark",
        )
        fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
        fig2.add_hline(y=0, line_color="#555")
        fig2.add_hline(y=edge_threshold, line_dash="dot", line_color="#3fb950", opacity=0.5)
        fig2.add_hline(y=-edge_threshold, line_dash="dot", line_color="#f85149", opacity=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    # Export
    csv_buf = BytesIO()
    results[
        ["Match", "Home", "Away", "Home_Win%", "Away_Win%", "Fair_Home",
         "Fair_Away", "Mkt_Home", "Edge_pp", "Ref_Boost_Home", "Suggested_Bet"]
    ].to_csv(csv_buf, index=False)
    st.download_button(
        "Export Round 15 to CSV", csv_buf.getvalue(),
        file_name="nrl_r15_predictions.csv", mime="text/csv",
    )

    st.caption(
        "Data verified from NRL official lists, ZeroTackle stats, "
        "and refereebias.com - Gamble responsibly"
    )

# ═══════════════════════════  EDIT DATA  ═════════════════════════════════════
with tab_edit:
    st.subheader("Edit Match Data")
    st.caption("Changes here update the Dashboard instantly.")

    edited = st.data_editor(
        st.session_state.match_data,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Match": st.column_config.TextColumn("Match", width="medium"),
            "Lineup_Notes": st.column_config.TextColumn("Lineup / Notes", width="large"),
            "Home_Runm": st.column_config.NumberColumn("Home Runm", min_value=0, step=10),
            "Away_Runm": st.column_config.NumberColumn("Away Runm", min_value=0, step=10),
            "Home_LB": st.column_config.NumberColumn("Home LB", step=0.1, format="%.1f"),
            "Away_LB": st.column_config.NumberColumn("Away LB", step=0.1, format="%.1f"),
            "Home_PD": st.column_config.NumberColumn("Home PD", step=1),
            "Away_PD": st.column_config.NumberColumn("Away PD", step=1),
            "Ref_Boost_Home": st.column_config.NumberColumn("Ref Boost (pp)", step=0.1, format="%.1f"),
            "Mkt_Home": st.column_config.NumberColumn("Mkt Home $", min_value=1.01, step=0.01, format="%.2f"),
        },
    )
    st.session_state.match_data = edited

    col_up, col_dl, col_reset = st.columns(3)
    with col_up:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                new = pd.read_csv(uploaded)
                need = {"Home", "Away", "Home_Runm", "Away_Runm", "Home_LB",
                        "Away_LB", "Home_PD", "Away_PD", "Ref_Boost_Home", "Mkt_Home"}
                missing = need - set(new.columns)
                if missing:
                    st.error(f"Missing columns: {', '.join(sorted(missing))}")
                else:
                    if "Match" not in new.columns:
                        new["Match"] = new["Home"] + " vs " + new["Away"]
                    if "Lineup_Notes" not in new.columns:
                        new["Lineup_Notes"] = ""
                    st.session_state.match_data = new
                    st.success(f"Loaded {len(new)} matches.")
            except Exception as e:
                st.error(str(e))
    with col_dl:
        dl_buf = BytesIO()
        st.session_state.match_data.to_csv(dl_buf, index=False)
        st.download_button(
            "Download current data", dl_buf.getvalue(),
            file_name="nrl_round_data.csv", mime="text/csv",
        )
    with col_reset:
        if st.button("Reset to Round 15 defaults"):
            st.session_state.match_data = DEFAULT_DATA.copy()
            st.rerun()

# ═══════════════════════════  SEASON TRACKING  ═══════════════════════════════
with tab_season:
    st.subheader("2026 Ladder (through Round 14)")
    ladder = SEASON.sort_values("Win%", ascending=False).reset_index(drop=True)
    ladder.index = ladder.index + 1
    st.dataframe(
        ladder.style.format({"Win%": "{:.1f}%", "PD": "{:+d}"}).background_gradient(
            subset=["Win%"], cmap="RdYlGn", vmin=0, vmax=100,
        ),
        use_container_width=True,
    )

    st.subheader("Win % Distribution")
    fig_lad = px.bar(
        ladder.sort_values("Win%"), x="Win%", y="Team",
        orientation="h", color="Win%",
        color_continuous_scale="RdYlGn", template="plotly_dark",
    )
    fig_lad.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        yaxis_title="", xaxis_title="Win %", showlegend=False,
        coloraxis_showscale=False,
    )
    fig_lad.add_vline(x=50, line_dash="dot", line_color="#555")
    st.plotly_chart(fig_lad, use_container_width=True)

# ═══════════════════════════  BACKTESTING  ═══════════════════════════════════
with tab_backtest:
    st.subheader("Backtest Model Against Results")
    st.markdown(
        "Upload a CSV with: **Home, Away, Home_Runm, Away_Runm, Home_LB, Away_LB, "
        "Home_PD, Away_PD, Ref_Boost_Home, Mkt_Home, Actual_Winner**. "
        "Add a **Round** column for per-round breakdown."
    )

    bt_file = st.file_uploader("Upload historical CSV", type=["csv"], key="bt")

    if bt_file is not None:
        try:
            bt = pd.read_csv(bt_file)
            need = {
                "Home", "Away", "Home_Runm", "Away_Runm", "Home_LB", "Away_LB",
                "Home_PD", "Away_PD", "Ref_Boost_Home", "Mkt_Home", "Actual_Winner",
            }
            missing = need - set(bt.columns)
            if missing:
                st.error(f"Missing columns: {', '.join(sorted(missing))}")
            else:
                if "Match" not in bt.columns:
                    bt["Match"] = bt["Home"] + " vs " + bt["Away"]

                bt["Home_Win%"] = bt.apply(calculate_prob, axis=1)
                bt["Away_Win%"] = round(1 - bt["Home_Win%"], 4)
                bt["Pick"] = bt.apply(
                    lambda r: r["Home"] if r["Home_Win%"] >= 0.5 else r["Away"], axis=1,
                )
                bt["Pick_Prob"] = bt.apply(
                    lambda r: r["Home_Win%"] if r["Home_Win%"] >= 0.5 else r["Away_Win%"], axis=1,
                )
                bt["Correct"] = bt["Pick"] == bt["Actual_Winner"]
                bt["Edge_pp"] = round(
                    (bt["Home_Win%"] - (1 / bt["Mkt_Home"])) * 100, 1,
                )
                bt["Is_Value"] = bt["Edge_pp"].abs() >= edge_threshold
                bt["Pick_Mkt"] = bt.apply(
                    lambda r: r["Mkt_Home"] if r["Home_Win%"] >= 0.5 else (
                        round(1 / (1 - 1 / r["Mkt_Home"]), 2) if r["Mkt_Home"] > 1 else 99
                    ),
                    axis=1,
                )

                n = len(bt)
                correct = int(bt["Correct"].sum())
                accuracy = correct / n * 100
                brier = sum(
                    (p - a) ** 2
                    for p, a in zip(bt["Pick_Prob"], bt["Correct"].astype(int))
                ) / n

                vb = bt[bt["Is_Value"]]
                vb_n = len(vb)
                vb_correct = int(vb["Correct"].sum()) if vb_n else 0
                vb_rate = vb_correct / vb_n * 100 if vb_n else 0

                vb_profit = sum(
                    (r["Pick_Mkt"] - 1) if r["Correct"] else -1
                    for _, r in vb.iterrows()
                )
                vb_roi = vb_profit / vb_n * 100 if vb_n else 0

                mc = st.columns(4)
                mc[0].metric("Accuracy", f"{accuracy:.1f}% ({correct}/{n})")
                mc[1].metric("Brier Score", f"{brier:.4f}")
                mc[2].metric("Value Bet Win Rate", f"{vb_rate:.1f}% ({vb_correct}/{vb_n})")
                mc[3].metric("Value Bet ROI", f"{vb_roi:+.1f}%")

                # Results table
                st.markdown("#### All Picks")
                bt_show = bt[
                    ["Match", "Pick", "Pick_Prob", "Edge_pp", "Pick_Mkt", "Actual_Winner", "Correct"]
                ].copy()
                bt_show["Correct"] = bt_show["Correct"].map({True: "Yes", False: "No"})
                st.dataframe(
                    bt_show.style.format({
                        "Pick_Prob": "{:.1%}", "Edge_pp": "{:+.1f}", "Pick_Mkt": "${:.2f}",
                    }).map(
                        lambda v: "color:#3fb950" if v == "Yes" else (
                            "color:#f85149" if v == "No" else ""
                        ),
                        subset=["Correct"],
                    ),
                    use_container_width=True, hide_index=True,
                )

                # Per-round chart
                if "Round" in bt.columns:
                    st.markdown("#### Per-Round Accuracy")
                    by_r = bt.groupby("Round").agg(
                        Games=("Correct", "size"), Correct=("Correct", "sum"),
                    ).reset_index()
                    by_r["Acc%"] = round(by_r["Correct"] / by_r["Games"] * 100, 1)
                    by_r["Cumul"] = round(
                        by_r["Correct"].cumsum() / by_r["Games"].cumsum() * 100, 1,
                    )
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Bar(
                        x=by_r["Round"], y=by_r["Acc%"],
                        name="Round", marker_color="#58a6ff",
                    ))
                    fig_bt.add_trace(go.Scatter(
                        x=by_r["Round"], y=by_r["Cumul"],
                        name="Cumulative", mode="lines+markers",
                        line=dict(color="#3fb950", dash="dot"),
                    ))
                    fig_bt.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        yaxis_title="Accuracy %", yaxis_range=[0, 100],
                    )
                    fig_bt.add_hline(y=50, line_dash="dot", line_color="#555")
                    st.plotly_chart(fig_bt, use_container_width=True)

        except Exception as e:
            st.error(f"Backtest error: {e}")
    else:
        st.info("Upload a CSV to get started. Sample format:")
        sample = pd.DataFrame({
            "Round": [1, 1],
            "Home": ["Rabbitohs", "Storm"],
            "Away": ["Broncos", "Panthers"],
            "Home_Runm": [1687, 1740],
            "Away_Runm": [1520, 1710],
            "Home_LB": [6.3, 6.8],
            "Away_LB": [4.2, 6.4],
            "Home_PD": [44, 134],
            "Away_PD": [-66, 96],
            "Ref_Boost_Home": [-9.1, 4.2],
            "Mkt_Home": [1.48, 1.65],
            "Actual_Winner": ["Rabbitohs", "Storm"],
        })
        st.dataframe(sample, use_container_width=True, hide_index=True)
        sbuf = BytesIO()
        sample.to_csv(sbuf, index=False)
        st.download_button(
            "Download sample template", sbuf.getvalue(),
            file_name="backtest_template.csv", mime="text/csv",
        )
