import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NRL Moneyball + Referee Factor",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #0e1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
    .metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; }
    .edge-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .edge-pos { background: rgba(63, 185, 80, 0.15); color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.3); }
    .edge-neg { background: rgba(248, 81, 73, 0.15); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.3); }
    .formula-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.82rem;
        color: #c9d1d9;
        line-height: 1.7;
    }
    div[data-testid="stExpander"] details {
        border: 1px solid #30363d;
        border-radius: 8px;
        background: #161b22;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Default Round 15 2026 data
# ---------------------------------------------------------------------------
DEFAULT_DATA = pd.DataFrame([
    {
        "Match": "Rabbitohs vs Broncos",
        "Home": "Rabbitohs",
        "Away": "Broncos",
        "Home_Runm": 1687,
        "Away_Runm": 1520,
        "Home_LB": 6.3,
        "Away_LB": 4.2,
        "Home_PD": 44,
        "Away_PD": -66,
        "Ref_Boost_Home": -9.1,
        "Mkt_Home_Price": 1.72,
        "Mkt_Away_Price": 2.20,
    },
    {
        "Match": "Dolphins vs Roosters",
        "Home": "Dolphins",
        "Away": "Roosters",
        "Home_Runm": 1580,
        "Away_Runm": 1645,
        "Home_LB": 5.1,
        "Away_LB": 5.8,
        "Home_PD": 12,
        "Away_PD": 82,
        "Ref_Boost_Home": 16.9,
        "Mkt_Home_Price": 2.87,
        "Mkt_Away_Price": 1.53,
    },
    {
        "Match": "Warriors vs Sharks",
        "Home": "Warriors",
        "Away": "Sharks",
        "Home_Runm": 1510,
        "Away_Runm": 1620,
        "Home_LB": 4.5,
        "Away_LB": 5.6,
        "Home_PD": -38,
        "Away_PD": 56,
        "Ref_Boost_Home": -31.3,
        "Mkt_Home_Price": 2.20,
        "Mkt_Away_Price": 1.75,
    },
    {
        "Match": "Eels vs Raiders",
        "Home": "Eels",
        "Away": "Raiders",
        "Home_Runm": 1555,
        "Away_Runm": 1590,
        "Home_LB": 4.8,
        "Away_LB": 5.2,
        "Home_PD": -22,
        "Away_PD": 18,
        "Ref_Boost_Home": 0.0,
        "Mkt_Home_Price": 2.10,
        "Mkt_Away_Price": 1.80,
    },
    {
        "Match": "Tigers vs Titans",
        "Home": "Tigers",
        "Away": "Titans",
        "Home_Runm": 1480,
        "Away_Runm": 1540,
        "Home_LB": 3.9,
        "Away_LB": 4.6,
        "Home_PD": -88,
        "Away_PD": -30,
        "Ref_Boost_Home": 0.0,
        "Mkt_Home_Price": 2.50,
        "Mkt_Away_Price": 1.58,
    },
])


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------
def run_model(df: pd.DataFrame, weights: dict, sensitivity: float,
              home_yard_adj: float, away_yard_adj: float) -> pd.DataFrame:
    """Run the prediction model on a DataFrame of matches."""
    results = []
    for _, row in df.iterrows():
        adj_home_runm = row["Home_Runm"] * (1 - home_yard_adj / 100)
        adj_away_runm = row["Away_Runm"] * (1 - away_yard_adj / 100)

        attack_diff = ((adj_home_runm - adj_away_runm) / 100) + (
            (row["Home_LB"] - row["Away_LB"]) * 2.5
        )
        defence_diff = (row["Home_PD"] - row["Away_PD"]) / 60
        home_advantage = 1.0
        ref_context = row["Ref_Boost_Home"] / 10

        raw_score = (
            weights["attack"] * attack_diff
            + weights["defence"] * defence_diff
            + weights["home"] * home_advantage
            + weights["context"] * ref_context
        )

        prob = 1 / (1 + np.exp(-raw_score * sensitivity))
        prob = np.clip(prob, 0.03, 0.97)

        mkt_impl_home = 1 / row["Mkt_Home_Price"] if row["Mkt_Home_Price"] > 0 else 0.5
        mkt_impl_away = 1 / row["Mkt_Away_Price"] if row["Mkt_Away_Price"] > 0 else 0.5

        fair_home = round(1 / prob, 2) if prob > 0.03 else 99.0
        fair_away = round(1 / (1 - prob), 2) if prob < 0.97 else 99.0

        edge_home = (prob - mkt_impl_home) * 100
        edge_away = ((1 - prob) - mkt_impl_away) * 100

        if prob >= 0.5:
            pick = row["Home"]
            pick_prob = prob
            pick_fair = fair_home
            pick_mkt = row["Mkt_Home_Price"]
            pick_edge = edge_home
        else:
            pick = row["Away"]
            pick_prob = 1 - prob
            pick_fair = fair_away
            pick_mkt = row["Mkt_Away_Price"]
            pick_edge = edge_away

        results.append({
            "Match": row["Match"],
            "Home": row["Home"],
            "Away": row["Away"],
            "Home_Win%": round(prob * 100, 1),
            "Away_Win%": round((1 - prob) * 100, 1),
            "Fair_Home": fair_home,
            "Fair_Away": fair_away,
            "Mkt_Home": row["Mkt_Home_Price"],
            "Mkt_Away": row["Mkt_Away_Price"],
            "Edge_Home_pp": round(edge_home, 1),
            "Edge_Away_pp": round(edge_away, 1),
            "Pick": pick,
            "Pick_Prob": round(pick_prob * 100, 1),
            "Pick_Fair": pick_fair,
            "Pick_Mkt": pick_mkt,
            "Pick_Edge_pp": round(pick_edge, 1),
            "Raw_Score": round(raw_score, 3),
            "Attack_Diff": round(attack_diff, 2),
            "Defence_Diff": round(defence_diff, 2),
            "Ref_Boost": row["Ref_Boost_Home"],
        })

    return pd.DataFrame(results)


def backtest(df: pd.DataFrame, weights: dict, sensitivity: float) -> pd.DataFrame:
    """Backtest against historical data with Actual_Winner column."""
    results = run_model(df, weights, sensitivity, 0, 0)
    results["Actual_Winner"] = df["Actual_Winner"].values
    results["Correct"] = results["Pick"] == results["Actual_Winner"]
    return results


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Model Weights")
    st.caption("Must sum to 1.00")

    w_attack = st.slider("Attack", 0.0, 1.0, 0.48, 0.01)
    w_defence = st.slider("Defence / Form", 0.0, 1.0, 0.22, 0.01)
    w_home = st.slider("Home Advantage", 0.0, 1.0, 0.12, 0.01)
    w_context = st.slider("Context (Ref + Lineup)", 0.0, 1.0, 0.18, 0.01)

    weight_sum = w_attack + w_defence + w_home + w_context
    if abs(weight_sum - 1.0) > 0.02:
        st.warning(f"Weights sum to {weight_sum:.2f}. Adjust to reach 1.00.")
    else:
        st.success(f"Weights: {weight_sum:.2f}")

    weights = {
        "attack": w_attack,
        "defence": w_defence,
        "home": w_home,
        "context": w_context,
    }

    st.markdown("---")
    st.markdown("## Logistic Sensitivity")
    st.caption("Controls how strongly stats differences affect probability. Lower = more conservative.")
    sensitivity = st.slider("Scale factor", 0.05, 0.60, 0.18, 0.01)

    st.markdown("---")
    st.markdown("## Yardage Adjustment %")
    st.caption("Reduce run metres for key absences (e.g. fullback out = 8-12%)")
    home_yard_adj = st.slider("Home yardage reduction", 0, 25, 0, 1)
    away_yard_adj = st.slider("Away yardage reduction", 0, 25, 0, 1)

    st.markdown("---")
    st.markdown("## Edge Threshold")
    edge_threshold = st.slider("Min edge (pp) to flag", 1.0, 15.0, 5.0, 0.5)

# ---------------------------------------------------------------------------
# Session state for match data
# ---------------------------------------------------------------------------
if "match_data" not in st.session_state:
    st.session_state.match_data = DEFAULT_DATA.copy()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# NRL Moneyball + Referee Factor")
st.caption("Round 15, 2026 - Transparent model. Editable data. Your weights, your edge.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_dash, tab_data, tab_formulas, tab_backtest = st.tabs(
    ["Dashboard", "Edit Data", "Model Logic", "Backtesting"]
)

# ==================== DASHBOARD TAB ========================================
with tab_dash:
    df = st.session_state.match_data
    results = run_model(df, weights, sensitivity, home_yard_adj, away_yard_adj)

    # --- Summary metrics ---
    value_bets = results[results["Pick_Edge_pp"].abs() >= edge_threshold]
    avg_edge = value_bets["Pick_Edge_pp"].mean() if len(value_bets) > 0 else 0

    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Games</div>
        </div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(value_bets)}</div>
            <div class="metric-label">Value Bets</div>
        </div>""", unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_edge:+.1f}pp</div>
            <div class="metric-label">Avg Edge</div>
        </div>""", unsafe_allow_html=True)
    with cols[3]:
        max_edge_row = results.loc[results["Pick_Edge_pp"].abs().idxmax()] if len(results) > 0 else None
        best = f"{max_edge_row['Pick_Edge_pp']:+.1f}pp" if max_edge_row is not None else "-"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{best}</div>
            <div class="metric-label">Best Edge</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # --- Results table ---
    st.markdown("### Predictions")
    display_df = results[["Match", "Pick", "Home_Win%", "Away_Win%",
                          "Fair_Home", "Fair_Away", "Mkt_Home", "Mkt_Away",
                          "Edge_Home_pp", "Edge_Away_pp", "Ref_Boost"]].copy()

    def highlight_edges(val):
        if isinstance(val, (int, float)):
            if val >= edge_threshold:
                return "background-color: rgba(63, 185, 80, 0.15); color: #3fb950; font-weight: 700"
            elif val <= -edge_threshold:
                return "background-color: rgba(248, 81, 73, 0.15); color: #f85149; font-weight: 700"
        return ""

    styled = display_df.style.applymap(
        highlight_edges, subset=["Edge_Home_pp", "Edge_Away_pp"]
    ).format({
        "Home_Win%": "{:.1f}%",
        "Away_Win%": "{:.1f}%",
        "Fair_Home": "${:.2f}",
        "Fair_Away": "${:.2f}",
        "Mkt_Home": "${:.2f}",
        "Mkt_Away": "${:.2f}",
        "Edge_Home_pp": "{:+.1f}",
        "Edge_Away_pp": "{:+.1f}",
        "Ref_Boost": "{:+.1f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Value bets highlight ---
    if len(value_bets) > 0:
        st.markdown(f"### Value Bets (edge >= {edge_threshold}pp)")
        for _, vb in value_bets.iterrows():
            edge_class = "edge-pos" if vb["Pick_Edge_pp"] > 0 else "edge-neg"
            st.markdown(f"""
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 1rem; margin-bottom: 0.8rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.1rem; font-weight: 700; color: #c9d1d9;">{vb['Pick']}</span>
                        <span style="color: #8b949e; margin-left: 8px;">in {vb['Match']}</span>
                    </div>
                    <div>
                        <span class="edge-badge {edge_class}">{vb['Pick_Edge_pp']:+.1f}pp edge</span>
                    </div>
                </div>
                <div style="display: flex; gap: 2rem; margin-top: 0.6rem; color: #8b949e; font-size: 0.85rem;">
                    <span>Model: <b style="color:#58a6ff">{vb['Pick_Prob']:.1f}%</b></span>
                    <span>Fair: <b style="color:#58a6ff">${vb['Pick_Fair']:.2f}</b></span>
                    <span>Market: <b style="color:#c9d1d9">${vb['Pick_Mkt']:.2f}</b></span>
                    <span>Ref Boost: <b style="color:#d2a8ff">{vb['Ref_Boost']:+.1f}pp</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No value bets detected at current threshold. Try lowering the edge threshold or adjusting weights.")

    # --- Charts ---
    st.markdown("### Win Probability")
    prob_df = results[["Match", "Home_Win%", "Away_Win%"]].melt(
        id_vars="Match", var_name="Side", value_name="Probability"
    )
    prob_df["Side"] = prob_df["Side"].map({"Home_Win%": "Home", "Away_Win%": "Away"})
    fig_prob = px.bar(
        prob_df, x="Match", y="Probability", color="Side",
        barmode="group",
        color_discrete_map={"Home": "#58a6ff", "Away": "#f85149"},
        template="plotly_dark",
    )
    fig_prob.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        yaxis_title="Win %",
        yaxis_range=[0, 100],
        legend_title_text="",
        margin=dict(l=40, r=20, t=30, b=60),
    )
    fig_prob.add_hline(y=50, line_dash="dot", line_color="#30363d")
    st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("### Edge Map (Model vs Market)")
    edge_df = results[["Match", "Edge_Home_pp", "Edge_Away_pp"]].melt(
        id_vars="Match", var_name="Side", value_name="Edge_pp"
    )
    edge_df["Side"] = edge_df["Side"].map({"Edge_Home_pp": "Home", "Edge_Away_pp": "Away"})
    fig_edge = px.bar(
        edge_df, x="Match", y="Edge_pp", color="Side",
        barmode="group",
        color_discrete_map={"Home": "#58a6ff", "Away": "#f85149"},
        template="plotly_dark",
    )
    fig_edge.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        yaxis_title="Edge (pp)",
        legend_title_text="",
        margin=dict(l=40, r=20, t=30, b=60),
    )
    fig_edge.add_hline(y=0, line_color="#30363d")
    fig_edge.add_hline(y=edge_threshold, line_dash="dot", line_color="#3fb950", opacity=0.5)
    fig_edge.add_hline(y=-edge_threshold, line_dash="dot", line_color="#f85149", opacity=0.5)
    st.plotly_chart(fig_edge, use_container_width=True)

    # --- Per-game breakdown ---
    st.markdown("### Per-Game Breakdown")
    for _, game in results.iterrows():
        with st.expander(f"{game['Match']} - Pick: {game['Pick']} ({game['Pick_Prob']:.1f}%)"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Attack Diff", f"{game['Attack_Diff']:.2f}")
                st.metric("Defence Diff", f"{game['Defence_Diff']:.2f}")
            with c2:
                st.metric("Raw Score", f"{game['Raw_Score']:.3f}")
                st.metric("Ref Boost", f"{game['Ref_Boost']:+.1f}pp")
            with c3:
                st.metric("Home Win%", f"{game['Home_Win%']:.1f}%")
                st.metric("Edge", f"{game['Pick_Edge_pp']:+.1f}pp")

            st.markdown("**Score components:**")
            atk_contrib = weights["attack"] * game["Attack_Diff"]
            def_contrib = weights["defence"] * game["Defence_Diff"]
            home_contrib = weights["home"] * 1.0
            ctx_contrib = weights["context"] * (game["Ref_Boost"] / 10)
            st.code(
                f"Attack:  {weights['attack']:.2f} x {game['Attack_Diff']:.2f} = {atk_contrib:.3f}\n"
                f"Defence: {weights['defence']:.2f} x {game['Defence_Diff']:.2f} = {def_contrib:.3f}\n"
                f"Home:    {weights['home']:.2f} x 1.00 = {home_contrib:.3f}\n"
                f"Context: {weights['context']:.2f} x {game['Ref_Boost']/10:.2f} = {ctx_contrib:.3f}\n"
                f"{'':=<45}\n"
                f"Raw Score: {game['Raw_Score']:.3f}\n"
                f"Prob = 1/(1+exp(-{game['Raw_Score']:.3f} x {sensitivity})) = {game['Home_Win%']:.1f}%"
            )

# ==================== DATA EDITOR TAB ======================================
with tab_data:
    st.markdown("### Edit Match Data")
    st.caption("Change stats, add referee boosts, update market prices. Changes apply instantly to the Dashboard.")

    edited = st.data_editor(
        st.session_state.match_data,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Match": st.column_config.TextColumn("Match", width="medium"),
            "Home": st.column_config.TextColumn("Home"),
            "Away": st.column_config.TextColumn("Away"),
            "Home_Runm": st.column_config.NumberColumn("Home Run m", min_value=0, step=10),
            "Away_Runm": st.column_config.NumberColumn("Away Run m", min_value=0, step=10),
            "Home_LB": st.column_config.NumberColumn("Home LB", min_value=0.0, step=0.1, format="%.1f"),
            "Away_LB": st.column_config.NumberColumn("Away LB", min_value=0.0, step=0.1, format="%.1f"),
            "Home_PD": st.column_config.NumberColumn("Home PD", step=1),
            "Away_PD": st.column_config.NumberColumn("Away PD", step=1),
            "Ref_Boost_Home": st.column_config.NumberColumn("Ref Boost (pp)", step=0.1, format="%.1f"),
            "Mkt_Home_Price": st.column_config.NumberColumn("Mkt Home $", min_value=1.01, step=0.01, format="%.2f"),
            "Mkt_Away_Price": st.column_config.NumberColumn("Mkt Away $", min_value=1.01, step=0.01, format="%.2f"),
        },
    )
    st.session_state.match_data = edited

    st.markdown("---")
    col_up, col_down, col_reset = st.columns(3)
    with col_up:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                new_data = pd.read_csv(uploaded)
                required = {"Home", "Away", "Home_Runm", "Away_Runm", "Home_LB", "Away_LB",
                            "Home_PD", "Away_PD", "Ref_Boost_Home", "Mkt_Home_Price", "Mkt_Away_Price"}
                if required.issubset(set(new_data.columns)):
                    if "Match" not in new_data.columns:
                        new_data["Match"] = new_data["Home"] + " vs " + new_data["Away"]
                    st.session_state.match_data = new_data
                    st.success(f"Loaded {len(new_data)} matches.")
                else:
                    missing = required - set(new_data.columns)
                    st.error(f"CSV missing columns: {', '.join(missing)}")
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
    with col_down:
        csv_buffer = BytesIO()
        st.session_state.match_data.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download CSV",
            csv_buffer.getvalue(),
            file_name="nrl_round_data.csv",
            mime="text/csv",
        )
    with col_reset:
        if st.button("Reset to Round 15 defaults"):
            st.session_state.match_data = DEFAULT_DATA.copy()
            st.rerun()

# ==================== MODEL LOGIC TAB ======================================
with tab_formulas:
    st.markdown("### How the Model Works")
    st.markdown("""
    The model combines four weighted signals into a composite score, then converts
    to a win probability using a logistic function. Every calculation is shown below.
    """)

    st.markdown("#### 1. Differentials")
    st.markdown("""<div class="formula-box">
    attack_diff = ((Home_Runm - Away_Runm) / 100) + ((Home_LB - Away_LB) * 2.5)<br>
    defence_diff = (Home_PD - Away_PD) / 60
    </div>""", unsafe_allow_html=True)
    st.caption("Run metres difference scaled by 100, line breaks weighted at 2.5x. "
               "Points differential scaled by 60 to normalise range.")

    st.markdown("#### 2. Context Factors")
    st.markdown("""<div class="formula-box">
    home_advantage = 1.0 &nbsp;&nbsp;(constant for home ground)<br>
    ref_context = Ref_Boost_Home / 10 &nbsp;&nbsp;(pp scaled to model units)
    </div>""", unsafe_allow_html=True)
    st.caption("Ref_Boost_Home is the net referee signal in percentage points. "
               "Positive = referee historically favours the home team. Divided by 10 for scaling.")

    st.markdown("#### 3. Composite Score")
    st.markdown(f"""<div class="formula-box">
    score = ({weights['attack']:.2f} x attack_diff) + ({weights['defence']:.2f} x defence_diff)
            + ({weights['home']:.2f} x home_advantage) + ({weights['context']:.2f} x ref_context)
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 4. Probability (Logistic Function)")
    st.markdown(f"""<div class="formula-box">
    Home_Win_Prob = 1 / (1 + exp(-score x {sensitivity:.2f}))
    </div>""", unsafe_allow_html=True)
    st.caption(f"Sensitivity = {sensitivity:.2f}. Higher values produce more extreme probabilities. "
               "Clamped to 3-97% range.")

    st.markdown("#### 5. Fair Odds and Edge")
    st.markdown("""<div class="formula-box">
    Fair_Home = round(1 / Home_Win_Prob, 2)<br>
    Fair_Away = round(1 / (1 - Home_Win_Prob), 2)<br>
    <br>
    Market_Implied_Home = 1 / Mkt_Home_Price<br>
    Edge_Home_pp = (Home_Win_Prob - Market_Implied_Home) x 100
    </div>""", unsafe_allow_html=True)
    st.caption(f"Value bet flagged when |Edge| >= {edge_threshold}pp. "
               "Positive edge = model says team is underpriced by the market.")

    st.markdown("#### 6. Yardage Adjustment")
    st.markdown("""<div class="formula-box">
    Adjusted_Runm = Runm x (1 - yardage_reduction / 100)
    </div>""", unsafe_allow_html=True)
    st.caption("Use the sidebar slider to reduce a team's run metres for key absences "
               "(e.g. fullback out = 8-12% reduction). Applied before attack_diff calculation.")

    # Logistic curve visualisation
    st.markdown("#### Logistic Curve (current sensitivity)")
    x_vals = np.linspace(-6, 6, 200)
    y_vals = 1 / (1 + np.exp(-x_vals * sensitivity))
    fig_logistic = go.Figure()
    fig_logistic.add_trace(go.Scatter(
        x=x_vals, y=y_vals * 100,
        mode="lines", line=dict(color="#58a6ff", width=2),
        name=f"scale={sensitivity:.2f}",
    ))
    fig_logistic.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        xaxis_title="Raw Score",
        yaxis_title="Home Win %",
        yaxis_range=[0, 100],
        margin=dict(l=40, r=20, t=30, b=40),
        height=300,
    )
    fig_logistic.add_hline(y=50, line_dash="dot", line_color="#30363d")
    st.plotly_chart(fig_logistic, use_container_width=True)

# ==================== BACKTESTING TAB ======================================
with tab_backtest:
    st.markdown("### Backtest Model Against Results")
    st.markdown("Upload a CSV with the same columns as the match data plus an **Actual_Winner** column. "
                "The model will run predictions and score accuracy.")

    bt_file = st.file_uploader("Upload historical CSV", type=["csv"], key="backtest_upload")

    if bt_file is not None:
        try:
            bt_data = pd.read_csv(bt_file)
            if "Actual_Winner" not in bt_data.columns:
                st.error("CSV must contain an 'Actual_Winner' column.")
            else:
                if "Match" not in bt_data.columns:
                    bt_data["Match"] = bt_data["Home"] + " vs " + bt_data["Away"]

                bt_results = backtest(bt_data, weights, sensitivity)

                n_total = len(bt_results)
                n_correct = bt_results["Correct"].sum()
                accuracy = n_correct / n_total * 100 if n_total > 0 else 0

                probs = bt_results["Pick_Prob"].values / 100
                actuals = bt_results["Correct"].astype(int).values
                brier = np.mean((probs - actuals) ** 2)

                value_picks = bt_results[bt_results["Pick_Edge_pp"].abs() >= edge_threshold]
                value_correct = value_picks["Correct"].sum() if len(value_picks) > 0 else 0
                value_acc = value_correct / len(value_picks) * 100 if len(value_picks) > 0 else 0

                mcols = st.columns(4)
                with mcols[0]:
                    st.metric("Total Matches", n_total)
                with mcols[1]:
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                with mcols[2]:
                    st.metric("Brier Score", f"{brier:.4f}")
                with mcols[3]:
                    st.metric("Value Bet Acc", f"{value_acc:.1f}% ({len(value_picks)})")

                st.markdown("#### Results")
                bt_display = bt_results[["Match", "Pick", "Pick_Prob", "Pick_Edge_pp",
                                         "Actual_Winner", "Correct"]].copy()
                bt_display["Correct"] = bt_display["Correct"].map({True: "Yes", False: "No"})

                def highlight_correct(val):
                    if val == "Yes":
                        return "background-color: rgba(63, 185, 80, 0.15); color: #3fb950"
                    elif val == "No":
                        return "background-color: rgba(248, 81, 73, 0.15); color: #f85149"
                    return ""

                st.dataframe(
                    bt_display.style.applymap(highlight_correct, subset=["Correct"]).format({
                        "Pick_Prob": "{:.1f}%",
                        "Pick_Edge_pp": "{:+.1f}pp",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                # Calibration chart
                st.markdown("#### Calibration")
                bt_results["Prob_Bin"] = pd.cut(
                    bt_results["Pick_Prob"],
                    bins=[0, 55, 60, 65, 70, 75, 80, 100],
                    labels=["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75-80%", "80%+"],
                )
                cal = bt_results.groupby("Prob_Bin", observed=True).agg(
                    predicted=("Pick_Prob", "mean"),
                    actual=("Correct", "mean"),
                    count=("Correct", "size"),
                ).reset_index()
                cal["actual"] = cal["actual"] * 100

                fig_cal = go.Figure()
                fig_cal.add_trace(go.Bar(
                    x=cal["Prob_Bin"], y=cal["actual"],
                    name="Actual Win%",
                    marker_color="#58a6ff",
                    text=cal["count"].apply(lambda c: f"n={c}"),
                    textposition="outside",
                ))
                fig_cal.add_trace(go.Scatter(
                    x=cal["Prob_Bin"], y=cal["predicted"],
                    name="Predicted%",
                    mode="lines+markers",
                    line=dict(color="#f0883e", dash="dot"),
                ))
                fig_cal.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                    yaxis_title="Win Rate %",
                    yaxis_range=[0, 100],
                    margin=dict(l=40, r=20, t=30, b=40),
                    barmode="group",
                )
                st.plotly_chart(fig_cal, use_container_width=True)

        except Exception as e:
            st.error(f"Backtest error: {e}")
    else:
        st.info("Upload a CSV with historical match data to run a backtest. "
                "Required columns: Home, Away, Home_Runm, Away_Runm, Home_LB, Away_LB, "
                "Home_PD, Away_PD, Ref_Boost_Home, Mkt_Home_Price, Mkt_Away_Price, Actual_Winner")

        st.markdown("#### Sample CSV format")
        sample = DEFAULT_DATA.head(2).copy()
        sample["Actual_Winner"] = [sample.iloc[0]["Home"], sample.iloc[1]["Away"]]
        st.dataframe(sample, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("NRL Moneyball + Referee Factor Dashboard. Not financial advice. Model is exploratory.")
