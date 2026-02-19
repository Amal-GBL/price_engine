#!/usr/bin/env python3
"""
PEPE AI Pricing Engine — Dashboard
Run:  source venv/bin/activate && streamlit run dashboard.py
"""

# Version: 2.1 - Sync Fix
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from pricing_ml_engine import run_full_pipeline

# ── Page Config ──────────────────────────────────────
st.set_page_config(page_title="Pepe Pricing Engine", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 1rem; max-width: 1400px; }
#MainMenu, footer { visibility: hidden; }

.kpi-row { display: flex; gap: 0.8rem; margin: 1rem 0 1.5rem 0; }
.kpi-card {
    flex: 1; background: #1a1a2e; border: 1px solid rgba(139,92,246,0.12);
    border-radius: 10px; padding: 0.9rem 1.1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.kpi-card .kpi-label {
    color: rgba(255,255,255,0.45); font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
}
.kpi-card .kpi-value {
    color: #fff; font-size: 1.35rem; font-weight: 700; margin-top: 0.15rem;
}
.kpi-card .kpi-sub { color: rgba(255,255,255,0.35); font-size: 0.72rem; }

.insight-box {
    background: #0f1f3d; border: 1px solid rgba(59,130,246,0.2);
    border-left: 4px solid #3b82f6; border-radius: 8px;
    padding: 1rem 1.2rem; margin: 1rem 0;
}
.insight-box h4 { color: #60a5fa; font-size: 0.8rem; margin: 0 0 0.4rem 0; text-transform: uppercase; letter-spacing: 0.04em; }
.insight-box p { color: rgba(255,255,255,0.8); font-size: 0.88rem; line-height: 1.55; margin: 0; }

section[data-testid="stSidebar"] {
    background-color: #0e1117;
    border-right: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] h2 {
    color: #fff !important;
    font-size: 1.25rem !important;
    margin-bottom: 10px !important;
}

/* Distinctive Sidebar Input Boxes */
div[data-baseweb="select"], div[data-baseweb="input"] {
    background-color: #1a1a2e !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"]:hover, div[data-baseweb="input"]:hover {
    border-color: #8b5cf6 !important;
}

/* Fade-in Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.stApp {
    animation: fadeIn 0.8s ease-out;
}
</style>
""", unsafe_allow_html=True)


# ── Load Data (Global 1-hr Cache) ────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_analysis(_progress_callback=None):
    # The underscore prefix tells Streamlit: "Don't try to hash this parameter"
    # This fixed the UnhashableParamError while keeping the progress tracker
    preds, recs = run_full_pipeline(fast_mode=True, progress_callback=_progress_callback)
    return preds, recs

# Premium Loading Experience
if "engine_ready" not in st.session_state:
    status_box = st.empty()
    with status_box.status("🚀 Initializing AI Pricing Engine...", expanded=True) as status:
        def update_status(msg):
            status.write(msg)
        
        # This will either fetch from cache (instant) or run with log updates (step-by-step)
        preds, recs = get_cached_analysis(_progress_callback=update_status)
        status.update(label="✅ Engine Ready", state="complete", expanded=False)
    
    st.session_state.engine_ready = True
    status_box.empty() # Clean up the status box
else:
    preds, recs = get_cached_analysis()


# ── State Persistence (Query Params) ────────────────
# Sync URL query parameters with session state
if "sel_sku" not in st.session_state:
    st.session_state.sel_sku = st.query_params.get("sku", None)
if "sel_channel" not in st.session_state:
    st.session_state.sel_channel = st.query_params.get("channel", None)

def update_params():
    if st.session_state.sel_sku:
        st.query_params["sku"] = st.session_state.sel_sku
    if st.session_state.sel_channel:
        st.query_params["channel"] = st.session_state.sel_channel


# ── Build lookup tables ──────────────────────────────
# For each SKU-channel, get the 3 recommended prices
@st.cache_data
def build_rec_lookup(_recs):
    lookup = {}
    for _, r in _recs.iterrows():
        key = (r["sku"], r["channel"])
        if key not in lookup:
            lookup[key] = {}
        for rtype in r["recommendation_type"].split(" | "):
            rtype = rtype.strip()
            lookup[key][rtype] = r
    return lookup

rec_lookup = build_rec_lookup(recs)


# ── Build summary table ─────────────────────────────
@st.cache_data
def build_summary_table(_preds, _recs):
    # Get unique SKU-channel combos
    combos = _preds[["sku", "channel", "tags", "last_price", "cost_price", "inventory"]].drop_duplicates(
        subset=["sku", "channel"]
    )

    # Get max_drr, max_profit, optimal prices
    max_drr = _preds[_preds["is_max_drr"] == True][["sku", "channel", "price", "ml_drr"]].rename(
        columns={"price": "max_drr_price", "ml_drr": "max_drr_drr"}
    )
    max_profit = _preds[_preds["is_max_profit"] == True][["sku", "channel", "price", "ml_monthly_profit"]].rename(
        columns={"price": "max_profit_price", "ml_monthly_profit": "max_profit_profit"}
    )
    optimal = _preds[_preds["is_optimal"] == True][["sku", "channel", "price"]].rename(
        columns={"price": "optimal_price"}
    )

    df = combos.merge(max_drr, on=["sku", "channel"], how="left")
    df = df.merge(max_profit, on=["sku", "channel"], how="left")
    df = df.merge(optimal, on=["sku", "channel"], how="left")
    return df

summary_table = build_summary_table(preds, recs)


# ── Sidebar Slicers ──────────────────────────────────
with st.sidebar:
    st.markdown("## Pricing Engine")
    st.caption("Filter & explore SKU pricing")
    st.divider()

    # Channel
    all_channels = sorted(preds["channel"].unique())
    sel_channels = st.multiselect("Channel", all_channels, default=all_channels)

    # Tags
    all_tags = sorted([t for t in preds["tags"].dropna().unique()])
    sel_tags = st.multiselect("Lifecycle Tag", all_tags, default=all_tags)
    inc_untagged = st.checkbox("Include untagged", value=True)

    # Filter SKUs based on channel + tag selection
    mask = preds["channel"].isin(sel_channels)
    if inc_untagged:
        mask &= (preds["tags"].isin(sel_tags) | preds["tags"].isna())
    else:
        mask &= preds["tags"].isin(sel_tags)
    filtered = preds[mask]

    # SKU selector
    sku_options = sorted(filtered["sku"].unique())
    
    # Handle initial index from session state
    initial_sku_idx = 0
    if st.session_state.sel_sku in sku_options:
        initial_sku_idx = sku_options.index(st.session_state.sel_sku)

    st.selectbox(
        "SKU", sku_options, 
        index=initial_sku_idx if sku_options else None,
        key="sel_sku",
        on_change=update_params
    )

    # Channel for this SKU
    if st.session_state.sel_sku:
        sku_channels = sorted(filtered[filtered["sku"] == st.session_state.sel_sku]["channel"].unique())
        
        initial_chan_idx = 0
        if st.session_state.sel_channel in sku_channels:
            initial_chan_idx = sku_channels.index(st.session_state.sel_channel)

        st.selectbox(
            "Channel for SKU", sku_channels, 
            index=initial_chan_idx if sku_channels else None,
            key="sel_channel",
            on_change=update_params
        )
    else:
        st.session_state.sel_channel = None

    st.divider()
    st.caption("v2.2 · Dynamic AI Engine")


# ── Get SKU data ─────────────────────────────────────
if st.session_state.sel_sku and st.session_state.sel_channel:
    sku_data = filtered[
        (filtered["sku"] == st.session_state.sel_sku) & (filtered["channel"] == st.session_state.sel_channel)
    ].sort_values("price")

    key = (st.session_state.sel_sku, st.session_state.sel_channel)
    sku_recs = rec_lookup.get(key, {})

    last_price = sku_data["last_price"].iloc[0] if len(sku_data) > 0 else 0
    cost_price = sku_data["cost_price"].iloc[0] if len(sku_data) > 0 else 0
    inventory = sku_data["inventory"].iloc[0] if len(sku_data) > 0 else 0
    tag = sku_data["tags"].iloc[0] if len(sku_data) > 0 and pd.notna(sku_data["tags"].iloc[0]) else "—"

    # Current DRR/DOH (at closest price to last_price)
    if len(sku_data) > 0:
        closest_idx = (sku_data["price"] - last_price).abs().idxmin()
        current_drr = sku_data.loc[closest_idx, "ml_drr"]
        current_doh = sku_data.loc[closest_idx, "ml_doh"]
    else:
        current_drr = current_doh = 0

    max_drr_rec = sku_recs.get("MAX_DRR")
    max_profit_rec = sku_recs.get("MAX_PROFIT")
    optimal_rec = sku_recs.get("OPTIMAL")

    max_drr_price = int(max_drr_rec["price"]) if max_drr_rec is not None else "—"
    max_profit_price = int(max_profit_rec["price"]) if max_profit_rec is not None else "—"
    optimal_price = int(optimal_rec["price"]) if optimal_rec is not None else "—"

    # ── Header ───────────────────────────────────────
    st.markdown(f"### {sel_sku} · {sel_sku_channel}")
    st.caption(f"Tag: **{tag}** · Inventory: **{inventory}** units · Cost: **₹{cost_price:,.0f}**")

    # ── KPIs ─────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Current Price</div>
            <div class="kpi-value">₹{last_price:,.0f}</div>
            <div class="kpi-sub">last transacted</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Current DRR</div>
            <div class="kpi-value">{current_drr:.2f}</div>
            <div class="kpi-sub">units/day</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Current DOH</div>
            <div class="kpi-value">{f"{current_doh:,.0f}" if pd.notna(current_doh) else "—"}</div>
            <div class="kpi-sub">days on hand</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Max DRR Price</div>
            <div class="kpi-value" style="color:#22c55e;">₹{max_drr_price}</div>
            <div class="kpi-sub">highest velocity</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Max Profit Price</div>
            <div class="kpi-value" style="color:#f59e0b;">₹{max_profit_price}</div>
            <div class="kpi-sub">highest margin</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Optimal Price</div>
            <div class="kpi-value" style="color:#8b5cf6;">₹{optimal_price}</div>
            <div class="kpi-sub">balanced</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── DRR + DOH Curve ──────────────────────────────
    if len(sku_data) > 0:
        fig = go.Figure()

        # DRR line
        fig.add_trace(go.Scatter(
            x=sku_data["price"], y=sku_data["ml_drr"],
            mode="lines+markers", name="DRR (units/day)",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=5), yaxis="y1",
            hovertemplate="₹%{x} → DRR: %{y:.2f}<extra></extra>",
        ))

        # DOH line (secondary axis)
        fig.add_trace(go.Scatter(
            x=sku_data["price"], y=sku_data["ml_doh"],
            mode="lines+markers", name="DOH (days)",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            marker=dict(size=5), yaxis="y2",
            hovertemplate="₹%{x} → DOH: %{y:.0f} days<extra></extra>",
        ))

        # Mark current price
        fig.add_trace(go.Scatter(
            x=[last_price], y=[current_drr],
            mode="markers", name="Current Price",
            marker=dict(color="#ef4444", size=14, symbol="diamond"),
            hovertemplate=f"Current: ₹{last_price}<extra></extra>",
        ))

        # Mark 3 recommended prices
        markers = [
            ("MAX_DRR", "#22c55e", "triangle-up", max_drr_rec),
            ("MAX_PROFIT", "#f59e0b", "square", max_profit_rec),
            ("OPTIMAL", "#8b5cf6", "star", optimal_rec),
        ]
        for label, color, symbol, rec in markers:
            if rec is not None:
                fig.add_trace(go.Scatter(
                    x=[rec["price"]], y=[rec["ml_drr"]],
                    mode="markers", name=label,
                    marker=dict(color=color, size=14, symbol=symbol, line=dict(width=1, color="white")),
                    hovertemplate=f"{label}: ₹{rec['price']:.0f}<br>DRR: {rec['ml_drr']:.2f}<extra></extra>",
                ))

        fig.update_layout(
            title=dict(text="DRR & DOH at Various Price Points", font=dict(size=14, color="white")),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)"),
            xaxis=dict(title="Price (₹)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="DRR (units/day)", gridcolor="rgba(255,255,255,0.06)", side="left"),
            yaxis2=dict(title="DOH (days)", overlaying="y", side="right", gridcolor="rgba(255,255,255,0.03)"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=10)),
            height=420, margin=dict(t=40, b=60, l=50, r=50),
        )
        st.plotly_chart(fig, key="drr_doh_chart")

    # ── AI Insights ──────────────────────────────────
    if sku_recs:
        insights = []
        
        if inventory < 10:
            insights = ["Replenish stock to unlock insights"]
        else:
            # Price direction
            if optimal_rec is not None:
                opt_p = optimal_rec["price"]
                delta = opt_p - last_price
                pct = delta / last_price * 100 if last_price > 0 else 0
                if delta > 0:
                    insights.append(f"<b>Increase price</b> from ₹{last_price:,.0f} to ₹{opt_p:,.0f} ({pct:+.1f}%) for optimal balance of profit and velocity.")
                elif delta < 0:
                    insights.append(f"<b>Decrease price</b> from ₹{last_price:,.0f} to ₹{opt_p:,.0f} ({pct:+.1f}%) to accelerate sell-through.")
                else:
                    insights.append(f"<b>Hold current price</b> at ₹{last_price:,.0f} — already near optimal.")

            # DRR change
            if optimal_rec is not None:
                opt_drr = optimal_rec["ml_drr"]
                drr_change = opt_drr - current_drr
                if abs(drr_change) > 0.01:
                    insights.append(f"At optimal price, DRR moves from <b>{current_drr:.2f}</b> → <b>{opt_drr:.2f}</b> units/day ({drr_change:+.2f}).")

            # Profit
            if max_profit_rec is not None:
                mp = max_profit_rec["ml_monthly_profit"]
                insights.append(f"Max profit price (₹{max_profit_price}) yields <b>₹{mp:,.0f}/month</b> estimated profit.")

            # Max DRR
            if max_drr_rec is not None:
                md = max_drr_rec["ml_drr"]
                # Use a small threshold (2% or 0.02 units)
                threshold = max(0.02, current_drr * 0.02)
                if md > current_drr + threshold:
                    insights.append(f"Max DRR price (₹{max_drr_price}) pushes velocity to <b>{md:.2f} units/day</b>.")
                else:
                    insights.append(f"Current price already achieves near-maximum velocity (<b>{current_drr:.2f} units/day</b>).")



            # Urgency
            if optimal_rec is not None:
                urg = optimal_rec.get("urgency_score", 0.5)
                if urg > 0.7:
                    insights.append(f"<b>Overstock alert</b> (urgency {urg:.2f}) — prioritize velocity over margin.")
                elif urg < 0.3:
                    insights.append(f"<b>Limited stock</b> (urgency {urg:.2f}) — protect margins, avoid aggressive discounting.")

            # Tag insight
            if tag == "MUDA":
                insights.append("MUDA (slow-mover) — consider aggressive pricing to clear aged inventory.")
            elif tag == "NEW":
                insights.append("NEW launch — price discovery phase, model relies on recent trends only.")
            elif tag == "CORE":
                insights.append("CORE staple — stable demand, prioritize long-term margin.")

        st.markdown(
            '<div class="insight-box">'
            '<h4>AI Insights</h4>'
            '<p>' + '<br>'.join(insights) + '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

else:
    st.info("Select a SKU and channel from the sidebar to view pricing analysis.")


# ── Summary Table ────────────────────────────────────
st.divider()
st.markdown("### All SKU Recommendations")

# Filter summary table by selected channels & tags
tbl_mask = summary_table["channel"].isin(sel_channels)
if inc_untagged:
    tbl_mask &= (summary_table["tags"].isin(sel_tags) | summary_table["tags"].isna())
else:
    tbl_mask &= summary_table["tags"].isin(sel_tags)
filtered_table = summary_table[tbl_mask].copy()

display_cols = {
    "sku": "SKU",
    "channel": "Channel",
    "tags": "Tag",
    "last_price": "Last Price (₹)",
    "cost_price": "Cost Price (₹)",
    "inventory": "Inventory",
    "max_drr_price": "Max DRR Price (₹)",
    "max_profit_price": "Max Profit Price (₹)",
    "optimal_price": "Optimal Price (₹)",
}

display_df = filtered_table[list(display_cols.keys())].rename(columns=display_cols)
display_df = display_df.sort_values(["Channel", "SKU"])

st.dataframe(
    display_df,
    height=450,
    column_config={
        "Last Price (₹)": st.column_config.NumberColumn(format="₹%d"),
        "Cost Price (₹)": st.column_config.NumberColumn(format="₹%d"),
        "Max DRR Price (₹)": st.column_config.NumberColumn(format="₹%d"),
        "Max Profit Price (₹)": st.column_config.NumberColumn(format="₹%d"),
        "Optimal Price (₹)": st.column_config.NumberColumn(format="₹%d"),
    },
)

csv = display_df.to_csv(index=False)
st.download_button("Download as CSV", csv, "pricing_recommendations.csv", "text/csv")
