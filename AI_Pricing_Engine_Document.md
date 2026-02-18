# AI-Powered Dynamic Pricing Engine for Pepe
### Technical Document — February 2026

---

## 1. Executive Summary

We have built an **AI-powered pricing engine** that replaces the existing rule-based pricing simulation with a machine learning system capable of predicting demand at any price point across Myntra, Flipkart, and AJIO.

The engine was trained on **2.27 million historical orders** spanning 13 months (Jan 2025 – Feb 2026), covering **3,868 SKUs** across all three channels.

### Key Results

| Metric | Value |
|--------|-------|
| **SKU-channel combinations analyzed** | 4,932 |
| **Price points simulated** | 84,871 |
| **Recommendations generated** | 10,198 |

#### Estimated Monthly Profit Uplift

| Channel | Current Monthly Profit | AI-Optimized Profit | **Uplift** |
|---------|----------------------|---------------------|-----------|
| **Myntra** | ₹1.18 Cr | ₹1.45 Cr | **+22.6%** |
| **Flipkart** | ₹1.24 Cr | ₹1.88 Cr | **+51.9%** |
| **AJIO** | ₹54.3 L | ₹66.6 L | **+22.6%** |
| **Total** | **₹2.96 Cr** | **₹3.99 Cr** | **+34.9%** |

> The engine recommends **price increases for 68%** of SKU-channel combinations (avg +6.2%), **decreases for 21%** (clearing overstock), and **no change for 11%** (already optimal).
>
> All insights are now accessible via a **Live Interactive Dashboard** with real-time model retraining.

---

## 2. Problem Statement

The existing pricing simulation has several fundamental limitations:

| Limitation | Impact |
|---|---|
| **Fixed elasticity fallback** (-1.5 for cold-start SKUs) | ~40% of SKUs get a hardcoded guess instead of a data-driven elasticity |
| **No seasonality awareness** | Same pricing logic for BAU days and sale events; ignores day-of-week and monthly patterns |
| **Static recency weighting** | Same decay rate for MUDA (slow-movers) and NEW (fast-learners) |
| **No inventory urgency** | An overstocked MUDA SKU and a fast-selling CORE SKU get identical pricing treatment |
| **Siloed SKU learning** | Each SKU learns independently — no cross-SKU intelligence transfer |
| **No guardrail enforcement** | Simulated prices may violate agreed BAU/Event pricing bounds |
| **Linear demand model** | Assumes log-linear relationship; misses non-linear interactions |

---

## 3. Solution Architecture

The new engine has two complementary layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA FOUNDATION                          │
│  2.27M orders  │  Event calendar  │  Guardrails  │  Inv.   │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼                             ▼
┌──────────────────────┐  ┌───────────────────────────────────┐
│  ENHANCED SQL ENGINE │  │     XGBOOST ML ENGINE             │
│  (6 AI Enhancements) │  │  (22 Features, Demand Prediction) │
│                      │  │                                   │
│  • Split elasticity  │  │  • Time-series cross-validation   │
│  • Lifecycle decay   │  │  • Non-linear demand curves       │
│  • Bayesian transfer │  │  • Feature interaction modeling   │
│  • Urgency scoring   │  │  • BAU + Event dual predictions   │
│  • DOW + seasonality │  │  • Inventory-constrained output   │
│  • Dynamic weights   │  │                                   │
└──────────┬───────────┘  └──────────────┬────────────────────┘
           │                             │
           ▼                             ▼
┌─────────────────────────────────────────────────────────────┐
│              UNIFIED RECOMMENDATION OUTPUT                  │
│  Per SKU-Channel: MAX_PROFIT │ MAX_DRR │ OPTIMAL            │
│  + Urgency Score │ Elasticity Source │ Confidence Signals    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. AI Enhancements — Detailed

### 4.1 Event/BAU Split Elasticity

**What changed**: Instead of a single elasticity per SKU-channel, the engine now computes **separate elasticity curves for BAU days vs Event/sale days**.

**Why it matters**: During sale events, customer behavior is fundamentally different — higher intent, more price-sensitive, higher volumes. A single elasticity averages out this critical distinction.

**Data source**: `pepe_event_calendar` table (BAU/Event flag per channel per day).

---

### 4.2 Lifecycle-Aware Demand Decay

**What changed**: The recency weighting now adapts to each SKU's lifecycle stage based on the Tags field in the item master.

| Tag | Decay Rate | Memory Window | Rationale |
|-----|-----------|---------------|-----------|
| **MUDA** (slow-movers) | 0.002 | ~500 days | Need maximum historical context for thin data |
| **NEW** (launches) | 0.010 | ~100 days | Only recent data matters; early signals change fast |
| **SUPER40** (top performers) | 0.006 | ~170 days | Slightly faster learning |
| **CORE** (staples) | 0.004 | ~250 days | Stable demand, balanced memory |
| **REGULAR** (default) | 0.005 | ~200 days | Standard balanced decay |

---

### 4.3 Bayesian Cross-SKU Elasticity Transfer

**What changed**: SKUs with fewer than 3 observed price points no longer fall back to a hardcoded elasticity of -1.5. Instead, they **borrow elasticity from peer SKUs** in the same tag (lifecycle) category.

**Formula**:
```
final_elasticity = (n_sku × sku_elasticity + 5 × category_elasticity) / (n_sku + 5)
```

Where `n_sku` is the number of data points for the specific SKU, and `5` is a regularization anchor. SKUs with rich data trust their own signal; thin-data SKUs lean on category norms.

**Impact**: Every SKU now gets a data-backed elasticity estimate rather than a static guess.

---

### 4.4 Inventory Urgency Scoring

**What changed**: A continuous urgency score (0–1) drives pricing strategy based on days-of-hand inventory relative to tag-specific targets.

| Tag | Target DOH | Rationale |
|-----|-----------|-----------|
| MUDA | 30 days | Aggressive clearance target |
| NEW | 45 days | Moderate — learning phase |
| SUPER40 | 50 days | Balanced growth |
| REGULAR | 60 days | Standard |
| CORE | 90 days | Hold for stable revenue |

**Urgency score** = sigmoid function: `1 / (1 + exp(-(current_DOH - target_DOH) / 15))`

- **Score > 0.7 (3,566 SKUs)**: Overstocked → engine shifts to **velocity mode** (prioritize sales volume over margin)
- **Score 0.3–0.7 (859 SKUs)**: Balanced → standard optimization
- **Score < 0.3 (507 SKUs)**: Limited stock → engine shifts to **margin protection mode**

---

### 4.5 Day-of-Week & Monthly Seasonality

**What changed**: The engine now models channel-specific demand multipliers for day-of-week and month.

Observed patterns from the data:

| Day | Orders Index | Insight |
|-----|-------------|---------|
| Sunday | 0.60× | 40% lower demand |
| Monday | 1.40× | Peak day |
| Tuesday | 1.50× | Peak day |
| Friday | 1.30× | High demand |

Monthly swings range from **0.5× (Aug/Dec)** to **1.5× (Jun)**, a **3× seasonal multiplier range** that the old engine completely ignored.

---

### 4.6 Dynamic Multi-Objective Optimization

**What changed**: The scoring weights are no longer fixed. They automatically adapt based on each SKU's inventory situation:

| Inventory State | Profit Weight | DRR Weight | DOH Weight | Proximity Weight |
|----------------|:---:|:---:|:---:|:---:|
| **Overstock** (urgency > 0.7) | 0.20 | **0.45** | 0.25 | 0.10 |
| **Balanced** (0.3–0.7) | 0.45 | 0.30 | 0.15 | 0.10 |
| **Limited Stock** (< 0.3) | **0.55** | 0.15 | 0.20 | 0.10 |

This means overstocked SKUs automatically get more aggressive pricing to clear inventory, while limited-stock SKUs protect margins.

---

## 5. XGBoost Machine Learning Model

### 5.1 Training Data

| Dimension | Value |
|---|---|
| Training records | 111,418 daily-SKU-channel-price observations |
| Date range | Jan 2025 – Feb 2026 (13 months) |
| Channels | Myntra, Flipkart, AJIO |
| SKUs | 3,868 unique |
| Validation | Time-series cross-validation (3 folds) |

### 5.2 Feature Engineering (22 Features)

| Category | Features |
|---|---|
| **Price signals** | Price, price-to-average ratio, discount depth (% off MRP), margin %, price bucket, price volatility |
| **Temporal** | Day-of-week, month, day-of-month, week-of-year, is_weekend, is_month_start, is_month_end |
| **Demand history** | Rolling avg units (7d, 14d, 30d) |
| **Context** | Event flag (BAU/Event), tag (lifecycle), channel |
| **Recency** | Days since observation, lifecycle-aware recency weight |

### 5.3 Model Performance

| Fold | MAE | RMSE | R² | Notes |
|------|-----|------|-----|-------|
| 1 | 2.28 | 10.31 | 0.02 | Limited training data (early time window) |
| 2 | 1.41 | 5.36 | 0.24 | Model learning improves with more data |
| **3** | **1.44** | **6.40** | **0.36** | **Best performance, most mature model** |

### 5.4 Top Feature Importances

The XGBoost model reveals which factors most influence purchasing decisions:

| Rank | Feature | Importance | Business Insight |
|------|---------|-----------|-----------------|
| 1 | **Channel** | 11.6% | Each channel has fundamentally different demand patterns |
| 2 | **Weekend flag** | 11.2% | Weekday vs weekend is a major demand driver |
| 3 | **Month-start** | 8.2% | Salary cycle drives purchasing; strong first-week effect |
| 4 | **7-day rolling demand** | 7.6% | Recent momentum is a strong predictor |
| 5 | **Month-end** | 7.0% | End-of-month buying patterns differ |
| 6 | **Margin %** | 6.7% | Higher-margin products have distinct demand curves |
| 7 | **Price ratio to avg** | 6.5% | How far price is from the SKU's typical selling price matters |

---

## 6. Recommendation Summary by Channel

### Myntra

| Metric | Value |
|---|---|
| SKUs analyzed | 2,266 |
| Avg optimal price | ₹556 |
| Avg predicted DRR | 0.59 units/day |
| Avg monthly profit per SKU | ₹6,383 |
| **Total monthly profit** | **₹1.45 Cr** |

### Flipkart

| Metric | Value |
|---|---|
| SKUs analyzed | 1,647 |
| Avg optimal price | ₹473 |
| Avg predicted DRR | 1.10 units/day |
| Avg monthly profit per SKU | ₹11,394 |
| **Total monthly profit** | **₹1.88 Cr** |

### AJIO

| Metric | Value |
|---|---|
| SKUs analyzed | 1,019 |
| Avg optimal price | ₹570 |
| Avg predicted DRR | 0.60 units/day |
| Avg monthly profit per SKU | ₹6,532 |
| **Total monthly profit** | **₹66.6 L** |

---

## 7. Pricing Action Distribution

The engine doesn't just recommend increases — it takes a nuanced, per-SKU approach:

| Action | SKU Count | % | Explanation |
|--------|-----------|---|-------------|
| **Price Increase** | 3,358 | 68.1% | SKUs that are underpriced relative to their demand curve |
| **Price Decrease** | 1,048 | 21.2% | Overstocked SKUs where lower price accelerates sell-through |
| **No Change** | 526 | 10.7% | Already at or near optimal pricing |

**Average recommended change**: +6.2% (net across all SKUs)

---

## 8. Interactive Management Dashboard

We have deployed a premium, high-fidelity Streamlit dashboard to allow stakeholders to explore and act on these recommendations.

### 8.1 Core Features
- **Slicer-Driven Discovery**: Filter by Channel, SKU, and Lifecycle Tag (CORE, MUDA, etc.) to drill down into specific focus areas.
- **Dynamic DRR/DOH Curve**: Visualization of the predicted relationship between Price and Velocity, with markers for **Current**, **Max Profit**, and **Max DRR** price points.
- **AI-Generated Insights**: Context-aware suggestions (e.g., "Increase price by 12% to balance profit" or "Replenish stock to unlock insights" for low-inventory SKUs).
- **KPI Performance Cards**: Instant visibility into Current vs. Optimal pricing and the associated monthly profit potential.

### 8.2 Live Operational Flow
The dashboard has been upgraded for production readiness:
- **Zero CSV Dependency**: Connects directly to the model pipeline; data is always fresh.
- **Auto-Refresh (1-hr Cache)**: Models are automatically retrained and re-simulated every hour to adapt to the latest market signals.
- **Stakeholder Ready**: Download any filtered set of recommendations as a CSV for immediate upload to channel partner portals.

---

## 9. Output Deliverables & Security

| Deliverable | Description | Access |
|---|---|---|
| **Live Dashboard** | Interactive pricing explorer (Streamlit v1.31+) | `localhost:8501` / Deployable to Cloud |
| `ml_predictions.csv` | Full simulation: 84,871 price-point predictions | Generated on demand |
| `ml_recommendations.csv` | Top picks (MAX_PROFIT, MAX_DRR, OPTIMAL) | Generated on demand |
| `secrets.toml` | Secured DB credentials (Streamlit Secrets) | `.streamlit/` |

---

## 10. Technical Stack

| Component | Technology |
|---|---|
| Database | PostgreSQL (AWS RDS) |
| Dashboard | Streamlit (Python) |
| ML Engine | XGBoost 3.2 |
| Data Processing | Pandas + NumPy |
| Security | Streamlit Secrets Management |

---

## 11. Next Steps & Roadmap

| Phase | Enhancement | Expected Impact |
|---|---|---|
| **Phase 2** | Add return-cost adjustment (RTO penalty in profit formula) | +3-5% profit accuracy |
| **Phase 2** | Deploy to Streamlit Community Cloud for stakeholder shared access | Remote accessibility |
| **Phase 3** | Real-time model retraining pipeline (daily) | Faster adaptation to trends |
| **Phase 4** | Automated price push to channel partners | Reduce manual lag to zero |

---

## 12. How to Run

```bash
# Navigate to engine directory
cd /Users/mac/Documents/pricing\ suggestion

# Activate environment
source venv/bin/activate

# Run the interactive dashboard
streamlit run dashboard.py
```

The app will pull live data, train the models (~40s), and serve the dashboard at **http://localhost:8501**.

---

*Document prepared: February 18, 2026*
*Engine Version: 2.0*
*Data coverage: January 2025 – February 2026*
