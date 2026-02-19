# Technical Architecture: Pepe AI Pricing Engine (V5.0)
## Unified Specification: Phase 1 Foundations to ML-Driven Production Logic
**Author**: Amal Bobby | AI Data Science Division  
**Last Updated**: February 19, 2026

---

## 1. Objective
The Pepe Pricing Engine transitions pricing operations from static, rule-based logic to a dynamic, machine-learning-driven ecosystem. It generates data-backed price recommendations by simulating demand, inventory health, and net realization across Myntra, Flipkart, and AJIO.

The system is designed to be:
- **State-of-the-Art (ML)**: Moving beyond constant elasticity to non-linear XGBoost demand curves.
- **Slab-Aware**: Explicitly modeling platform-specific fee "cliffs" to protect margins.
- **Deterministic & Explainable**: Every "AI" decision is traceable to specific weight parameters.
- **Persistent**: Deep-linkable dashboard state for seamless stakeholder collaboration.

---

## 2. Technical Scope & Channels
The engine currently orchestrates intelligence across:
- **Myntra**: High-velocity fashion modeling.
- **Flipkart**: Category-specific fee "Uplift" slab modeling.
- **AJIO**: Normalized revenue modeling with fixed commission offsets.

---

## 3. Data Flow & Normalization
The engine utilizes a warehouse-native ingestion layer (AWS RDS PostgreSQL) with several critical normalization steps:

### 3.1 Historical Normalization (CTE: base_sales)
- **Status Filtering**: Exclusion of `CANCELLED` and `UNFULFILLABLE` orders to prevent demand inflation.
- **AJIO Offset**: `Selling Price + 65` normalization ensures all channels use a comparable "Base Price" for calculation.
- **Metadata Dedup**: Row-ranking logic ensures the latest Cost Price, MRP, and Category Name are always used.

### 3.2 Time-Windowing
The engine prioritizes the most relevant market signals by focusing on data starting **June 1, 2025**, ensuring the model doesn't over-rely on stale historical outliers.

---

## 4. The Mathematical "Triple-Engine"
The core intelligence is driven by three primary mathematical frameworks:

### 4.1 Lifecycle-Aware Recency Weighting
Signal strength is adjusted using an **Exponential Decay Function**:  
`Weight = exp(- Decay_Rate * Days_Since_Sale)`
- **NEW (Launches)**: Rate 0.010 (Fast adaptation).
- **MUDA (Slow-Movers)**: Rate 0.002 (Deep historical context).
- **Core Strategy**: Higher weights on recent sales allow the engine to detect "breaking" trends within 48-72 hours.

### 4.2 Bayesian Elasticity Transfer (Cold-Start Solution)
To handle products with sparse data ($N < 3$ price points), the engine performs **Category-Level Pooling**:
- **Mechanism**: The SKU-level signal is regularized against the "Peer Group" (the Lifecycle Tag).
- **Formula**: `final_e = ( (n_sku * e_sku) + (5 * e_category) ) / (n_sku + 5)`
- This prevents the "Cold-Start" problem where new products have no demand curve.

### 4.3 Inventory Urgency (Sigmoid System)
The engine calculates a stock health index (0.0 to 1.0) using a **Sigmoid function**:  
`Urgency = 1 / (1 + exp( - (Current_DOH - Target_DOH) / 15 ))`
- Higher urgency scores (e.g., DOH > 60) automatically shift the engine's priorities from "Profit Protection" to "Velocity/Clearance."

---

## 5. Machine Learning Architecture (XGBoost)
Moving beyond basic linear regression, the engine employs a Gradient Boosted Tree model (`pricing_ml_engine.py`) to capture complex feature interactions.

### 5.1 Advanced Feature Vectors (22 Signals)
The model evaluates every SKU-channel-price point against:
- **Price Factors**: Absolute Price, Price/MRP Ratio, Discount Depth.
- **Temporal Waves**: DOW, Month, Payday Spikes (Month-first/end flags).
- **Momentum**: 7d, 14d, and 30d Rolling Burn Rates (Velocity).
- **Platform Context**: Event vs. BAU status, Channel ID, and Category slabs.

### 5.2 Model Specs
- **Algorithm**: XGBoost (Histogram-based learning).
- **Depth**: max_depth: 6 (Balancing interaction detection vs. overfit).
- **Validation**: **TimeSeriesSplit** (3-Fold) ensures the model is validated on "relative future" data, simulating real-world performance.

---

## 6. Price Simulation & Slab-Aware Net Realization
The engine simulates a comprehensive "Price Ladder" (20 points per SKU) focused on **Net Realization** (True Bottom Line).

### 6.1 Simulation Grid
Included in the grid are **Psychological Anchors** (Charm Pricing like ₹X49, ₹X99) to verify conversion boundaries at the most common commercial price points.

### 6.2 Flipkart Uplift slabs (Category-Wise)
The engine models the specific "Uplift" (Fixed fee + Shipping) for every category:
- **Logic**: `Net_Realization = (Price * 0.82) - Category_Uplift_Fee`
- **Slab Mapping**: Sub-500, Sub-1000, and 1000+ cliffs are modeled for BOXER, BRIEF, T SHIRT, PYJAMA, etc.
- **Benefit**: The AI will not recommend a price of ₹505 if ₹499 results in higher Net Profit due to lower fees.

---

## 7. Multi-Objective Scoring Framework
Recommendations are chosen using a **Dynamic Weighted Composite Score** that adapts to inventory states.

| Mode | Trigger | Focus | Dominant Weight |
|:---|:---:|:---|:---:|
| **Margin Mode** | $\mu < 0.3$ | Yield protection | **Profit (0.55)** |
| **Balanced Mode** | $\mu = 0.3-0.7$ | Growth & Balance | **Profit (0.45) / DRR (0.30)** |
| **Velocity Mode** | $\mu > 0.7$ | Liquidation | **DRR (0.45) / DOH (0.25)** |

---

## 8. Dashboard & Consumption
The output is served via a high-performance Streamlit Dashboard (`dashboard.app`).

- **Pulse Loading**: A 7-step initialization UI provides transparent health checks for the AI pipeline.
- **Global Cache**: `@st.cache_data` (1-hour TTL) ensures instant responsiveness.
- **Deep Linking**: Dashboard selections are synced with URL Query Params, allowing users to share direct links to specific SKU views.
- **BI Native**: All results are persisted to `output/ml_recommendations.csv` for direct warehouse ingestion and Power BI consumption.

---

## 9. Future Roadmap (Phase 3 & 4)
1. **RTO Penalty Integration**: Penalizing price points that drive high Return-to-Origin rates in the profit formula.
2. **Quick-Commerce Expansion**: Launching pricing logic for **Blinkit, Swiggy Instamart, and Zepto**.
3. **Automated Feed Push**: Daily automated price updates to channel seller portals via API.

---
**Version**: 5.0 Stable  
**Technical Lead**: Antigravity / Amal Bobby AI Operations
