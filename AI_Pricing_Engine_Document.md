# Technical Specification: AI-Powered Dynamic Pricing Engine
## Mathematical Foundations, ML Architecture, and Orchestration Logic
**Document Version: 3.0**

---

## 1. Data Foundation & Ingestion Logic

The engine sits atop a PostgreSQL (AWS RDS) data warehouse. The ingestion layer handles **2.27 million sales records** and **85,000+ simulation points**.

### 1.1 SQL Aggregation Strategy
Data is extracted using a high-density CTE-based SQL engine that performs several critical preprocessing steps in-db:
- **Deduplication**: `mapping_dedup` CTE enforces a `ROW_NUMBER() OVER(PARTITION BY "Product Code" ORDER BY "updatedAt" DESC)` to handle record duplication in the `itemmaster` table.
- **Event Alignment**: Join against a granular channel-date calendar to assign BAU vs. Event flags.
- **Pricing Normalization**: Standardizes selling prices across channels by compensating for channel-specific overheads (e.g., AJIO +65 offset logic).

### 1.2 Training Data Windowing
The production pipeline implements a sliding window for memory optimization:
- **Historical Scope**: Jan 2025 onwards for general trends.
- **Active Feature Window**: June 2025 onwards for high-fidelity training (V2.7 optimization) to minimize RAM pressure on cloud nodes while preserving seasonal relevance.

---

## 2. Feature Engineering & Preprocessing

The model operates on **22 distinct features** engineered to capture temporal demand elasticity and SKU lifecycle dynamics.

### 2.1 recency_weight (Lifecycle-Aware Decay)
A unique weighting system ensures the model learns "fast" for new trends while respecting "history" for slow-movers.
$$Weight = \exp(-\text{decay\_rate} \times \text{days\_diff})$$
- **NEW launches**: decay = 0.010 (fast learning, 100-day memory).
- **MUDA (slow-movers)**: decay = 0.002 (deep context, 500-day memory).

### 2.2 Elasticity Features
The model doesn't just look at price; it looks at "Relative Price" to capture market psychology:
- **Price Ratio**: `Price / Mean_Price` (per SKU-channel).
- **Discount Depth**: `(MRP - Price) / MRP`.
- **Rolling Demand**: 7-day, 14-day, and 30-day moving averages of velocity.

---

## 3. XGBoost Model Architecture

The core forecasting engine uses **XGBoost (Extreme Gradient Boosting)** for its ability to model non-linear demand curves and complex feature interactions.

### 3.1 Model Configuration
- **Objective**: `reg:squarederror` (minimizing Mean Squared Error).
- **Tree Parameters**: `max_depth: 6`, `learning_rate: 0.08`, `subsample: 0.8`.
- **Turbo-Mode Optimization**: For real-time production, `n_estimators` is pruned to 200, allowing for sub-60s retraining while maintaining a Validation MAE of **1.44**.

### 3.2 Time-Series Cross-Validation
Validation is conducted via `TimeSeriesSplit(n_splits=3)`. This prevents "look-ahead bias" by ensuring the model is always tested on data that occurs *after* the training set in chronological order.

---

## 4. Simulation & Optimization Logic

Once the model identifies the demand curve, the engine performs a granular simulation across the price-grid for every SKU-channel.

### 4.1 Grid Simulation Strategy
1. **Search Space**: For every active SKU, 20 price points are simulated between the SKU's minimum and maximum guardrails.
2. **Prediction Pipeline**: The trained XGBoost model predicts the **Daily Run Rate (DRR)** for all 20 points simultaneously.
3. **Inventory Constraints**: Logic computes **Days on Hand (DOH)** for every simulated point. If $Inventory \le 0$, DOH and Urgency are forced to null to avoid misleading clearance signals.

### 4.2 Bayesian Elasticity Transfer (Cold-Start Solution)
For SKUs with sparse data ($N < 3$ price points), we implement a Bayesian anchor:
$$E_{final} = \frac{n \times E_{sku} + C \times E_{category}}{n + C}$$
Where $C=5$ (regularization constant). This allows "thin" SKUs to borrow the elasticity profile of their lifecycle category (e.g., SUPER40 norms).

---

## 5. Multi-Objective Scoring Framework

Generating a recommendation is not just about maximizing profit; it’s about balancing yield against inventory health.

### 5.1 Urgency Sigmoid
Urgency is modeled using a Sigmoid function relative to tag-specific DOH targets:
$$Urgency = \frac{1}{1 + \exp(-(Current\_DOH - Target\_DOH) / 15)}$$

### 5.2 Composite Score Calculation
Every simulated price point is ranked using a dynamic weighting system that adapts to the Urgency score:

| Component | Calculation |
|:---|:---|
| **Profit Score** | $ML\_Monthly\_Profit / Max\_Profit$ |
| **DRR Score** | $ML\_DRR / Max\_DRR$ |
| **DOH Score** | $1 / \text{fmax}(ML\_DOH, 0.01)$ |
| **Proximity Score** | $1 - (|Price - Last\_Price| / Last\_Price)$ |

**Final Score** = $\sum (Weight_i \times Score_i)$
*Overstocked SKUs (Urgency > 0.7) shift focus to **DRR Score** (weight: 0.45), while low-stock SKUs shift to **Profit Score** (weight: 0.55).*

---

## 6. Production Orchestration

### 6.1 Computational Pipeline
The engine executes in a 7-step decoupled pipeline:
1. `extract_training_data()`: Raw SQL ingestion.
2. `engineer_features()`: Vectorized Pandas preprocessing.
3. `train_model()`: XGBoost training on the feature set.
4. `build_simulation_grid()`: Permutation of SKUs $\times$ 20 price points.
5. `simulate_with_model()`: Batch inference via the trained model.
6. `score_and_rank()`: Dynamic weight application and ranking.
7. `format_recommendations()`: Selection of MAX_PROFIT, MAX_DRR, and OPTIMAL rows.

### 6.2 Caching & State Management
- **State Persistence**: URL query parameters are synced with `st.session_state` to enable deep-linking and state recovery during browser refreshes.
- **Global Cache**: `@st.cache_data(ttl=3600)` stores the finalized results in a shared pointer, preventing redundant compute across different stakeholder sessions.

---
**Technical Lead**: AI Data Science Division  
**Code Reference**: `pricing_ml_engine.py`, `dashboard.py`  
**Stack**: Python 3.13, XGBoost 3.2, PostgreSQL 15, Streamlit 1.35  
