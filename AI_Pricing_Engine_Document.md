# Technical Architecture: AI-Powered Dynamic Pricing Engine
## Comprehensive Specification: Mathematical Systems, ML Architecture, and Operational Logic
**Document Version: 5.0 | February 2026**

---

## 1. Data Foundation & In-Database Orchestration

The system utilizes a high-performance PostgreSQL (AWS RDS) data warehouse. The foundation of the logic is contained within the `enhanced_pricing_engine.sql` and `pricing_ml_engine.py` layers, performing large-scale statistical aggregation.

### 1.1 Data Cleansing & Integrity (CTE: mapping_dedup)
The `itemmaster` table contains duplicate records for single SKUs. To ensure revenue accuracy, the engine uses a Windows Function to isolate the most recent metadata:
- **Partitioning**: Grouped by `Product Code`.
- **Ranking**: Ordered by `updatedAt` in descending order.
- **Attributes Retained**: Cost Price, MRP, Lifecycle Tags, and Category Name.

### 1.2 Channel Normalization (CTE: base_sales)
The engine applies specific revenue normalization to align disparate channel data:
- **AJIO Logic**: `Selling Price + 65` (Normalizes for platform-specific fixed revenue offsets).
- **Price Rounding**: All prices are rounded to the nearest Integer to eliminate floating-point noise in elastic modeling.
- **Time Window**: Analysis strictly follows a window starting **June 1, 2025** to ensures model focus on current market elasticity.

---

## 2. Mathematical Logic Engines

The engine uses four specific mathematical frameworks to handle specific business challenges.

### 2.1 Lifecycle-Aware Recency Weighting
Observed sales are weighted using an **Exponential Decay Function** to prioritize recent trends:
- **Calculation**: `Weight = exp(- Decay_Rate * Days_Since_Sale)`
- **Parameters**: 
  - **MUDA (Slow-Movers)**: Rate = 0.002 (Deep logic context across ~500 days).
  - **NEW (Launches)**: Rate = 0.010 (Aggressive adaptation, focuses on last 100 days).
  - **CORE (Staples)**: Rate = 0.004 (Balanced stability).
  - **REGULAR**: Rate = 0.005 (Standard decay).

### 2.2 Cold-Start Problem: Bayesian Elasticity Transfer
When a SKU has sparse data ($N < 3$ price points), the engine implements a **Bayesian Prior** from the category group:
- **Elasticity Slope Calculation**: `final_e = ( (n_sku * e_sku) + (5 * e_category) ) / (n_sku + 5)`
- **Constraint**: Slopes are clamped to a minimum of **-2.5** to prevent irrational demand forecasts.
- **R-Squared Filter**: SKU-level elasticity is only used if `R2 > 0.2`, otherwise it defaults to the category transfer logic.

### 2.3 Sector-Specific Net Realization (Flipkart Uplift)
Profit is calculated based on **Net Realization** (True Bottom Line) rather than Gross Revenue.
- **Calculation**: `Net_Realization = (Price * (1 - 0.18)) - Uplift_Fee`
- **Uplift Parameters (Flipkart)**:
  - **Uplift** is a lookup based on `Category` and `Price Slab`.
  - **Example (BOXER)**: ₹0-150: ₹58 | ₹151-300: ₹90 | ₹301-500: ₹92 | ₹501-1000: ₹135 | ₹1001+: ₹138.
  - **Example (T SHIRT)**: ₹0-150: ₹58 | ₹151-300: ₹78 | ₹301-500: ₹80 | ₹501-1000: ₹157 | ₹1001+: ₹192.
  - **Fallback**: 18% Commission + ₹35 (Myntra) or Category-average default (Flipkart).

### 2.4 Inventory Urgency (The Sigmoid System)
The engine calculates a probability of overstocking using a **Sigmoid function**:
- **Calculation**: `Urgency Score = 1 / (1 + exp( - (Current_DOH - Target_DOH) / 15 ))`
- **Targets**: MUDA: 30 days | NEW: 45 days | SUPER40: 50 days | CORE: 90 days.
- **Mechanism**: A score > 0.7 triggers a strategic weight shift towards velocity.

---

## 3. XGBoost Machine Learning Model

The Python layer implements a non-linear Gradient Boosted Tree model for demand forecasting.

### 3.1 Advanced Feature Engineering (22 Vectors)
Model inputs include:
- **Core**: Price, Price/MRP Ratio, Price/Avg Ratio, Margin %, Price Bucket.
- **Temporal**: Day-of-Week (DOW), Month, Day-of-Month, Week-of-Year.
- **Seasonality**: Weekend Flag, Month-Start Flag (Payday), Month-End Flag.
- **Momentum**: 7-day, 14-day, and 30-day Rolling Units Sold (Velocity).
- **Environment**: Event Flag (BAU vs Sale), Channel ID, Lifecycle Tag, Category ID.
- **Stability**: Price Volatility (30-day standard deviation).

### 3.2 Hyperparameters & Validation
- **Algorithm**: XGBoost Regressor (`tree_method='hist'`).
- **Learning Parameters**: `max_depth: 6`, `learning_rate: 0.08`, `subsample: 0.8`, `colsample_bytree: 0.8`.
- **Estimators**: 600 (Production) / 200 (Turbo-Mode Dashboard).
- **Validation**: **TimeSeriesSplit** (3-Fold) prevents temporal leakage.

---

## 4. Multi-Objective Optimization

The engine ranks 20 simulated price points per SKU using a **Dynamic Weighted Composite Score**.

### 4.1 Weight Distribution Matrix
Weights shift based on the Urgency Score ($\mu$):

| Mode | Trigger | Profit Weight | Velocity (DRR) Weight | DOH Weight | Proximity Weight |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Margin Mode** | $\mu < 0.3$ | **0.55** | 0.15 | 0.20 | 0.10 |
| **Balanced Mode** | $\mu = 0.3-0.7$ | **0.45** | 0.30 | 0.15 | 0.10 |
| **Velocity Mode** | $\mu > 0.7$ | 0.20 | **0.45** | 0.25 | 0.10 |

### 4.2 Score Normalization
To ensure disparate units (Rupees vs. Units) can be summed, all components are normalized:
- **Score Component**: `(Value / Max_Value_in_Pool) * Weight`

---

## 5. Deployment Orchestration

### 5.1 Pipeline Execution Flow
The AI Engine runs in a decoupled 7-step sequence:
1. `extract_data()`: Pulls active inventory, last prices, and 2-year sales history.
2. `engineer_features()`: Vectorizes the sales history into the 22-signal matrix.
3. `train_model()`: Executes XGBoost demand learning.
4. `build_grid()`: Expands target SKUs into 20 scenario-points each.
5. `simulate()`: AI batch-inference to predict demand for every scenario.
6. `rank()`: Calculates the Composite Score using the weight matrix.
7. `serve()`: Pushes results to the Streamlit Dashboard and CSV exports.

### 5.2 Performance & Security
- **Global Cache**: `@st.cache_data(ttl=3600)` ensures 1-hour global persistence.
- **State Persistence**: Syncs `st.session_state` with URL Query Params for deep-linking.
- **Infrastructure**: Python 3.13 | PostgreSQL 15 | Linux (Debian 12).

---
**Prepared by**: AI Data Science Division | Pepe Pricing Operations
**Technical Lead**: Antigravity AI Engine
**Version**: 5.0 (Stable)
