# Technical Architecture: AI-Powered Dynamic Pricing Engine
## Comprehensive Specification: Mathematical Systems, ML Architecture, and Operational Logic
**Document Version: 4.0 | February 2026**

---

## 1. Data Foundation & In-Database Orchestration

The system utilizes a high-performance PostgreSQL (AWS RDS) data warehouse. The foundation of the logic is contained within the `enhanced_pricing_engine.sql` layer, which performs large-scale data cleansing and statistical aggregation.

### 1.1 Data Cleansing & Integrity (CTE: mapping_dedup)
The `itemmaster` table contains duplicate records for single SKUs. To ensure revenue accuracy, the engine uses a Windows Function to isolate the most recent metadata:
- **Partitioning**: Grouped by `Product Code`.
- **Ranking**: Ordered by `updatedAt` in descending order.
- **Filtering**: Only row rank #1 is used for Cost Price, MRP, and Lifecycle Tags.

### 1.2 Channel Normalization (CTE: base_sales)
Pricing behavior differs across channels. The engine applies a **Reverse Commission Offset** for AJIO to normalize its "Selling Price" against Myntra/Flipkart benchmarks:
- **AJIO Logic**: `Selling Price + 65` (Normalizes for platform-specific fixed costs).

### 1.3 Marketplace Slab Integration (V4.1)
The profit calculation now uses **Category-Specific Net Realization**. It explicitly models the marketplace commission (18%) and "Fixed Fee + Shipping" (Uplift) based on the product category and price range:
- **Flipkart Category Slabs**: The model utilizes a granular lookup table (Uplift CTE) for categories like 'T SHIRT', 'BOXER', 'BRIEF', etc., across 5 price buckets (0-150, 151-300, 301-500, 501-1000, 1001+).
- **Optimization Strategy**: By modeling these fee "cliffs," the engine identifies cases where a price increase actually reduces net profit (e.g., if a price shift from ₹499 to ₹505 triggers a ₹40 jump in fees).
- **Default Fallbacks**: SKUs without a matched category default to a conservative 18% commission + ₹65/₹95/₹145 fixed fee structure to protect operational margins.

---

## 2. Mathematical Logic Engines

The engine uses three specific mathematical frameworks to handle different business challenges.

### 2.1 Lifecycle-Aware Recency Weighting
Observed sales are not equal. A sale from yesterday is a stronger signal than a sale from 6 months ago. We apply an **Exponential Decay Function**:
- **Formula**: `Weight = exp(- Decay_Rate * Days_Since_Sale)`
- **MUDA (Slow-Movers)**: Rate = 0.002 (Deep memory, retains signals for ~500 days).
- **NEW (Launches)**: Rate = 0.010 (Aggressive learning, focuses on the last 100 days).
- **CORE (Staples)**: Rate = 0.004 (Balanced stability).

### 2.2 Cold-Start Problem: Bayesian Elasticity Transfer
Traditional simulations fail when a SKU has limited price changes. We solve this using **Category-Level Pooling**:
- **Mechanism**: If a SKU has < 3 price points, its elasticity is calculated as a blend of its own history and the average elasticity of its "Peer Group" (the Lifecycle Tag).
- **Blending Formula**: `Final_Elasticity = ( (SKU_Points * SKU_Slope) + (5 * Category_Slope) ) / (SKU_Points + 5)`
- **Anchor**: The constant '5' acts as a regularization anchor, preventing the model from overreacting to small SKU-level data samples.

### 2.3 Inventory Urgency (The Sigmoid System)
Strategy shifts based on stock health. We calculate an **Urgency Score (0 to 1)** using a Sigmoid function relative to a Target Days-on-Hand (DOH):
- **Formula**: `Urgency = 1 / (1 + exp( - (Current_DOH - Target_DOH) / 15 ))`
- **Result**: Once Current DOH exceeds the Target by 30+ days, the Urgency score crosses 0.8, triggering "Velocity Mode" in the recommendation engine.

---

## 3. XGBoost Machine Learning Model

The Python layer (`pricing_ml_engine.py`) implements a Gradient Boosted Tree model that captures non-linear relationships that traditional SQL cannot.

### 3.1 Advanced Feature Engineering (22 Vectors)
Every simulation point is evaluated against 22 distinct signals:
- **Price Psychology**: Absolute Price, Price-to-MRP ratio, Discount Depth, and Price-Step buckets.
- **Market Velocity**: 7-day, 14-day, and 30-day rolling average demand.
- **Seasonal Waves**: Day-of-week, Month, weekends, and Payday flags (Month-Start/End spikes).
- **SKU Context**: Event flag (BAU vs Sale), Lifecycle Tag (encoded), and Channel ID.

### 3.2 Model Specifications
- **Architecture**: XGBoost Regressor using the `hist` tree method for high-speed processing on large grids.
- **Hyperparameters**: 
    - `max_depth: 6` (Captures deep interactions without overfitting).
    - `learning_rate: 0.08` (Ensures stable convergence).
    - `n_estimators: 300` (Optimized for <40s training).
- **Validation**: **TimeSeriesSplit** (3-Fold Cross-Validation) ensures the model is tested by following chronological boundaries, preventing data leakage from "future" sales into "past" training.

---

## 4. Simulation & Recommendation Logic

The engine does not just "predict" a single price; it simulates a comprehensive "Price Ladder" for every active SKU-Channel.

### 4.1 Grid Simulation Strategy
1. **Search Space**: The engine builds a 20-point price grid per SKU between the permitted BAU and Event guardrails.
2. **Psychological Anchors**: Specific points like ₹X49, ₹X99, and ₹X95 are added to the grid to test consumer conversion at "Charm Prices."
3. **Inference**: The XGBoost model performs batch inference on the entire 20-point grid, returning the predicted Daily Run Rate (DRR) for every scenario.

### 4.2 Dynamic Strategic Scoring
Final recommendations are chosen based on a **Composite Score** that adapts to inventory states:

| Mode | Urgency | Objective | Primary Weight |
|:---|:---:|:---|:---:|
| **Margin Mode** | < 0.3 | Protect Profitability | Profit: 0.55 |
| **Balanced Mode**| 0.3-0.7 | Growth & Profit | Profit: 0.45, Velocity: 0.30 |
| **Velocity Mode**| > 0.7 | Clearance / Liquidation | Velocity (DRR): 0.45 |

---

## 5. Dashboard Performance & Flow

### 5.1 Orchestration Pipeline
The dashboard initializes via a 7-step decoupled pipeline:
1. **Extraction**: Live SQL ingestion of sales/inventory.
2. **Engineering**: Vectorization of ML features.
3. **Training**: Real-time XGBoost demand learning.
4. **Grid Scaling**: Expansion of active SKUs into 20-point grids.
5. **Inference**: AI demand prediction.
6. **Ranking**: Sorting for MAX_PROFIT and OPTIMAL picks.
7. **Serving**: Rendering via the interactive Streamlit interface.

### 5.2 Efficiency & Persistence
- **Global Caching**: Results are stored in an encrypted 1-hour RAM cache (`st.cache_data`), allowing for instant search performance for all users.
- **Deep Linking**: The dashboard state (selected SKU/Channel) is synchronized with URL query parameters, enabling direct sharing of SKU-specific insights between team members.

---
**Tech Stack**: Python 3.13 | XGBoost | PostgreSQL (AWS RDS) | Streamlit v1.35
**Authors**: Pepe AI Data Science & Pricing Ops
**Version**: 4.0 Stable
