# Technical Specification: AI-Powered Dynamic Pricing Engine
## Mathematical Foundations, SQL Orchestration & XGBoost ML Logic
**Document Version: 3.1**

---

## 1. Executive Summary

The Pepe Dynamic Pricing Engine is a multi-layered analytical system that predicts product demand and optimizes price points across Myntra, Flipkart, and AJIO. It replaces static pricing with a probabilistic model that balances profit maximization against inventory liquidation urgency.

---

## 2. SQL Engine: In-Database Feature Engineering

The SQL layer (`enhanced_pricing_engine.sql`) serves as the core data processing and statistical foundation. It performs complex transformations and "cold-start" calculations directly within the data warehouse.

### 2.1 Lifecycle-Aware Demand Decay
The engine uses a time-decaying weight for historical sales observations to ensure'recency' is respected. The decay constant ($\lambda$) is dynamically assigned based on the SKU's lifecycle tag:
$$Weight = \exp(-\lambda \times \Delta t)$$
- **NEW**: $\lambda = 0.010$ (100-day window) - Adapts rapidly to launch trends.
- **MUDA**: $\lambda = 0.002$ (500-day window) - Accumulates deeper history for slow-movers.
- **CORE**: $\lambda = 0.004$ (250-day window) - Stable, balanced training context.

### 2.2 Split-Elasticity Modeling
Unlike standard simulations, the SQL engine calculates separate Log-Linear regressions for **BAU** and **Event** periods per SKU-channel.
- **Formula**: $LN(DRR) = \beta_0 + \beta_1 LN(Price)$
- **Constraint**: Slopes are clamped to a maximum of -2.5 (aggressive elasticity) to prevent outlier predictions.

### 2.3 Bayesian Cross-SKU Transfer
To handle SKUs with sparse data (cold-starts), the engine implements a Bayesian anchor where SKU-level signal is blended with the category-level norm:
$$Elasticity_{Final} = \frac{(n \times Elasticity_{SKU}) + (C \times Elasticity_{Category})}{n + C}$$
Where $n$ is the SKU's observation count and $C=5$ is the regularization anchor. This ensures every product has a data-backed demand curve from day one.

---

## 3. Python ML Engine: XGBoost Forecasting

The Python layer (`pricing_ml_engine.py`) implements a non-linear gradient boosted tree model to capture complex feature interactions (e.g., how "Weekend" interacts with "Discount Depth").

### 3.1 Advanced Feature Engineering (22 Features)
The ML model consumes high-dimensional vectors for every SKU-channel-day:
- **Price Psychology**: Price-to-MRP ratio, price-to-average ratio (per SKU), and ₹50 price buckets.
- **Temporal Waves**: Day-of-week, day-of-month, month, and "Payday Effect" flags (month-start/end).
- **Demand Momentum**: 7-day, 14-day, and 30-day rolling average velocities.
- **Contextual Signals**: Event flags, lifecycle tags (encoded), and channel labels.

### 3.2 Model Architecture & Training
- **Model**: XGBoost Regressor (`hist` tree method for high-speed computation).
- **Optimization**: Squared Error loss with `learning_rate=0.08` and `max_depth=6`.
- **Validation**: Uses **TimeSeriesSplit** (3 folds) to ensure the model is validated on future data relative to its training set, preventing temporal leakage.

### 3.3 Simulation Grid & Predictive Inference
The model doesn't just predict once; it simulates a "Price Ladder":
1. Generate 20 price points (±15% of last price) including psychological anchors (e.g., ₹X99).
2. Parallel inference for all 20 points for both BAU and Event scenarios.
3. Conversion of ML outputs into Daily Run Rate (DRR) and monthly profit projections.

---

## 4. Optimization & Scoring Framework

Both engines feed into a dynamic multi-objective scoring system that selects the "Optimal" price based on business phase.

### 4.1 Inventory Urgency Sigmoid
Urgency is not linear. We use a Sigmoid function relative to tag-specific Days-on-Hand (DOH) targets:
$$Urgency = \frac{1}{1 + \exp(-(Current\_DOH - Target\_DOH) / 15)}$$

### 4.2 Dynamic Strategic Weights
The recommendation logic shifts its priorities based on the Urgency score:

| Strategy | Trigger | Profit Weight | Velocity (DRR) Weight |
|:---|:---:|:---:|:---:|
| **Margin Mode** | Urgency < 0.3 | **0.55** | 0.15 |
| **Balanced Mode** | Urgency 0.3-0.7 | 0.45 | 0.30 |
| **Velocity Mode** | Urgency > 0.7 | 0.20 | **0.45** |

---

## 5. Implementation Roadmap

- **Production Sync**: The pipeline runs in 7 modular steps, from SQL extraction to final CSV/Dashboard export.
- **Data Security**: Credential isolation via Streamlit Secrets and `.gitignore` policies.
- **Future State**: Phase 3 will integrate **RTO (Return) Penalties** directly into the Profit Score, penalizing price points that drive high return rates.

---
**Prepared by**: AI Data Science Lead  
**Last Updated**: February 19, 2026  
**Files**: `pricing_ml_engine.py` | `enhanced_pricing_engine.sql`
