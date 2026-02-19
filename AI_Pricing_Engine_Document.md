# AI-Driven Dynamic Pricing Strategy & Operational Engine
## Technical Specification & Executive Report — v2.5
**February 2026**

---

## 1. Executive Summary

We have successfully deployed a sophisticated **AI-Powered Pricing Engine** that transitions our pricing operations from static, rule-based logic to a dynamic, machine-learning-driven ecosystem. 

By analyzing **2.27 million transactions** across Myntra, Flipkart, and AJIO, the engine identifies the hidden relationships between price, demand (velocity), and inventory lifecycle. The primary objective is to maximize high-margin revenue while maintaining healthy sell-through rates.

### 1.1 Performance Overview

| Metric | Business Impact |
|:---|:---|
| **Optimized Profit Uplift** | **+34.9%** (across all channels) |
| **Analyzed SKU-Channels** | 4,932 unique combinations |
| **AI Recommendations** | 10,198 high-confidence picks |
| **Model Retraining** | Automated every 60 minutes |

### 1.2 Projected Monthly Profit Transformation

| Channel | Baseline Profit | AI-Optimized Target | Estimated Uplift |
|:---|:---:|:---:|:---:|
| **Flipkart** | ₹1.24 Cr | ₹1.88 Cr | **+51.9%** |
| **Myntra** | ₹1.18 Cr | ₹1.45 Cr | **+22.6%** |
| **AJIO** | ₹54.3 L | ₹66.6 L | **+22.6%** |
| **Cumulative** | **₹2.96 Cr** | **₹3.99 Cr** | **+34.9%** |

---

## 2. Technical Architecture & Innovation

The V2.5 engine utilizes a hybrid approach, combining high-speed SQL aggregation with a "Turbo-Mode" XGBoost Machine Learning pipeline.

### 2.1 The "Turbo-Mode" ML Pipeline (V2.5)
To facilitate real-time exploration on the stakeholder dashboard, we implemented a specialized high-speed training mode:
- **Compressed Data Window**: Focuses on the most relevant market signals from **June 2025 – Present**.
- **Pruned Model Complexity**: Optimized XGBoost architecture using 200 estimators for <60s retraining without sacrificing directionality.
- **Inventory Filtering**: Dramatically improved simulation speed by focusing only on SKUs with active stock (10k-15k simulations).

### 2.2 Core AI Enhancements
1. **Event-Specific Elasticity**: Separate demand curves for BAU vs. Sale Event days (Myntra EORS, Flipkart BBD, etc.).
2. **Lifecycle-Aware Decay**: Faster learning rates for **NEW** launches; deeper historical memory for **MUDA** slow-movers.
3. **Bayesian Elasticity Transfer**: "Cold-start" problem solved by allowing new SKUs to borrow intelligence from peer categories.
4. **Demand Heartbeats**: Real-time modeling of Sunday-Monday spikes and monthly salary-cycle purchasing peaks.

---

## 3. Interactive Management Dashboard

The stakeholder interface has been upgraded to a premium production-grade platform.

### 3.1 Advanced Capabilities (V2.5)
- **Zero-Latency Interaction**: Powered by a 1-hour global server-side cache. Once initialized, all SKU searches are **instant**.
- **Persistent Deep-Linking**: URL query parameters (`?sku=...&channel=...`) are synchronized with the UI. Stakeholders can share exact SKU views via a simple link.
- **Pulse Status Display**: A "7-Step Pulse" loader provides a transparent view of the AI engine's health and initialization steps.
- **High-Fidelity Aesthetics**: Custom CSS with fade-in animations, distinctive input boxes, and interactive hover effects for a premium user experience.

### 3.2 Key Views
- **Profit v. Velocity Curve**: Interactive visualization of the predicted demand curve with clear markers for current and optimal price points.
- **AI-Generated Insights**: Context-aware prompts like *"Decrease price to clear overstock"* or *"Increase price to capture untapped margin."*
- **KPI Performance Cards**: Real-time visibility into current inventory, target DOH, and monthly profit potential.

---

## 4. Operational Guardrails & Security

The engine operates within strict business and technical constraints.

### 4.1 Pricing Bounds
Every AI recommendation is automatically clamped between internal BAU and Event guardrails to ensure zero violation of platform agreements.

### 4.2 Data Security
- **Credential Masking**: All database secrets are managed via **Streamlit Cloud Secrets**, ensuring zero exposure in the Git repository.
- **Source Control**: Full versioning on GitHub with restricted `.gitignore` preventing sensitive data leaks.

---

## 5. Technology Stack

| Component | Technology |
|:---|:---|
| **Database** | PostgreSQL (AWS RDS Production) |
| **Analysis** | Python 3.13 + Pandas + NumPy |
| **AI/ML** | XGBoost 3.2 (Histogram-based learning) |
| **Dashboard** | Streamlit v1.35+ (Custom CSS / Responsive) |
| **Deployment** | Streamlit Community Cloud (Global Access) |

---

## 6. Strategic Roadmap (Q1-Q2 2026)

1. **Phase 3 (Next)**: Integration of **RTO (Return-to-Origin) Penalties** into the profit formula to penalize high-return price points.
2. **Phase 3**: Automated daily push to channel portals (API-driven price updates).
3. **Phase 4**: Expansion to Quick-Commerce channels (**Blinkit, Swiggy Instamart, Zepto**).

---

## 7. How to Access the Live Engine

The engine is deployed as a live web application for immediate use.

**Dashboard URL**: `https://pepe-pricing.streamlit.app`
*(Requires internal database access credentials to be configured in secrets)*

**Local Execution**:
```bash
cd "/Users/mac/Documents/pricing suggestion"
source venv/bin/activate
streamlit run dashboard.py
```

---
*Document Version: 2.5*  
*Last Updated: February 19, 2026*  
*Contact: AI Data Science Team | Pepe Pricing Operations*
