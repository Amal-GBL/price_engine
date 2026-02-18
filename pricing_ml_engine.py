#!/usr/bin/env python3
"""
PEPE AI Pricing Engine v2.0 — XGBoost ML Layer
================================================
Trains an XGBoost model on historical sales data to predict demand (DRR)
at any price point, then simulates optimal pricing across a price ladder.

Outputs:
  - output/ml_predictions.csv          : All simulated price points with ML-predicted DRR
  - output/ml_recommendations.csv      : Top recommendations (max profit / max DRR / optimal)
  - output/model_diagnostics.csv       : Model feature importance & performance metrics
  - output/sql_enhanced_results.csv    : Results from the enhanced SQL engine (run first)

Usage:
  pip install -r requirements.txt
  python pricing_ml_engine.py
"""

import os
import sys
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("PricingEngine")

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
def get_db_config():
    try:
        import streamlit as st
        return {
            "host": st.secrets["db"]["host"],
            "port": st.secrets["db"]["port"],
            "dbname": st.secrets["db"]["dbname"],
            "user": st.secrets["db"]["user"],
            "password": st.secrets["db"]["password"],
        }
    except Exception:
        # Local fallback if not running through streamlit
        return {
            "host": "gbl-crawler-production-1.cjsbw32v9qjg.ap-south-1.rds.amazonaws.com",
            "port": 5432,
            "dbname": "gbl_data_lake",
            "user": "readuser_access",
            "password": "readuser@123",
        }

DB_CONFIG = get_db_config()

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Channels to model (following original query channel-name ILIKE rules)
CHANNELS = ["MYNTRA", "FLIPKART", "AJIO"]

# Tag-specific configs
TAG_CONFIG = {
    "MUDA":    {"decay": 0.002, "target_doh": 30},
    "NEW":     {"decay": 0.010, "target_doh": 45},
    "SUPER40": {"decay": 0.006, "target_doh": 50},
    "CORE":    {"decay": 0.004, "target_doh": 90},
    "REGULAR": {"decay": 0.005, "target_doh": 60},
}
DEFAULT_TAG_CONFIG = {"decay": 0.005, "target_doh": 60}


# ─────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────
def get_connection():
    """Get a psycopg2 connection to the database."""
    return psycopg2.connect(**DB_CONFIG)


def run_query(query: str, params=None) -> pd.DataFrame:
    """Execute a query and return a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


# ─────────────────────────────────────────────────────
# STEP 1: EXTRACT TRAINING DATA
# ─────────────────────────────────────────────────────
def extract_training_data() -> pd.DataFrame:
    """Pull historical sales data enriched with features for ML."""
    log.info("📥 Extracting training data from database...")

    query = """
    WITH
    mapping_dedup AS (
        SELECT *
        FROM (
            SELECT
                "Product Code",
                "Cost Price",
                "Tags",
                "MRP",
                "Size",
                "Color",
                ROW_NUMBER() OVER (
                    PARTITION BY "Product Code"
                    ORDER BY "updatedAt" DESC NULLS LAST
                ) AS rn
            FROM "DataWarehouse".pepe_in_unicommerce_itemmaster
        ) x
        WHERE rn = 1
    ),

    event_calendar_long AS (
        SELECT "Date"::DATE AS cal_date, 'MYNTRA'   AS channel, "Myntra"   AS event_type
        FROM "DataWarehouse".pepe_event_calendar WHERE "Myntra" IS NOT NULL
        UNION ALL
        SELECT "Date"::DATE, 'FLIPKART', "Flipkart"
        FROM "DataWarehouse".pepe_event_calendar WHERE "Flipkart" IS NOT NULL
        UNION ALL
        SELECT "Date"::DATE, 'AJIO', "AJIO"
        FROM "DataWarehouse".pepe_event_calendar WHERE "AJIO" IS NOT NULL
    ),

    daily_sales AS (
        SELECT
            s."Item SKU Code" AS sku,
            DATE(s."Order Date as dd/mm/yyyy hh:MM:ss") AS order_date,

            CASE
                WHEN s."Channel Name" ILIKE '%%MYNTRA%%'   THEN 'MYNTRA'
                WHEN s."Channel Name" ILIKE '%%FLIPKART%%' THEN 'FLIPKART'
                WHEN s."Channel Name" ILIKE '%%AJIO%%'     THEN 'AJIO'
            END AS channel,

            ROUND(
                CASE
                    WHEN s."Channel Name" ILIKE '%%AJIO%%'
                    THEN s."Selling Price" + 65
                    ELSE s."Selling Price"
                END
            )::INT AS price,

            md."Tags"        AS tags,
            md."Cost Price"  AS cost_price,
            md."MRP"         AS mrp,
            md."Size"        AS size,
            md."Color"       AS color,

            COUNT(*) AS units_sold

        FROM "DataWarehouse".pepe_in_unicommerce_saleorders s
        LEFT JOIN mapping_dedup md
            ON s."Item SKU Code" = md."Product Code"
        WHERE s."Order Date as dd/mm/yyyy hh:MM:ss" >= '2025-01-01'
          AND s."Sale Order Item Status" NOT IN ('CANCELLED','UNFULFILLABLE')
          AND s."Selling Price" > 0
          AND s."Channel Name" NOT ILIKE '%%SOR%%'
          AND s."Bundle SKU Code Number" = ''
          AND (
                s."Channel Name" ILIKE '%%MYNTRA%%'
             OR s."Channel Name" ILIKE '%%FLIPKART%%'
             OR s."Channel Name" ILIKE '%%AJIO%%'
          )
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9
    )

    SELECT
        ds.*,
        COALESCE(ec.event_type, 'BAU') AS event_flag,
        EXTRACT(DOW FROM ds.order_date)::INT AS dow,
        EXTRACT(MONTH FROM ds.order_date)::INT AS month,
        EXTRACT(DAY FROM ds.order_date)::INT AS day_of_month
    FROM daily_sales ds
    LEFT JOIN event_calendar_long ec
        ON ds.order_date = ec.cal_date
       AND ds.channel = ec.channel
    ORDER BY ds.sku, ds.channel, ds.order_date;
    """

    df = run_query(query)
    log.info(f"   → Extracted {len(df):,} daily-SKU-channel-price rows")
    return df


# ─────────────────────────────────────────────────────
# STEP 2: EXTRACT INVENTORY & GUARDRAILS
# ─────────────────────────────────────────────────────
def extract_inventory() -> pd.DataFrame:
    """Get current inventory snapshot."""
    log.info("📦 Extracting inventory data...")
    query = """
    SELECT
        "Item SkuCode" AS sku,
        SUM("Inventory"::INT) AS inventory
    FROM "DataWarehouse".pepe_in_unicommerce_inventorysnapshot
    WHERE "Inventory Snapshot Date" = (
        SELECT MAX("Inventory Snapshot Date")
        FROM "DataWarehouse".pepe_in_unicommerce_inventorysnapshot
    )
    GROUP BY 1;
    """
    return run_query(query)


def extract_guardrails() -> pd.DataFrame:
    """Get pricing guardrails (BAU/Event bounds)."""
    log.info("🛡️  Extracting pricing guardrails...")
    query = """
    SELECT
        "Pcode"   AS sku,
        "Channel" AS channel,
        NULLIF(REPLACE("BAU", ',', ''), '')::NUMERIC   AS bau_price,
        NULLIF(REPLACE("Event", ',', ''), '')::NUMERIC AS event_price
    FROM "DataWarehouse".pepe_pricing_guardrails;
    """
    return run_query(query)


def extract_last_prices() -> pd.DataFrame:
    """Get last transaction price per SKU-channel."""
    log.info("💰 Extracting last transaction prices...")
    query = """
    WITH base_sales AS (
        SELECT
            s."Item SKU Code" AS sku,
            DATE(s."Order Date as dd/mm/yyyy hh:MM:ss") AS order_date,
            CASE
                WHEN s."Channel Name" ILIKE '%%MYNTRA%%'   THEN 'MYNTRA'
                WHEN s."Channel Name" ILIKE '%%FLIPKART%%' THEN 'FLIPKART'
                WHEN s."Channel Name" ILIKE '%%AJIO%%'     THEN 'AJIO'
            END AS channel,
            ROUND(
                CASE
                    WHEN s."Channel Name" ILIKE '%%AJIO%%'
                    THEN s."Selling Price" + 65
                    ELSE s."Selling Price"
                END
            )::INT AS price
        FROM "DataWarehouse".pepe_in_unicommerce_saleorders s
        WHERE s."Order Date as dd/mm/yyyy hh:MM:ss" >= '2025-01-01'
          AND s."Sale Order Item Status" NOT IN ('CANCELLED','UNFULFILLABLE')
          AND s."Selling Price" > 0
          AND s."Channel Name" NOT ILIKE '%%SOR%%'
          AND s."Bundle SKU Code Number" = ''
          AND (
                s."Channel Name" ILIKE '%%MYNTRA%%'
             OR s."Channel Name" ILIKE '%%FLIPKART%%'
             OR s."Channel Name" ILIKE '%%AJIO%%'
          )
    )
    SELECT DISTINCT ON (sku, channel)
        sku, channel, price AS last_price, order_date
    FROM base_sales
    ORDER BY sku, channel, order_date DESC;
    """
    return run_query(query)


# ─────────────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from raw sales data."""
    log.info("🔧 Engineering features...")

    df = df.copy()

    # --- Price features ---
    # Average price per SKU-channel
    avg_price = df.groupby(["sku", "channel"])["price"].transform("mean")
    df["price_ratio_to_avg"] = df["price"] / avg_price.replace(0, np.nan)

    # Price relative to MRP
    df["discount_depth"] = np.where(
        df["mrp"] > 0,
        1.0 - (df["price"] / df["mrp"]),
        0.0
    )

    # Price relative to cost (margin %)
    df["margin_pct"] = np.where(
        df["price"] > 0,
        (df["price"] - df["cost_price"].fillna(0)) / df["price"],
        0.0
    )

    # Price bucket (₹50 bands)
    df["price_bucket"] = (df["price"] / 50).round() * 50

    # --- Temporal features ---
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["days_ago"] = (pd.Timestamp.now() - df["order_date"]).dt.days
    df["week_of_year"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["dow"].isin([0, 6]).astype(int)
    df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 25).astype(int)

    # --- Categorical encoding ---
    df["is_event"] = (df["event_flag"] == "Event").astype(int)

    # Tag encoding
    tag_map = {"MUDA": 0, "REGULAR": 1, "SUPER40": 2, "NEW": 3, "CORE": 4}
    df["tag_encoded"] = df["tags"].map(tag_map).fillna(1).astype(int)

    # Channel encoding
    channel_map = {"MYNTRA": 0, "FLIPKART": 1, "AJIO": 2}
    df["channel_encoded"] = df["channel"].map(channel_map).fillna(0).astype(int)

    # --- Recency weight (lifecycle-aware) ---
    def get_decay(tag):
        return TAG_CONFIG.get(tag, DEFAULT_TAG_CONFIG)["decay"]

    df["decay_rate"] = df["tags"].apply(get_decay)
    df["recency_weight"] = np.exp(-df["decay_rate"] * df["days_ago"])

    # --- Rolling features per SKU-channel ---
    df = df.sort_values(["sku", "channel", "order_date"])

    for window in [7, 14, 30]:
        col_name = f"rolling_units_{window}d"
        df[col_name] = (
            df.groupby(["sku", "channel"])["units_sold"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    # Price volatility (std of price in last 30 entries)
    df["price_volatility"] = (
        df.groupby(["sku", "channel"])["price"]
        .transform(lambda x: x.rolling(30, min_periods=3).std())
        .fillna(0)
    )

    log.info(f"   → Engineered {len(df.columns)} features")
    return df


# ─────────────────────────────────────────────────────
# STEP 4: TRAIN XGBOOST MODEL
# ─────────────────────────────────────────────────────
FEATURE_COLS = [
    "price", "price_ratio_to_avg", "discount_depth", "margin_pct",
    "price_bucket",
    "dow", "month", "day_of_month", "week_of_year",
    "is_weekend", "is_month_start", "is_month_end",
    "is_event", "tag_encoded", "channel_encoded",
    "days_ago", "recency_weight",
    "rolling_units_7d", "rolling_units_14d", "rolling_units_30d",
    "price_volatility",
]

TARGET_COL = "units_sold"


def train_model(df: pd.DataFrame) -> tuple:
    """Train XGBoost with time-series aware cross-validation."""
    log.info("🤖 Training XGBoost demand model...")

    # Filter to valid rows
    train_df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    train_df = train_df.sort_values("order_date")

    X = train_df[FEATURE_COLS].values
    y = train_df[TARGET_COL].values

    # Use sample weights = recency weight
    sample_weights = train_df["recency_weight"].values

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    cv_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = sample_weights[train_idx]

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        )

        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        cv_metrics.append({"fold": fold + 1, "MAE": mae, "RMSE": rmse, "R2": r2})
        log.info(f"   Fold {fold+1}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

    # Final model on all data
    log.info("   Training final model on full dataset...")
    final_model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    final_model.fit(X, y, sample_weight=sample_weights, verbose=False)

    # Feature importance
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    log.info("   Top 10 features:")
    for _, row in importance.head(10).iterrows():
        log.info(f"      {row['feature']:30s} {row['importance']:.4f}")

    # Save diagnostics
    metrics_df = pd.DataFrame(cv_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "model_cv_metrics.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "model_feature_importance.csv", index=False)

    return final_model, metrics_df, importance


# ─────────────────────────────────────────────────────
# STEP 5: SIMULATE PRICE LADDER WITH ML PREDICTIONS
# ─────────────────────────────────────────────────────
def build_simulation_grid(
    df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    last_prices_df: pd.DataFrame,
    guardrails_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the price ladder grid for simulation."""
    log.info("📊 Building simulation grid...")

    # Aggregate SKU-channel level stats from training data
    sku_stats = (
        df.groupby(["sku", "channel"])
        .agg(
            tags=("tags", "first"),
            cost_price=("cost_price", "first"),
            mrp=("mrp", "first"),
            avg_price=("price", "mean"),
            avg_units=("units_sold", "mean"),
            max_units=("units_sold", "max"),
            tag_encoded=("tag_encoded", "first"),
            channel_encoded=("channel_encoded", "first"),
            rolling_units_7d=("rolling_units_7d", "last"),
            rolling_units_14d=("rolling_units_14d", "last"),
            rolling_units_30d=("rolling_units_30d", "last"),
            price_volatility=("price_volatility", "last"),
        )
        .reset_index()
    )

    # Merge inventory and last price
    sku_stats = sku_stats.merge(inventory_df, on="sku", how="inner")
    sku_stats = sku_stats.merge(last_prices_df[["sku", "channel", "last_price"]], on=["sku", "channel"], how="inner")
    sku_stats = sku_stats.merge(guardrails_df, on=["sku", "channel"], how="left")

    # Build price ladder per SKU-channel
    rows = []
    today = pd.Timestamp.now()
    today_dow = today.dayofweek  # 0=Mon → convert to SQL DOW (0=Sun)
    sql_dow = (today_dow + 1) % 7
    today_month = today.month
    today_dom = today.day
    today_woy = today.isocalendar().week

    for _, s in sku_stats.iterrows():
        last_p = s["last_price"]
        low = int(round(last_p * 0.85))
        high = int(round(last_p * 1.15))

        # Generate price points
        price_points = set(range(low, high + 1, 10))

        # Add psychological anchors
        price_points.add(int(round(last_p / 50) * 50 - 1))
        price_points.add(int(round(last_p / 100) * 100 - 1))
        price_points.add(int(round(last_p / 100) * 100 + 99))

        # Add guardrail prices
        if pd.notna(s.get("bau_price")):
            price_points.add(int(s["bau_price"]))
        if pd.notna(s.get("event_price")):
            price_points.add(int(s["event_price"]))

        # Filter to valid range
        price_points = sorted([p for p in price_points if low <= p <= high and p > 0])

        for pp in price_points:
            avg_p = s["avg_price"] if s["avg_price"] > 0 else 1
            mrp_val = s["mrp"] if pd.notna(s["mrp"]) and s["mrp"] > 0 else pp
            cost_val = s["cost_price"] if pd.notna(s["cost_price"]) else 0

            row = {
                "sku": s["sku"],
                "channel": s["channel"],
                "tags": s["tags"],
                "cost_price": cost_val,
                "mrp": mrp_val,
                "last_price": last_p,
                "inventory": s["inventory"],
                "bau_price": s.get("bau_price"),
                "event_price": s.get("event_price"),

                # Features for prediction
                "price": pp,
                "price_ratio_to_avg": pp / avg_p,
                "discount_depth": max(0, 1.0 - (pp / mrp_val)),
                "margin_pct": (pp - cost_val) / pp if pp > 0 else 0,
                "price_bucket": round(pp / 50) * 50,
                "dow": sql_dow,
                "month": today_month,
                "day_of_month": today_dom,
                "week_of_year": today_woy,
                "is_weekend": 1 if sql_dow in [0, 6] else 0,
                "is_month_start": 1 if today_dom <= 5 else 0,
                "is_month_end": 1 if today_dom >= 25 else 0,
                "is_event": 0,  # will be overridden below
                "tag_encoded": s["tag_encoded"],
                "channel_encoded": s["channel_encoded"],
                "days_ago": 0,
                "recency_weight": 1.0,
                "rolling_units_7d": s["rolling_units_7d"],
                "rolling_units_14d": s["rolling_units_14d"],
                "rolling_units_30d": s["rolling_units_30d"],
                "price_volatility": s["price_volatility"],
            }
            rows.append(row)

    grid_df = pd.DataFrame(rows)
    log.info(f"   → Built grid with {len(grid_df):,} price points across "
             f"{grid_df[['sku','channel']].drop_duplicates().shape[0]} SKU-channel combos")
    return grid_df


def simulate_with_model(model, grid_df: pd.DataFrame) -> pd.DataFrame:
    """Run predictions and score results."""
    log.info("🎯 Running ML demand predictions...")

    # Predict for BAU scenario
    grid_df["is_event"] = 0
    X_bau = grid_df[FEATURE_COLS].values
    grid_df["ml_drr_bau"] = np.maximum(model.predict(X_bau), 0)

    # Predict for Event scenario
    grid_df["is_event"] = 1
    X_event = grid_df[FEATURE_COLS].values
    grid_df["ml_drr_event"] = np.maximum(model.predict(X_event), 0)

    # Reset event flag (will use actual today value later)
    grid_df["is_event"] = 0  # default BAU; engine caller can toggle

    # Use BAU prediction as primary DRR
    grid_df["ml_drr"] = grid_df["ml_drr_bau"]


    # Compute metrics
    grid_df["ml_doh"] = grid_df["inventory"] / grid_df["ml_drr"].replace(0, np.nan)
    grid_df["ml_monthly_profit"] = (
        np.maximum(grid_df["price"] - grid_df["cost_price"].fillna(0), 0)
        * grid_df["ml_drr"]
        * 30
    )

    # Inventory urgency score
    def compute_urgency(row):
        if row["inventory"] <= 0:
            return 0.0
            
        tag = row["tags"] if pd.notna(row["tags"]) else "REGULAR"
        target_doh = TAG_CONFIG.get(tag, DEFAULT_TAG_CONFIG)["target_doh"]
        current_doh = row["ml_doh"] if pd.notna(row["ml_doh"]) and row["ml_doh"] > 0 else 999
        return 1.0 / (1.0 + np.exp(-(current_doh - target_doh) / 15.0))

    grid_df["urgency_score"] = grid_df.apply(compute_urgency, axis=1)

    log.info(f"   → Generated {len(grid_df):,} predictions")
    return grid_df


def score_and_rank(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Apply dynamic multi-objective scoring and select top picks."""
    log.info("🏆 Scoring and ranking recommendations...")

    # Group-level max for normalization
    group_max = grid_df.groupby(["sku", "channel"]).agg(
        max_profit=("ml_monthly_profit", "max"),
        max_drr=("ml_drr", "max"),
    ).reset_index()

    grid_df = grid_df.merge(group_max, on=["sku", "channel"], how="left")

    # Dynamic weights based on urgency
    def compute_score(row):
        urgency = row["urgency_score"]
        if urgency > 0.7:
            w_profit, w_drr, w_doh, w_prox = 0.20, 0.45, 0.25, 0.10
        elif urgency < 0.3:
            w_profit, w_drr, w_doh, w_prox = 0.55, 0.15, 0.20, 0.10
        else:
            w_profit, w_drr, w_doh, w_prox = 0.45, 0.30, 0.15, 0.10

        max_p = row["max_profit"] if row["max_profit"] > 0 else 1
        max_d = row["max_drr"] if row["max_drr"] > 0 else 1
        doh_val = max(row["ml_doh"], 0.01) if pd.notna(row["ml_doh"]) else 999
        last_p = row["last_price"] if row["last_price"] > 0 else 1

        score = (
            w_profit * (row["ml_monthly_profit"] / max_p) +
            w_drr * (row["ml_drr"] / max_d) +
            w_doh * (1.0 / doh_val) +
            w_prox * (1.0 - abs(row["price"] - row["last_price"]) / last_p)
        )
        return score

    grid_df["composite_score"] = grid_df.apply(compute_score, axis=1)

    # Rank within each SKU-channel
    grid_df["rn_profit"] = (
        grid_df.groupby(["sku", "channel"])["ml_monthly_profit"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    grid_df["rn_drr"] = (
        grid_df.groupby(["sku", "channel"])["ml_drr"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    grid_df["rn_optimal"] = (
        grid_df.groupby(["sku", "channel"])["composite_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Flag top picks
    grid_df["is_max_profit"] = grid_df["rn_profit"] == 1
    grid_df["is_max_drr"] = grid_df["rn_drr"] == 1
    grid_df["is_optimal"] = grid_df["rn_optimal"] == 1

    return grid_df


# ─────────────────────────────────────────────────────
# STEP 6: RUN ENHANCED SQL ENGINE
# ─────────────────────────────────────────────────────
def run_sql_engine():
    """Execute the enhanced SQL engine and save to CSV."""
    log.info("🔄 Running enhanced SQL engine...")

    sql_path = Path(__file__).parent / "enhanced_pricing_engine.sql"
    if not sql_path.exists():
        log.warning("⚠️  enhanced_pricing_engine.sql not found, skipping SQL engine.")
        return None

    sql = sql_path.read_text()

    # Remove INSERT INTO line if present, we just want SELECT
    lines = sql.split("\n")
    cleaned_lines = [l for l in lines if not l.strip().upper().startswith("INSERT INTO")]
    sql_query = "\n".join(cleaned_lines)

    try:
        df = run_query(sql_query)
        output_path = OUTPUT_DIR / "sql_enhanced_results.csv"
        df.to_csv(output_path, index=False)
        log.info(f"   → SQL engine: {len(df):,} rows → {output_path}")
        return df
    except Exception as e:
        log.error(f"   ❌ SQL engine failed: {e}")
        return None


# ─────────────────────────────────────────────────────
# STEP 7: OUTPUT
# ─────────────────────────────────────────────────────
def save_outputs(grid_df: pd.DataFrame):
    """Save all outputs to CSV."""
    log.info("💾 Saving outputs...")

    # Full predictions
    output_cols = [
        "sku", "channel", "tags", "price", "last_price",
        "cost_price", "inventory", "urgency_score",
        "ml_drr_bau", "ml_drr_event", "ml_drr",
        "ml_doh", "ml_monthly_profit",
        "composite_score",
        "is_max_profit", "is_max_drr", "is_optimal",
    ]

    all_preds = grid_df[output_cols].round(3)
    all_preds.to_csv(OUTPUT_DIR / "ml_predictions.csv", index=False)
    log.info(f"   → All predictions: {len(all_preds):,} rows → ml_predictions.csv")

    # Top recommendations only
    recs = grid_df[
        grid_df["is_max_profit"] | grid_df["is_max_drr"] | grid_df["is_optimal"]
    ][output_cols].round(3)

    # Add recommendation type label
    def get_rec_type(row):
        types = []
        if row["is_max_profit"]: types.append("MAX_PROFIT")
        if row["is_max_drr"]: types.append("MAX_DRR")
        if row["is_optimal"]: types.append("OPTIMAL")
        return " | ".join(types)

    recs = recs.copy()
    recs["recommendation_type"] = recs.apply(get_rec_type, axis=1)
    recs.to_csv(OUTPUT_DIR / "ml_recommendations.csv", index=False)
    log.info(f"   → Recommendations: {len(recs):,} rows → ml_recommendations.csv")

    # Summary stats
    summary = recs.groupby(["channel", "recommendation_type"]).agg(
        avg_price=("price", "mean"),
        avg_drr=("ml_drr", "mean"),
        avg_monthly_profit=("ml_monthly_profit", "mean"),
        count=("sku", "count"),
    ).round(2).reset_index()
    summary.to_csv(OUTPUT_DIR / "ml_summary.csv", index=False)
    log.info(f"   → Summary: {len(summary)} rows → ml_summary.csv")


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    start = datetime.now()
    log.info("=" * 60)
    log.info("🚀 PEPE AI PRICING ENGINE v2.0")
    log.info("=" * 60)

    # 1. Extract data
    df = extract_training_data()
    inventory_df = extract_inventory()
    guardrails_df = extract_guardrails()
    last_prices_df = extract_last_prices()

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Train XGBoost model
    model, cv_metrics, importance = train_model(df)

    # 4. Build simulation grid
    grid_df = build_simulation_grid(df, inventory_df, last_prices_df, guardrails_df)

    # 5. Simulate with ML
    grid_df = simulate_with_model(model, grid_df)

    # 6. Score and rank
    grid_df = score_and_rank(grid_df)

    # 7. Save ML outputs
    save_outputs(grid_df)

    # 8. Run enhanced SQL engine
    sql_results = run_sql_engine()

    elapsed = (datetime.now() - start).total_seconds()
    log.info("=" * 60)
    log.info(f"✅ Complete! Elapsed: {elapsed:.1f}s")
    log.info(f"📁 Outputs in: {OUTPUT_DIR}")
    log.info("=" * 60)

    # Print quick summary
    if grid_df is not None and len(grid_df) > 0:
        optimal = grid_df[grid_df["is_optimal"]]
        print("\n" + "=" * 80)
        print("TOP OPTIMAL RECOMMENDATIONS (sample)")
        print("=" * 80)
        display_cols = ["sku", "channel", "tags", "price", "last_price",
                       "ml_drr", "ml_monthly_profit", "urgency_score"]
        print(optimal[display_cols].head(20).to_string(index=False))
        print()


def run_full_pipeline(progress_callback=None):
    """Run the entire ML pipeline and return the dataframes directly."""
    def log_step(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    log_step("🚀 PIPELINE STARTING...")

    # 1. Extract data
    log_step("📥 Step 1/7: Extracting data from database...")
    df = extract_training_data()
    inventory_df = extract_inventory()
    guardrails_df = extract_guardrails()
    last_prices_df = extract_last_prices()
    log_step(f"   Done. Found {len(df):,} records.")

    # 2. Feature engineering
    log_step("⚙️  Step 2/7: Engineering features...")
    df = engineer_features(df)

    # 3. Train XGBoost model
    log_step("🧠 Step 3/7: Training XGBoost model...")
    model, _, _ = train_model(df)

    # 4. Build simulation grid
    log_step("📏 Step 4/7: Building simulation grid (active stock only)...")
    active_skus = inventory_df[inventory_df["inventory"] > 0]["sku"].unique()
    grid_df = build_simulation_grid(df, inventory_df, last_prices_df, guardrails_df)
    grid_df = grid_df[grid_df["sku"].isin(active_skus)]
    log_step(f"   Done. Simulating {len(active_skus):,} SKUs.")

    # 5. Simulate with ML
    log_step("🔮 Step 5/7: Generating AI predictions...")
    grid_df = simulate_with_model(model, grid_df)

    # 6. Score and rank
    log_step("🏆 Step 6/7: Scoring and ranking...")
    grid_df = score_and_rank(grid_df)
    
    # 7. Format Recommendations
    log_step("📝 Step 7/7: Formatting recommendations...")
    recs = grid_df[
        grid_df["is_max_profit"] | grid_df["is_max_drr"] | grid_df["is_optimal"]
    ].copy()
    
    def get_rec_type(row):
        types = []
        if row["is_max_profit"]: types.append("MAX_PROFIT")
        if row["is_max_drr"]: types.append("MAX_DRR")
        if row["is_optimal"]: types.append("OPTIMAL")
        return " | ".join(types)

    recs["recommendation_type"] = recs.apply(get_rec_type, axis=1)
    
    log_step("✅ PIPELINE COMPLETE.")
    return grid_df, recs



if __name__ == "__main__":
    main()


