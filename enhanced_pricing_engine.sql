-- =====================================================================
-- PEPE AI-ENHANCED PRICING ENGINE v2.0
-- =====================================================================
-- Enhancements over v1.0:
--   1. Event/BAU-aware split elasticity
--   2. Lifecycle-aware demand decay (by Tag)
--   3. Inventory velocity & urgency scoring
--   4. Cross-SKU elasticity transfer (Bayesian pooling)
--   5. Day-of-week & monthly seasonality multipliers
--   6. Dynamic multi-objective optimization weights
--   7. Guardrail enforcement from pepe_pricing_guardrails
-- =====================================================================

-- =====================================================================
-- 0B. FLIPKART UPLIFT (FEES) SLABS
-- =====================================================================
WITH
-- =====================================================================
-- 0B. FLIPKART UPLIFT (FEES) SLABS
-- =====================================================================
fk_uplift AS (
    SELECT *
    FROM (
        VALUES
        ('BOXER',       0,   150,  58),
        ('BOXER',     151,   300,  90),
        ('BOXER',     301,   500,  92),
        ('BOXER',     501,  1000, 135),
        ('BOXER',    1001, 999999,138),

        ('BRIEF',       0,   150,  33),
        ('BRIEF',     151,   300,  33),
        ('BRIEF',     301,   500,   0),
        ('BRIEF',     501,  1000, 131),
        ('BRIEF',    1001, 999999,139),

        ('TRACKSUIT',   0,   150,  93),
        ('TRACKSUIT', 151,   300,  93),
        ('TRACKSUIT', 301,   500, 103),
        ('TRACKSUIT', 501,  1000, 129),
        ('TRACKSUIT',1001, 999999,155),

        ('T SHIRT',     0,   150,  58),
        ('T SHIRT',   151,   300,  78),
        ('T SHIRT',   301,   500,  80),
        ('T SHIRT',   501,  1000, 157),
        ('T SHIRT',  1001, 999999,192),

        ('SHORTS',      0,   150,  62),
        ('SHORTS',    151,   300,  94),
        ('SHORTS',    301,   500,  93),
        ('SHORTS',    501,  1000,  96),
        ('SHORTS',   1001, 999999,148),

        ('PYJAMA',      0,   150,  32),
        ('PYJAMA',    151,   300,  84),
        ('PYJAMA',    301,   500,  90),
        ('PYJAMA',    501,  1000, 108),
        ('PYJAMA',   1001, 999999,145),

        ('SOCKS',       0,   150,  58),
        ('SOCKS',     151,   300,  90),
        ('SOCKS',     301,   500,  84),
        ('SOCKS',     501,  1000, 101),
        ('SOCKS',    1001, 999999,159),

        ('THERMALS',    0,   150, 118),
        ('THERMALS',  151,   300, 118),
        ('THERMALS',  301,   500, 118),
        ('THERMALS',  501,  1000, 154),
        ('THERMALS', 1001, 999999,169),

        ('TRACK PANT',  0,   150,  97),
        ('TRACK PANT',151,   300,  77),
        ('TRACK PANT',301,   500,  80),
        ('TRACK PANT',501,  1000, 147),
        ('TRACK PANT',1001, 999999,248),

        ('VEST',        0,   150,  27),
        ('VEST',      151,   300,  27),
        ('VEST',      301,   500,   0),
        ('VEST',      501,  1000,  82),
        ('VEST',     1001, 999999, 75)
    ) AS t(category, min_price, max_price, uplift)
),

-- =====================================================
-- 0. ITEMMASTER DEDUP
-- =====================================================
mapping_dedup AS (
    SELECT *
    FROM (
        SELECT
            "Product Code",
            "Cost Price",
            "Tags",
            "MRP",
            "Category Name" AS category,
            ROW_NUMBER() OVER (
                PARTITION BY "Product Code"
                ORDER BY "updatedAt" DESC NULLS LAST
            ) AS rn
        FROM "DataWarehouse".pepe_in_unicommerce_itemmaster
    ) x
    WHERE rn = 1
),

-- =====================================================
-- 1. BASE SALES  (AJIO / Flipkart / Myntra only)
-- =====================================================
base_sales AS (
    SELECT
        s."Item SKU Code"                              AS sku,
        DATE(s."Order Date as dd/mm/yyyy hh:MM:ss")    AS order_date,

        CASE
            WHEN s."Channel Name" ILIKE '%MYNTRA%'   THEN 'MYNTRA'
            WHEN s."Channel Name" ILIKE '%FLIPKART%' THEN 'FLIPKART'
            WHEN s."Channel Name" ILIKE '%AJIO%'     THEN 'AJIO'
        END AS channel,

        ROUND(
            CASE
                WHEN s."Channel Name" ILIKE '%AJIO%'
                THEN s."Selling Price" + 65
                ELSE s."Selling Price"
            END
        )::INT AS price,

        md."Tags"        AS tags,
        md."category"    AS category,
        md."Cost Price"  AS cost_price
    FROM "DataWarehouse".pepe_in_unicommerce_saleorders s
    LEFT JOIN mapping_dedup md
        ON s."Item SKU Code" = md."Product Code"
    WHERE s."Order Date as dd/mm/yyyy hh:MM:ss" >= '2025-01-01'
      AND s."Sale Order Item Status" NOT IN ('CANCELLED','UNFULFILLABLE')
      AND s."Selling Price" > 0
      AND s."Channel Name" NOT ILIKE '%SOR%'
      AND s."Bundle SKU Code Number" = ''
      AND (
            s."Channel Name" ILIKE '%MYNTRA%'
         OR s."Channel Name" ILIKE '%FLIPKART%'
         OR s."Channel Name" ILIKE '%AJIO%'
          )
),

-- =====================================================
-- 1B. EVENT CALENDAR UNPIVOT
--     (Flatten the wide event_calendar into long form)
-- =====================================================
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

-- =====================================================
-- 1C. TAG EACH SALE AS BAU OR EVENT
-- =====================================================
base_sales_tagged AS (
    SELECT
        bs.*,
        COALESCE(ec.event_type, 'BAU') AS event_type
    FROM base_sales bs
    LEFT JOIN event_calendar_long ec
        ON bs.order_date = ec.cal_date
       AND bs.channel    = ec.channel
),

-- =====================================================
-- 2. LAST PRICE
-- =====================================================
last_txn_price AS (
    SELECT *
    FROM (
        SELECT
            sku,
            channel,
            price       AS last_price,
            order_date,
            ROW_NUMBER() OVER (
                PARTITION BY sku, channel
                ORDER BY order_date DESC
            ) rn
        FROM base_sales
    ) x
    WHERE rn = 1
),

-- =====================================================
-- 2B. IS TODAY BAU OR EVENT?
-- =====================================================
today_event AS (
    SELECT
        channel,
        COALESCE(event_type, 'BAU') AS today_event_type
    FROM event_calendar_long
    WHERE cal_date = CURRENT_DATE

    UNION ALL
    -- Fallback rows for channels missing from calendar
    SELECT ch, 'BAU'
    FROM (VALUES ('MYNTRA'),('FLIPKART'),('AJIO')) AS v(ch)
    WHERE ch NOT IN (
        SELECT channel FROM event_calendar_long
        WHERE cal_date = CURRENT_DATE
    )
),

-- =====================================================
-- 3. PRICE BUCKETING
-- =====================================================
bucketed_sales AS (
    SELECT
        sku,
        channel,
        tags,
        order_date,
        event_type,
        ROUND(price / 5.0) * 5 AS price_point
    FROM base_sales_tagged
),

-- =====================================================
-- 4. RECENCY-WEIGHTED PRICE LEARNING
--    Enhancement: Lifecycle-aware decay constants
--      MUDA:    0.002  (long memory — slow movers need more history)
--      NEW:     0.010  (fast learning — recent trend matters most)
--      SUPER40: 0.006  (slightly faster than default)
--      CORE:    0.004  (stable products, balanced memory)
--      DEFAULT: 0.005
-- =====================================================
price_learning AS (
    SELECT
        sku,
        channel,
        tags,
        category,
        price_point,
        event_type,
        SUM(weight)                                      AS weighted_qty,
        COUNT(DISTINCT order_date)                       AS exposure_days,
        SUM(weight) / NULLIF(COUNT(DISTINCT order_date), 0) AS drr
    FROM (
        SELECT
            sku,
            channel,
            tags,
            category,
            price_point,
            order_date,
            event_type,
            EXP(
                - (CASE
                    WHEN tags = 'MUDA'    THEN 0.002
                    WHEN tags = 'NEW'     THEN 0.010
                    WHEN tags = 'SUPER40' THEN 0.006
                    WHEN tags = 'CORE'    THEN 0.004
                    ELSE                       0.005
                END)
                * (CURRENT_DATE - order_date)
            ) AS weight
        FROM bucketed_sales
    ) x
    GROUP BY sku, channel, tags, category, price_point, event_type
    HAVING COUNT(DISTINCT order_date) >= 3
),

-- =====================================================
-- 4B. SPLIT ELASTICITY MODEL (BAU vs EVENT)
-- =====================================================
elasticity_model AS (
    SELECT
        sku,
        channel,
        event_type,
        CASE
            WHEN COUNT(*) < 3 THEN NULL  -- will be filled via cross-SKU transfer
            WHEN REGR_R2(LN(drr), LN(price_point)) < 0.2 THEN NULL
            ELSE GREATEST(REGR_SLOPE(LN(drr), LN(price_point)), -2.5)
        END AS elasticity,
        COUNT(*) AS n_points
    FROM price_learning
    WHERE drr > 0
    GROUP BY sku, channel, event_type
),

-- =====================================================
-- 4C. CATEGORY-LEVEL (TAG) POOLED ELASTICITY
--     For cross-SKU transfer / cold-start
-- =====================================================
category_elasticity AS (
    SELECT
        pl.tags,
        pl.channel,
        pl.event_type,
        CASE
            WHEN COUNT(*) < 5 THEN -1.5
            ELSE GREATEST(REGR_SLOPE(LN(pl.drr), LN(pl.price_point)), -2.5)
        END AS cat_elasticity,
        COUNT(*) AS cat_n_points
    FROM price_learning pl
    WHERE pl.drr > 0
    GROUP BY pl.tags, pl.channel, pl.event_type
),

-- =====================================================
-- 4D. BAYESIAN BLENDED ELASTICITY
--     final_e = (n_sku * sku_e + n_prior * cat_e) / (n_sku + n_prior)
--     n_prior = 5 (regularization anchor)
-- =====================================================
blended_elasticity AS (
    SELECT
        COALESCE(em.sku, pl_fallback.sku)       AS sku,
        COALESCE(em.channel, pl_fallback.channel) AS channel,
        COALESCE(em.event_type, pl_fallback.event_type) AS event_type,

        CASE
            -- SKU has its own good elasticity → blend with category
            WHEN em.elasticity IS NOT NULL THEN
                (em.n_points * em.elasticity + 5 * COALESCE(ce.cat_elasticity, -1.5))
                / (em.n_points + 5)
            -- No SKU-level elasticity → use category
            ELSE COALESCE(ce.cat_elasticity, -1.5)
        END AS elasticity,

        CASE
            WHEN em.elasticity IS NOT NULL THEN 'BLENDED'
            WHEN ce.cat_elasticity IS NOT NULL THEN 'CATEGORY_TRANSFER'
            ELSE 'DEFAULT_FALLBACK'
        END AS elasticity_source

    FROM elasticity_model em
    FULL OUTER JOIN (
        -- Get all (sku, channel, event_type) combos that need elasticity
        SELECT DISTINCT sku, channel, tags, event_type
        FROM price_learning
    ) pl_fallback
        ON em.sku = pl_fallback.sku
       AND em.channel = pl_fallback.channel
       AND em.event_type = pl_fallback.event_type
    LEFT JOIN category_elasticity ce
        ON COALESCE(pl_fallback.tags, (SELECT tags FROM mapping_dedup WHERE "Product Code" = em.sku)) = ce.tags
       AND COALESCE(em.channel, pl_fallback.channel) = ce.channel
       AND COALESCE(em.event_type, pl_fallback.event_type) = ce.event_type
),

-- =====================================================
-- 5. INVENTORY + VELOCITY
-- =====================================================
inventory AS (
    SELECT
        "Item SkuCode" AS sku,
        SUM("Inventory"::INT) AS inventory
    FROM "DataWarehouse".pepe_in_unicommerce_inventorysnapshot
    WHERE "Inventory Snapshot Date" = (
        SELECT MAX("Inventory Snapshot Date")
        FROM "DataWarehouse".pepe_in_unicommerce_inventorysnapshot
    )
    GROUP BY 1
),

-- Inventory velocity: avg daily burn over last 14 days
inventory_velocity AS (
    SELECT
        snap.sku,
        CASE
            WHEN snap.days_span > 0
            THEN (snap.first_inv - snap.last_inv)::NUMERIC / snap.days_span
            ELSE 0
        END AS daily_burn_rate
    FROM (
        SELECT
            "Item SkuCode" AS sku,
            MAX(CASE WHEN "Inventory Snapshot Date" = dt_min THEN "Inventory"::INT END) AS first_inv,
            MAX(CASE WHEN "Inventory Snapshot Date" = dt_max THEN "Inventory"::INT END) AS last_inv,
            dt_max - dt_min AS days_span
        FROM "DataWarehouse".pepe_in_unicommerce_inventorysnapshot,
        LATERAL (
            SELECT
                MAX("Inventory Snapshot Date") AS dt_max,
                MAX("Inventory Snapshot Date") - 14 AS dt_min
            FROM "DataWarehouse".pepe_in_unicommerce_inventorysnapshot
        ) dates
        WHERE "Inventory Snapshot Date" IN (dates.dt_min, dates.dt_max)
        GROUP BY "Item SkuCode", dates.dt_max, dates.dt_min
    ) snap
),

-- =====================================================
-- 5B. INVENTORY URGENCY SCORE
--     Sigmoid: 1 / (1 + exp(-(target_doh - current_doh) / scale))
--     Target DOH varies by tag:
--       MUDA=30, NEW=45, SUPER40=50, CORE=90, DEFAULT=60
-- =====================================================
inventory_urgency AS (
    SELECT
        i.sku,
        i.inventory,
        iv.daily_burn_rate,
        i.inventory / NULLIF(GREATEST(iv.daily_burn_rate, 0.01), 0) AS current_doh,
        CASE md."Tags"
            WHEN 'MUDA'    THEN 30
            WHEN 'NEW'     THEN 45
            WHEN 'SUPER40' THEN 50
            WHEN 'CORE'    THEN 90
            ELSE 60
        END AS target_doh,
        -- Urgency: higher when overstocked (current_doh >> target_doh)
        1.0 / (1.0 + EXP(
            -(
                (i.inventory / NULLIF(GREATEST(iv.daily_burn_rate, 0.01), 0))
                - (CASE md."Tags"
                    WHEN 'MUDA'    THEN 30
                    WHEN 'NEW'     THEN 45
                    WHEN 'SUPER40' THEN 50
                    WHEN 'CORE'    THEN 90
                    ELSE 60
                END)
            ) / 15.0
        )) AS urgency_score
    FROM inventory i
    LEFT JOIN inventory_velocity iv ON i.sku = iv.sku
    LEFT JOIN mapping_dedup md ON i.sku = md."Product Code"
),

-- =====================================================
-- 5C. DOW SEASONALITY INDEX (per channel)
-- =====================================================
dow_index AS (
    SELECT
        channel,
        EXTRACT(DOW FROM order_date)::INT AS dow,
        COUNT(*)::NUMERIC / NULLIF(
            AVG(COUNT(*)) OVER (PARTITION BY channel), 0
        ) AS dow_multiplier
    FROM base_sales
    GROUP BY channel, EXTRACT(DOW FROM order_date)
),

-- =====================================================
-- 5D. MONTHLY SEASONALITY INDEX (per channel)
-- =====================================================
monthly_index AS (
    SELECT
        channel,
        EXTRACT(MONTH FROM order_date)::INT AS mth,
        COUNT(*)::NUMERIC / NULLIF(
            AVG(COUNT(*)) OVER (PARTITION BY channel), 0
        ) AS month_multiplier
    FROM base_sales
    GROUP BY channel, EXTRACT(MONTH FROM order_date)
),

-- =====================================================
-- 6. BASE DEMAND (best-performing price point per SKU-channel-event_type)
-- =====================================================
base_demand AS (
    SELECT *
    FROM (
        SELECT
            pl.sku,
            pl.channel,
            pl.tags,
            pl.category,
            pl.event_type,
            pl.price_point          AS base_price,
            pl.drr                  AS base_drr,
            lp.last_price,
            iu.inventory,
            iu.urgency_score,
            iu.current_doh,
            iu.target_doh,
            md."Cost Price"         AS cost_price,
            be.elasticity,
            be.elasticity_source,
            MAX(pl.drr) OVER (PARTITION BY pl.sku, pl.channel, pl.event_type) AS max_hist_drr,
            ROW_NUMBER() OVER (
                PARTITION BY pl.sku, pl.channel, pl.event_type
                ORDER BY pl.drr DESC
            ) rn
        FROM price_learning pl
        JOIN inventory_urgency iu  ON pl.sku = iu.sku
        JOIN mapping_dedup md      ON pl.sku = md."Product Code"
        JOIN last_txn_price lp     ON pl.sku = lp.sku AND pl.channel = lp.channel
        JOIN blended_elasticity be ON pl.sku = be.sku AND pl.channel = be.channel
                                   AND pl.event_type = be.event_type
    ) x
    WHERE rn = 1
),

-- =====================================================
-- 6B. GUARDRAILS LOOKUP
-- =====================================================
guardrails AS (
    SELECT
        "Pcode"   AS sku,
        "Channel" AS channel,
        NULLIF(REPLACE("BAU", ',', ''), '')::NUMERIC   AS bau_price,
        NULLIF(REPLACE("Event", ',', ''), '')::NUMERIC AS event_price
    FROM "DataWarehouse".pepe_pricing_guardrails
),

-- =====================================================
-- 7. PRICE LADDER (respecting guardrails)
-- =====================================================
price_ladder AS (
    SELECT DISTINCT
        b.*,
        p.price_point AS sim_price_point
    FROM base_demand b
    CROSS JOIN LATERAL (
        SELECT price_point
        FROM (
            -- Regular 10-step ladder within 15% band
            SELECT generate_series(
                ROUND(b.last_price * 0.85)::INT,
                ROUND(b.last_price * 1.15)::INT,
                10
            ) AS price_point

            UNION

            -- Psychological anchors
            SELECT (ROUND(b.last_price/50.0)*50 - 1)::INT
            UNION
            SELECT (ROUND(b.last_price/100.0)*100 - 1)::INT
            UNION
            SELECT (ROUND(b.last_price/100.0)*100 + 99)::INT

            UNION

            -- Include guardrail BAU and Event prices directly
            SELECT g.bau_price::INT
            FROM guardrails g
            WHERE g.sku = b.sku AND g.channel = b.channel
              AND g.bau_price IS NOT NULL
            UNION
            SELECT g.event_price::INT
            FROM guardrails g
            WHERE g.sku = b.sku AND g.channel = b.channel
              AND g.event_price IS NOT NULL
        ) x
        WHERE price_point BETWEEN
              ROUND(b.last_price * 0.85)::INT
          AND ROUND(b.last_price * 1.15)::INT
          AND price_point > 0
    ) p
),

-- =====================================================
-- HISTORICAL OUTPUT
-- =====================================================
historical AS (
    SELECT
        pl.sku,
        pl.channel,
        pl.tags,
        pl.event_type,
        pl.price_point,
        pl.drr,
        i.inventory / NULLIF(pl.drr, 0)        AS doh,
        -- Monthly Profit Calculation with Channel Fees
        GREATEST(
            (CASE
                WHEN pl.channel = 'FLIPKART' THEN
                    (pl.price_point * 0.82) - COALESCE(fu.uplift, 145) -- 18% comm + slab fee
                WHEN pl.channel = 'MYNTRA'   THEN (pl.price_point * 0.82) - 35
                ELSE pl.price_point - md."Cost Price"
             END) - md."Cost Price",
            0
        ) * pl.drr * 30                       AS monthly_profit,
        lp.last_price,
        NULL::NUMERIC                            AS urgency_score,
        NULL::TEXT                               AS elasticity_source,
        FALSE                                   AS is_max_profit,
        FALSE                                   AS is_max_drr,
        FALSE                                   AS is_optimal,
        'HISTORICAL'                            AS record_type
    FROM price_learning pl
    JOIN inventory i        ON pl.sku = i.sku
    JOIN mapping_dedup md   ON pl.sku = md."Product Code"
    JOIN last_txn_price lp  ON pl.sku = lp.sku AND pl.channel = lp.channel
    LEFT JOIN fk_uplift fu  ON pl.channel = 'FLIPKART'
                           AND UPPER(pl.category) = fu.category
                           AND pl.price_point BETWEEN fu.min_price AND fu.max_price
),

-- =====================================================
-- 8. SIMULATION
--    Enhanced demand model with seasonality multipliers
-- =====================================================
simulated_raw AS (
    SELECT
        pl.sku,
        pl.channel,
        pl.tags,
        pl.category,
        pl.event_type,
        pl.sim_price_point AS price_point,
        pl.elasticity,
        pl.elasticity_source,
        pl.urgency_score,
        pl.inventory,
        pl.cost_price,
        pl.last_price,
        pl.max_hist_drr,
        pl.base_drr,
        pl.base_price,
        pl.current_doh,
        pl.target_doh,

        -- Seasonality multipliers for today
        COALESCE(dw.dow_multiplier, 1.0)   AS dow_mult,
        COALESCE(mi.month_multiplier, 1.0) AS month_mult,

        -- Core demand curve with cap and price-shock dampener
        LEAST(
            pl.base_drr
            * POWER(
                GREATEST(pl.sim_price_point / NULLIF(pl.base_price, 0), 0.1),
                pl.elasticity
              )
            * EXP(-2.0 * ABS(pl.sim_price_point - pl.last_price)
                   / NULLIF(pl.last_price, 1)),
            2.5 * pl.max_hist_drr
        )
        -- Apply seasonality
        * COALESCE(dw.dow_multiplier, 1.0)
        * COALESCE(mi.month_multiplier, 1.0)
        -- Inventory constraint (no phantom demand)
        * LEAST(
            1.0,
            pl.inventory / NULLIF(
                GREATEST(
                    pl.base_drr
                    * POWER(
                        GREATEST(pl.sim_price_point / NULLIF(pl.base_price, 0), 0.1),
                        pl.elasticity
                    )
                    * EXP(-2.0 * ABS(pl.sim_price_point - pl.last_price)
                           / NULLIF(pl.last_price, 1)),
                    0.01
                ) * 30,
                1
            )
          ) AS drr

    FROM price_ladder pl
    LEFT JOIN dow_index dw
        ON pl.channel = dw.channel
       AND dw.dow = EXTRACT(DOW FROM CURRENT_DATE)::INT
    LEFT JOIN monthly_index mi
        ON pl.channel = mi.channel
       AND mi.mth = EXTRACT(MONTH FROM CURRENT_DATE)::INT
),

simulated_enriched AS (
    SELECT
        sr.sku,
        sr.channel,
        sr.tags,
        sr.event_type,
        sr.price_point,
        sr.elasticity,
        sr.elasticity_source,
        sr.urgency_score,
        sr.drr,
        sr.inventory / NULLIF(sr.drr, 0)                        AS doh,
        -- Monthly Profit with Slab-aware Fees
        GREATEST(
            (CASE
                WHEN sr.channel = 'FLIPKART' THEN
                    (sr.price_point * 0.82) - COALESCE(fu.uplift, 145)
                WHEN sr.channel = 'MYNTRA'   THEN (sr.price_point * 0.82) - 35
                ELSE sr.price_point - sr.cost_price
             END) - sr.cost_price,
            0
        ) * sr.drr * 30                                         AS monthly_profit,
        sr.last_price,
        sr.current_doh,
        sr.target_doh,
        sr.dow_mult,
        sr.month_mult,
        'SIMULATED' AS record_type
    FROM simulated_raw sr
    LEFT JOIN fk_uplift fu  ON sr.channel = 'FLIPKART'
                           AND UPPER(sr.category) = fu.category
                           AND sr.price_point BETWEEN fu.min_price AND fu.max_price
),

simulated_with_max AS (
    SELECT
        *,
        MAX(monthly_profit) OVER (PARTITION BY sku, channel) AS max_profit,
        MAX(drr) OVER (PARTITION BY sku, channel)            AS max_drr
    FROM simulated_enriched
),

-- =====================================================
-- 9. DYNAMIC SCORING
--    Weights shift based on inventory urgency:
--      OVERSTOCK (urgency > 0.7): velocity mode  → profit 0.20, drr 0.45, doh 0.25, prox 0.10
--      BALANCED  (0.3–0.7):      balanced mode   → profit 0.45, drr 0.30, doh 0.15, prox 0.10
--      LIMITED   (urgency < 0.3): margin mode     → profit 0.55, drr 0.15, doh 0.20, prox 0.10
-- =====================================================
scored AS (
    SELECT
        *,

        -- Dynamic weights
        CASE
            WHEN urgency_score > 0.7 THEN 0.20
            WHEN urgency_score < 0.3 THEN 0.55
            ELSE 0.45
        END AS w_profit,
        CASE
            WHEN urgency_score > 0.7 THEN 0.45
            WHEN urgency_score < 0.3 THEN 0.15
            ELSE 0.30
        END AS w_drr,
        CASE
            WHEN urgency_score > 0.7 THEN 0.25
            WHEN urgency_score < 0.3 THEN 0.20
            ELSE 0.15
        END AS w_doh,
        0.10 AS w_prox,

        ROW_NUMBER() OVER (
            PARTITION BY sku, channel
            ORDER BY monthly_profit DESC
        ) AS rn_profit,

        ROW_NUMBER() OVER (
            PARTITION BY sku, channel
            ORDER BY drr DESC
        ) AS rn_drr,

        ROW_NUMBER() OVER (
            PARTITION BY sku, channel
            ORDER BY
                -- Dynamic weighted composite score
                (CASE WHEN urgency_score > 0.7 THEN 0.20
                      WHEN urgency_score < 0.3 THEN 0.55
                      ELSE 0.45
                 END) * (monthly_profit / NULLIF(max_profit, 0))
              + (CASE WHEN urgency_score > 0.7 THEN 0.45
                      WHEN urgency_score < 0.3 THEN 0.15
                      ELSE 0.30
                 END) * (drr / NULLIF(max_drr, 0))
              + (CASE WHEN urgency_score > 0.7 THEN 0.25
                      WHEN urgency_score < 0.3 THEN 0.20
                      ELSE 0.15
                 END) * (1.0 / GREATEST(doh, 0.01))
              + 0.10  * (1.0 - ABS(price_point - last_price) / NULLIF(last_price, 1))
            DESC
        ) AS rn_optimal
    FROM simulated_with_max
)

-- =====================================================
-- FINAL OUTPUT
-- =====================================================
SELECT
    sku,
    channel,
    tags,
    event_type,
    price_point,
    ROUND(drr::NUMERIC, 2)              AS drr,
    ROUND(doh::NUMERIC, 2)              AS doh,
    ROUND(monthly_profit::NUMERIC, 0)   AS monthly_profit,
    last_price,
    urgency_score,
    elasticity_source,
    is_max_profit,
    is_max_drr,
    is_optimal,
    record_type,
    CURRENT_DATE                        AS run_date
FROM historical

UNION ALL

SELECT
    sku,
    channel,
    tags,
    event_type,
    price_point,
    ROUND(drr::NUMERIC, 2),
    ROUND(doh::NUMERIC, 2),
    ROUND(monthly_profit::NUMERIC, 0),
    last_price,
    ROUND(urgency_score::NUMERIC, 3),
    elasticity_source,
    rn_profit = 1,
    rn_drr = 1,
    rn_optimal = 1,
    record_type,
    CURRENT_DATE
FROM scored
WHERE rn_profit  = 1
   OR rn_drr     = 1
   OR rn_optimal = 1;
