# Sales-Forecasting-System
An end-to-end supply chain intelligence system. Features time-series forecasting (SARIMA), anomaly detection (Isolation Forest), demand segmentation (K-Means), and a live Streamlit executive dashboard for inventory optimization.

**Live Executive Dashboard:** https://sales-forecasting-system-inarss5vdkkywuv493bvh7.streamlit.app/

## Executive Summary
This repository contains an end-to-end demand intelligence pipeline designed to optimize retail supply chain operations. Built to address the costly balance between inventory overstock and stock-outs, this system ingests historical transactional data to project future baseline demand, isolate supply chain shocks, and categorize inventory based on operational volatility. 

The project culminates in a live, interactive web application deployed via Streamlit, allowing supply chain managers and financial executives to make data-backed working capital allocations.

## Core Architecture & Features

### 1. Predictive Forecasting Engine
Tested against Prophet and XGBoost, a rigorously tuned **SARIMA** model was selected as the production champion. It accurately captures core growth trends and rigid Q4 seasonal dependencies to project a reliable 90-day inventory baseline across product categories and geographic regions.

### 2. Supply Chain Anomaly Auditing
To prevent historical data distortions from ruining future forecasts, the system deploys a dual-method anomaly detection protocol:
*   **Isolation Forest (Machine Learning):** Detects global volume outliers across a 4-year timeline.
*   **Rolling Z-Score (Statistical):** Identifies sudden, local demand shocks outside a 12-week rolling baseline. 

### 3. Strategic Inventory Segmentation (Clustering)
Products are dynamically grouped using **K-Means Clustering** and reduced via **PCA** into 4 distinct operational archetypes:
*   **High Volume/Stable:** Candidates for automated regular replenishment.
*   **High Volatility:** Candidates for centralized warehousing or supplier drop-shipping.
*   **High Growth:** Priority capital allocation to prevent momentum-stunting stock-outs.
*   **Stagnant:** Targeted for aggressive liquidation.
