# P2 ETF FDA‑SHAPE Engine

**Functional Data Analysis for ETF Shape‑Based Prediction.**

[![GitHub Actions](https://github.com/P2SAMAPA/P2-ETF-FDA-SHAPE/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-FDA-SHAPE/actions/workflows/daily_run.yml)

## Overview

This engine treats rolling windows of ETF returns as smooth functions using B‑splines, extracts dominant shape features via functional PCA, and predicts next‑day returns with a lightweight Ridge model. It complements traditional factor models by capturing the "term structure of momentum."

**Key Features:**
- **Functional Smoothing**: B‑spline basis with automatic smoothness selection.
- **fPCA Decomposition**: Extracts level, slope, and curvature components.
- **Cross‑Validated Window Selection**: Global training optimizes lookback window.
- **Adaptive Window**: Uses change‑point detection for regime‑aware lookback.
- **Three Universes**: FI/Commodities, Equity Sectors, and Combined.

## Data

- **Input**: `P2SAMAPA/fi-etf-macro-signal-master-data` (master_data.parquet)
- **Output**: `P2SAMAPA/p2-etf-fda-shape-results`

## Usage

```bash
pip install -r requirements.txt
python trainer.py           # Runs training and pushes to HF
streamlit run streamlit_app.py
Configuration
All parameters are in config.py:

CANDIDATE_WINDOWS: lookbacks tested in cross‑validation.

FPCA_COMPONENTS: number of functional principal components.

INCLUDE_DERIVATIVES: whether to add first/second derivative features.

CP_PENALTY: sensitivity of change‑point detection.
