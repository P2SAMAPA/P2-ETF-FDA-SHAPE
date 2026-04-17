"""
Global (with CV window selection) and Adaptive Window training.
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error

import config
from data_manager import load_master_data, prepare_data, get_universe_returns
from fda_processor import create_multivariate_fdata, fit_fpca, extract_shape_features
from model import ShapePredictor
from change_point_detector import universe_adaptive_start_date
from push_results import push_daily_result


def evaluate_etf(ticker: str, returns: pd.DataFrame) -> dict:
    col = f"{ticker}_ret"
    if col not in returns.columns:
        return {}
    ret_series = returns[col].dropna()
    if len(ret_series) < 5:
        return {}
    ann_return = ret_series.mean() * config.TRADING_DAYS_PER_YEAR
    ann_vol = ret_series.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    cum = (1 + ret_series).cumprod()
    rolling_max = cum.expanding().max()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()
    hit_rate = (ret_series > 0).mean()
    cum_return = (1 + ret_series).prod() - 1
    return {
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "max_dd": max_dd, "hit_rate": hit_rate, "cum_return": cum_return,
        "n_days": len(ret_series)
    }


def create_window_samples(returns: pd.DataFrame, window_size: int):
    """Create sliding windows of shape (n_samples, n_features, window_size)."""
    data = returns.values
    samples = []
    for i in range(len(data) - window_size + 1):
        samples.append(data[i:i+window_size].T)  # (n_features, window_size)
    return np.array(samples)  # (n_samples, n_features, window_size)


def train_global(universe: str, returns: pd.DataFrame) -> dict:
    print(f"\n--- Global Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    total_days = len(returns)
    train_end = int(total_days * config.TRAIN_RATIO)
    val_end = train_end + int(total_days * config.VAL_RATIO)

    train_ret = returns.iloc[:train_end]
    val_ret = returns.iloc[train_end:val_end]
    test_ret = returns.iloc[val_end:]

    # ----- Cross‑validated window selection -----
    best_window = None
    best_val_mse = float('inf')

    print("  Selecting optimal window via cross‑validation...")
    for window in config.CANDIDATE_WINDOWS:
        if len(train_ret) < window + 20:
            continue

        n_basis = min(15, window // config.N_BASIS_FACTOR)
        train_samples = create_window_samples(train_ret, window)
        val_samples = create_window_samples(val_ret, window)

        if len(train_samples) < 10 or len(val_samples) < 5:
            continue

        # FDApy pipeline
        train_smoothed = create_multivariate_fdata(train_samples, n_basis=n_basis, smoothing_parameter=config.SMOOTHING_PENALTY)
        fpca_models, train_scores_list = fit_fpca(train_smoothed, n_components=config.FPCA_COMPONENTS)
        train_features = extract_shape_features(train_smoothed, fpca_models, train_scores_list, include_derivatives=config.INCLUDE_DERIVATIVES)

        val_smoothed = create_multivariate_fdata(val_samples, n_basis=n_basis, smoothing_parameter=config.SMOOTHING_PENALTY)
        _, val_scores_list = fit_fpca(val_smoothed, n_components=config.FPCA_COMPONENTS, refit=False, fpca_models=fpca_models)
        val_features = extract_shape_features(val_smoothed, fpca_models, val_scores_list, include_derivatives=config.INCLUDE_DERIVATIVES)

        # Targets: next‑day returns for each ETF
        y_train = train_ret.shift(-1).iloc[window-1:len(train_samples)+window-1].values
        y_val = val_ret.shift(-1).iloc[window-1:len(val_samples)+window-1].values

        # Train one Ridge per ETF
        val_preds = np.zeros_like(y_val)
        for i, ticker in enumerate(tickers):
            predictor = ShapePredictor()
            valid_train = ~np.isnan(y_train[:, i])
            valid_val = ~np.isnan(y_val[:, i])
            if valid_train.sum() < 10:
                continue
            X_tr = train_features.iloc[valid_train]
            y_tr = y_train[valid_train, i]
            X_val = val_features.iloc[valid_val]
            predictor.fit(X_tr, y_tr)
            val_preds[valid_val, i] = predictor.predict(X_val)

        mse = mean_squared_error(y_val[~np.isnan(y_val)], val_preds[~np.isnan(y_val)])
        print(f"    Window {window:3d} -> Validation MSE: {mse:.6f}")

        if mse < best_val_mse:
            best_val_mse = mse
            best_window = window

    if best_window is None:
        best_window = 60  # fallback

    print(f"  Selected optimal window: {best_window} days (MSE: {best_val_mse:.6f})")

    # ----- Train final model on train+val -----
    train_val_ret = pd.concat([train_ret, val_ret])
    n_basis = min(15, best_window // config.N_BASIS_FACTOR)
    samples = create_window_samples(train_val_ret, best_window)
    smoothed_data = create_multivariate_fdata(samples, n_basis=n_basis, smoothing_parameter=config.SMOOTHING_PENALTY)
    fpca_models, scores_list = fit_fpca(smoothed_data, n_components=config.FPCA_COMPONENTS)
    features = extract_shape_features(smoothed_data, fpca_models, scores_list, include_derivatives=config.INCLUDE_DERIVATIVES)
    y_all = train_val_ret.shift(-1).iloc[best_window-1:len(samples)+best_window-1].values

    predictors = {}
    for i, ticker in enumerate(tickers):
        valid = ~np.isnan(y_all[:, i])
        if valid.sum() < 10:
            continue
        X_tr = features.iloc[valid]
        y_tr = y_all[valid, i]
        pred = ShapePredictor().fit(X_tr, y_tr)
        predictors[ticker] = pred

    # ----- Predict on test set -----
    test_samples = create_window_samples(test_ret, best_window)
    if len(test_samples) > 0:
        test_smoothed = create_multivariate_fdata(test_samples, n_basis=n_basis, smoothing_parameter=config.SMOOTHING_PENALTY)
        _, test_scores_list = fit_fpca(test_smoothed, n_components=config.FPCA_COMPONENTS, refit=False, fpca_models=fpca_models)
        test_features = extract_shape_features(test_smoothed, fpca_models, test_scores_list, include_derivatives=config.INCLUDE_DERIVATIVES)
        latest_features = test_features.iloc[-1:]

        pred_returns = {}
        for ticker, pred in predictors.items():
            try:
                pred_returns[ticker] = pred.predict(latest_features)[0]
            except:
                pred_returns[ticker] = -np.inf

        best_ticker = max(pred_returns, key=pred_returns.get)
        best_pred_return = pred_returns[best_ticker]
    else:
        best_ticker = tickers[0]
        best_pred_return = 0.0
        pred_returns = {t: 0.0 for t in tickers}

    metrics = evaluate_etf(best_ticker, test_ret)
    print(f"  Selected ETF: {best_ticker}, Predicted Return: {best_pred_return*100:.2f}%")
    return {
        "ticker": best_ticker,
        "pred_return": best_pred_return,
        "all_pred_returns": pred_returns,
        "optimal_window": best_window,
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d"),
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d"),
    }


def train_adaptive(universe: str, returns: pd.DataFrame) -> dict:
    print(f"\n--- Adaptive Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    cp_date = universe_adaptive_start_date(returns)
    print(f"  Adaptive window starts: {cp_date.date()}")

    end_date = returns.index[-1] - pd.Timedelta(days=config.MIN_TEST_DAYS)
    if end_date <= cp_date:
        end_date = returns.index[-1] - pd.Timedelta(days=10)
    train_mask = (returns.index >= cp_date) & (returns.index <= end_date)
    train_ret = returns.loc[train_mask]
    test_ret = returns.loc[returns.index > end_date]

    if len(train_ret) < config.MIN_TRAIN_DAYS:
        print("  Insufficient training days. Falling back to global.")
        return train_global(universe, returns)

    lookback = min(config.ADAPTIVE_MAX_LOOKBACK, (returns.index[-1] - cp_date).days)
    lookback = max(lookback, 20)  # minimum

    n_basis = min(15, lookback // config.N_BASIS_FACTOR)
    samples = create_window_samples(train_ret, lookback)
    smoothed_data = create_multivariate_fdata(samples, n_basis=n_basis, smoothing_parameter=config.SMOOTHING_PENALTY)
    fpca_models, scores_list = fit_fpca(smoothed_data, n_components=config.FPCA_COMPONENTS)
    features = extract_shape_features(smoothed_data, fpca_models, scores_list, include_derivatives=config.INCLUDE_DERIVATIVES)
    y_train = train_ret.shift(-1).iloc[lookback-1:len(samples)+lookback-1].values

    predictors = {}
    for i, ticker in enumerate(tickers):
        valid = ~np.isnan(y_train[:, i])
        if valid.sum() < 10:
            continue
        X_tr = features.iloc[valid]
        y_tr = y_train[valid, i]
        pred = ShapePredictor().fit(X_tr, y_tr)
        predictors[ticker] = pred

    test_samples = create_window_samples(test_ret, lookback)
    if len(test_samples) > 0:
        test_smoothed = create_multivariate_fdata(test_samples, n_basis=n_basis, smoothing_parameter=config.SMOOTHING_PENALTY)
        _, test_scores_list = fit_fpca(test_smoothed, n_components=config.FPCA_COMPONENTS, refit=False, fpca_models=fpca_models)
        test_features = extract_shape_features(test_smoothed, fpca_models, test_scores_list, include_derivatives=config.INCLUDE_DERIVATIVES)
        latest_features = test_features.iloc[-1:]

        pred_returns = {}
        for ticker, pred in predictors.items():
            try:
                pred_returns[ticker] = pred.predict(latest_features)[0]
            except:
                pred_returns[ticker] = -np.inf

        best_ticker = max(pred_returns, key=pred_returns.get)
        best_pred_return = pred_returns[best_ticker]
    else:
        best_ticker = tickers[0]
        best_pred_return = 0.0
        pred_returns = {t: 0.0 for t in tickers}

    metrics = evaluate_etf(best_ticker, test_ret) if len(test_ret) > 0 else {}
    print(f"  Selected ETF: {best_ticker}, Predicted Return: {best_pred_return*100:.2f}%")
    return {
        "ticker": best_ticker,
        "pred_return": best_pred_return,
        "all_pred_returns": pred_returns,
        "adaptive_window": lookback,
        "change_point_date": cp_date.strftime("%Y-%m-%d"),
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d") if len(test_ret) else "",
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d") if len(test_ret) else "",
    }


def run_training():
    print("Loading data...")
    df_raw = load_master_data()
    df = prepare_data(df_raw)

    all_results = {}
    for universe in ["fi", "equity", "combined"]:
        print(f"\n{'='*50}\nProcessing {universe.upper()}\n{'='*50}")
        returns = get_universe_returns(df, universe)
        if returns.empty:
            continue
        global_res = train_global(universe, returns)
        adaptive_res = train_adaptive(universe, returns)
        all_results[universe] = {"global": global_res, "adaptive": adaptive_res}
    return all_results


if __name__ == "__main__":
    output = run_training()
    if config.HF_TOKEN:
        push_daily_result(output)
    else:
        print("HF_TOKEN not set.")
