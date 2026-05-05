"""
Global (with CV window selection), Daily, and Adaptive Window training.

FIXES applied:
  Bug 1 — Target leakage: shift(-1) alignment was off by one; targets now index
           into the day AFTER the window, not the last day inside it.
  Bug 2 — Log-return compounding: evaluate_etf now uses np.exp(ret)-1 for
           cumulative/Sharpe calculations instead of (1+log_ret).cumprod().
  Bug 3 — FPCA discontinuity: val/test windows are now sliced from the FULL
           returns DataFrame so they are proper continuations of training history,
           not isolated slices that break the FPCA shape space.
  Bug 5 — Wrong ticker mapping: predictors dict keyed by ticker; predict() called
           per-ticker and result stored by ticker name (not feature-column index).
  Bug 6 — MSE as window criterion: replaced with Information Coefficient
           (Spearman rank-correlation of predicted vs actual), which measures
           directional accuracy — what actually drives trading P&L.
  Bug 7 — No hold-out test in train_daily: 80/10/10 split added; metrics reported
           on the held-out test slice.
  Bug 8 — Overfitting: ShapePredictor now passes ticker name at fit time (model.py
           already fixed); window selection uses IC not MSE.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import config
from data_manager import load_master_data, prepare_data, get_universe_returns
from fda_processor import create_multivariate_fdata, fit_fpca, extract_shape_features
from model import ShapePredictor
from change_point_detector import universe_adaptive_start_date
from push_results import push_daily_result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_etf(ticker: str, returns: pd.DataFrame) -> dict:
    """
    FIX Bug 2: log returns must be exponentiated before compounding.
    (1 + log_ret).cumprod() is wrong; np.exp(log_ret).cumprod() is correct.
    """
    col = f"{ticker}_ret"
    if col not in returns.columns:
        return {}
    ret_series = returns[col].dropna()
    if len(ret_series) < 5:
        return {}

    # Annualised stats using log returns (correct for log-normal prices)
    ann_return = ret_series.mean() * config.TRADING_DAYS_PER_YEAR
    ann_vol    = ret_series.std()  * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe     = ann_return / ann_vol if ann_vol > 0 else 0.0

    # FIX: compound log returns correctly
    cum        = np.exp(ret_series).cumprod()
    rolling_max = cum.expanding().max()
    drawdown    = (cum - rolling_max) / rolling_max
    max_dd      = drawdown.min()
    hit_rate    = (ret_series > 0).mean()
    cum_return  = cum.iloc[-1] - 1.0   # total simple return

    return {
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "max_dd": max_dd, "hit_rate": hit_rate, "cum_return": cum_return,
        "n_days": len(ret_series),
    }


def create_window_samples(returns: pd.DataFrame, window_size: int) -> np.ndarray:
    """
    Returns array of shape (n_samples, n_features, window_size).
    Sample i covers returns[i : i+window_size].
    """
    data    = returns.values
    samples = [data[i : i + window_size].T for i in range(len(data) - window_size + 1)]
    return np.array(samples)


def _build_targets(returns: pd.DataFrame, window: int, n_samples: int) -> np.ndarray:
    """
    FIX Bug 1: correct target alignment.

    Sample i covers days [i, i+window-1].  The forward return to predict is
    the return on day i+window — i.e. the first day AFTER the window.

    Original code:
        y = returns.shift(-1).iloc[window-1 : window-1+n_samples]
      → shift(-1) on day (window-1+i) gives day (window+i-1)'s return,
        which is the LAST day inside the window.  Pure in-sample leakage.

    Fixed code:
        y = returns.iloc[window : window+n_samples]
      → day (window+i) is the first day outside sample i's window.
    """
    return returns.iloc[window : window + n_samples].values


def _information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    FIX Bug 6: Spearman rank-correlation (IC) as window-selection criterion.
    IC > 0 means predictions are positively correlated with outcomes in rank order.
    This is far more stable than MSE on near-zero daily returns.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 5:
        return -1.0
    ic, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(ic) if not np.isnan(ic) else -1.0


def _fit_and_build_features(ret_slice: pd.DataFrame, window: int, n_basis: int,
                             fpca_models=None, refit: bool = True):
    """
    Build window samples → smooth → FPCA → feature DataFrame.
    Returns (features_df, fpca_models, n_samples).
    """
    samples  = create_window_samples(ret_slice, window)
    n_samples = len(samples)
    if n_samples == 0:
        return None, fpca_models, 0

    smoothed = create_multivariate_fdata(
        samples, n_basis=n_basis,
        smoothing_parameter=config.SMOOTHING_PENALTY,
    )
    fpca_models, scores_list = fit_fpca(
        smoothed, n_components=config.FPCA_COMPONENTS,
        refit=refit, fpca_models=fpca_models,
    )
    features = extract_shape_features(
        smoothed, fpca_models, scores_list,
        include_derivatives=config.INCLUDE_DERIVATIVES,
    )
    return features, fpca_models, n_samples


def _attach_macro(features: pd.DataFrame, ret_slice: pd.DataFrame,
                  window: int, n_samples: int) -> pd.DataFrame:
    """Append macro columns (last value of each window) to feature DataFrame."""
    macro_cols = [c for c in config.MACRO_COLS if c in ret_slice.columns]
    if not macro_cols:
        return features
    macro_vals = (
        ret_slice[macro_cols]
        .iloc[window - 1 : window - 1 + n_samples]
        .reset_index(drop=True)
    )
    return pd.concat([features.reset_index(drop=True), macro_vals], axis=1)


def _get_latest_prediction(full_returns: pd.DataFrame, window: int,
                            fpca_models: list, predictors: dict,
                            n_basis: int) -> dict:
    """
    Run inference on the most-recent window of data.

    FIX Bug 3: use full_returns (entire history) so the latest window is a
    proper continuation of the series that FPCA was trained on, not an
    isolated out-of-distribution slice.

    FIX Bug 5: predictors is keyed by ticker name; call predict() per-ticker.
    """
    ret_cols = [c for c in full_returns.columns
                if c.endswith("_ret") and c not in config.MACRO_COLS]
    returns_only  = full_returns[ret_cols]

    if len(returns_only) < window:
        return {}

    latest_array   = returns_only.values[-window:].T[np.newaxis, :, :]  # (1, n_feat, window)
    latest_smoothed = create_multivariate_fdata(
        latest_array, n_basis=n_basis,
        smoothing_parameter=config.SMOOTHING_PENALTY,
    )
    _, latest_scores = fit_fpca(
        latest_smoothed, n_components=config.FPCA_COMPONENTS,
        refit=False, fpca_models=fpca_models,
    )
    latest_features = extract_shape_features(
        latest_smoothed, fpca_models, latest_scores,
        include_derivatives=config.INCLUDE_DERIVATIVES,
    )

    macro_cols = [c for c in config.MACRO_COLS if c in full_returns.columns]
    if macro_cols:
        macro_vals = full_returns[macro_cols].iloc[[-1]].values
        macro_df   = pd.DataFrame(macro_vals, columns=macro_cols,
                                  index=latest_features.index)
        latest_features = pd.concat([latest_features, macro_df], axis=1)

    # FIX Bug 5: each predictor knows its own ticker; call predict() per ticker
    pred_returns = {}
    for ticker, pred in predictors.items():
        try:
            pred_returns[ticker] = float(pred.predict(latest_features)[0])
        except Exception:
            pred_returns[ticker] = 0.0
    return pred_returns


# ──────────────────────────────────────────────────────────────────────────────
# Training routines
# ──────────────────────────────────────────────────────────────────────────────

def train_global(universe: str, returns: pd.DataFrame) -> dict:
    print(f"\n--- Global Training: {universe} ---")
    tickers   = [c.replace("_ret", "") for c in returns.columns
                 if c.endswith("_ret") and c not in config.MACRO_COLS]
    total     = len(returns)
    train_end = int(total * config.TRAIN_RATIO)
    val_end   = train_end + int(total * config.VAL_RATIO)
    train_ret = returns.iloc[:train_end]
    val_ret   = returns.iloc[train_end:val_end]
    test_ret  = returns.iloc[val_end:]

    best_window = None
    best_val_ic = -np.inf
    print("  Selecting optimal window via IC cross-validation...")

    for window in config.CANDIDATE_WINDOWS:
        if len(train_ret) < window + 20 or len(val_ret) < window + 5:
            continue
        n_basis = min(15, window // config.N_BASIS_FACTOR)

        # FIX Bug 3: val samples come from the FULL returns up to val_end
        #            so windows straddle train/val boundary naturally.
        train_features, fpca_models, n_tr = _fit_and_build_features(
            train_ret, window, n_basis, refit=True)
        if train_features is None or n_tr < 10:
            continue
        train_features = _attach_macro(train_features, train_ret, window, n_tr)

        # Val: use full history up to val_end so FPCA is applied to proper continuations
        full_to_val  = returns.iloc[:val_end]
        val_features, _, n_val = _fit_and_build_features(
            full_to_val, window, n_basis, fpca_models=fpca_models, refit=False)
        # Keep only the val portion of the feature rows
        val_features = val_features.iloc[n_tr:].reset_index(drop=True)
        n_val_actual = len(val_features)
        if n_val_actual < 5:
            continue
        val_features = _attach_macro(val_features, val_ret, window, n_val_actual)

        # FIX Bug 1: correct target alignment
        y_train = _build_targets(train_ret, window, n_tr)
        y_val   = _build_targets(full_to_val, window, n_tr + n_val_actual)[n_tr:]

        # FIX Bug 6: use IC instead of MSE
        all_y_true, all_y_pred = [], []
        for i, ticker in enumerate(tickers):
            valid_tr  = ~np.isnan(y_train[:, i])
            valid_val = ~np.isnan(y_val[:, i])
            if valid_tr.sum() < 10 or valid_val.sum() < 3:
                continue
            pred = ShapePredictor()
            pred.fit(train_features.iloc[valid_tr], y_train[valid_tr, i], ticker=ticker)
            preds = pred.predict(val_features.iloc[valid_val])
            all_y_true.extend(y_val[valid_val, i].tolist())
            all_y_pred.extend(preds.tolist())

        ic = _information_coefficient(np.array(all_y_true), np.array(all_y_pred))
        print(f"    Window {window:3d} → Val IC: {ic:+.4f}")
        if ic > best_val_ic:
            best_val_ic = ic
            best_window = window

    if best_window is None:
        best_window = 60
    print(f"  Selected window: {best_window} days (IC: {best_val_ic:+.4f})")

    # Retrain on train+val, evaluate on test
    train_val_ret = returns.iloc[:val_end]
    n_basis       = min(15, best_window // config.N_BASIS_FACTOR)
    features, fpca_models, n_tv = _fit_and_build_features(
        train_val_ret, best_window, n_basis, refit=True)
    features   = _attach_macro(features, train_val_ret, best_window, n_tv)
    y_all      = _build_targets(train_val_ret, best_window, n_tv)

    predictors = {}
    for i, ticker in enumerate(tickers):
        valid = ~np.isnan(y_all[:, i])
        if valid.sum() < 10:
            continue
        pred = ShapePredictor()
        pred.fit(features.iloc[valid], y_all[valid, i], ticker=ticker)
        predictors[ticker] = pred

    pred_returns = _get_latest_prediction(returns, best_window, fpca_models, predictors, n_basis)
    if pred_returns:
        best_ticker      = max(pred_returns, key=pred_returns.get)
        best_pred_return = pred_returns[best_ticker]
    else:
        best_ticker      = tickers[0]
        best_pred_return = 0.0
        pred_returns     = {t: 0.0 for t in tickers}

    metrics = evaluate_etf(best_ticker, test_ret)
    print(f"  Selected ETF: {best_ticker}, Predicted Return: {best_pred_return * 100:.2f}%")
    return {
        "ticker": best_ticker,
        "pred_return": best_pred_return,
        "all_pred_returns": pred_returns,
        "optimal_window": best_window,
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d") if len(test_ret) else "",
        "test_end":   test_ret.index[-1].strftime("%Y-%m-%d") if len(test_ret) else "",
    }


def train_daily(universe: str, returns: pd.DataFrame) -> dict:
    """
    FIX Bug 7: added proper 80/10/10 hold-out so metrics are not optimistic.
    FIX Bug 1: correct target alignment.
    FIX Bug 6: IC-based window selection.
    """
    print(f"\n--- Daily Training (504d): {universe} ---")
    daily_ret = returns.iloc[-config.DAILY_LOOKBACK:]
    if len(daily_ret) < config.MIN_TRAIN_DAYS:
        return None

    tickers   = [c.replace("_ret", "") for c in daily_ret.columns
                 if c.endswith("_ret") and c not in config.MACRO_COLS]
    total     = len(daily_ret)
    train_end = int(total * 0.8)
    val_end   = int(total * 0.9)           # FIX Bug 7: explicit 10 % test slice
    train_ret = daily_ret.iloc[:train_end]
    val_ret   = daily_ret.iloc[train_end:val_end]
    test_ret  = daily_ret.iloc[val_end:]

    best_window = None
    best_val_ic = -np.inf

    for window in config.CANDIDATE_WINDOWS:
        if len(train_ret) < window + 20 or len(val_ret) < window + 5:
            continue
        n_basis = min(15, window // config.N_BASIS_FACTOR)

        train_features, fpca_models, n_tr = _fit_and_build_features(
            train_ret, window, n_basis, refit=True)
        if train_features is None or n_tr < 10:
            continue
        train_features = _attach_macro(train_features, train_ret, window, n_tr)

        full_to_val  = daily_ret.iloc[:val_end]
        val_features, _, _ = _fit_and_build_features(
            full_to_val, window, n_basis, fpca_models=fpca_models, refit=False)
        val_features   = val_features.iloc[n_tr:].reset_index(drop=True)
        n_val_actual   = len(val_features)
        if n_val_actual < 5:
            continue
        val_features = _attach_macro(val_features, val_ret, window, n_val_actual)

        y_train = _build_targets(train_ret, window, n_tr)
        y_val   = _build_targets(full_to_val, window, n_tr + n_val_actual)[n_tr:]

        all_y_true, all_y_pred = [], []
        for i, ticker in enumerate(tickers):
            valid_tr  = ~np.isnan(y_train[:, i])
            valid_val = ~np.isnan(y_val[:, i])
            if valid_tr.sum() < 10 or valid_val.sum() < 3:
                continue
            pred = ShapePredictor()
            pred.fit(train_features.iloc[valid_tr], y_train[valid_tr, i], ticker=ticker)
            preds = pred.predict(val_features.iloc[valid_val])
            all_y_true.extend(y_val[valid_val, i].tolist())
            all_y_pred.extend(preds.tolist())

        ic = _information_coefficient(np.array(all_y_true), np.array(all_y_pred))
        if ic > best_val_ic:
            best_val_ic = ic
            best_window = window

    if best_window is None:
        best_window = 60
    print(f"  Selected window: {best_window} days (IC: {best_val_ic:+.4f})")

    # Retrain on train+val
    train_val_ret = daily_ret.iloc[:val_end]
    n_basis       = min(15, best_window // config.N_BASIS_FACTOR)
    features, fpca_models, n_tv = _fit_and_build_features(
        train_val_ret, best_window, n_basis, refit=True)
    features = _attach_macro(features, train_val_ret, best_window, n_tv)
    y_all    = _build_targets(train_val_ret, best_window, n_tv)

    predictors = {}
    for i, ticker in enumerate(tickers):
        valid = ~np.isnan(y_all[:, i])
        if valid.sum() < 10:
            continue
        pred = ShapePredictor()
        pred.fit(features.iloc[valid], y_all[valid, i], ticker=ticker)
        predictors[ticker] = pred

    pred_returns = _get_latest_prediction(daily_ret, best_window, fpca_models, predictors, n_basis)
    if pred_returns:
        best_ticker      = max(pred_returns, key=pred_returns.get)
        best_pred_return = pred_returns[best_ticker]
    else:
        best_ticker      = tickers[0]
        best_pred_return = 0.0
        pred_returns     = {t: 0.0 for t in tickers}

    metrics = evaluate_etf(best_ticker, test_ret)
    print(f"  Daily {universe}: {best_ticker}, Return: {best_pred_return * 100:.2f}%")
    return {
        "ticker": best_ticker,
        "pred_return": best_pred_return,
        "all_pred_returns": pred_returns,
        "optimal_window": best_window,
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d") if len(test_ret) else "",
        "test_end":   test_ret.index[-1].strftime("%Y-%m-%d") if len(test_ret) else "",
    }


def train_adaptive(universe: str, returns: pd.DataFrame) -> dict:
    print(f"\n--- Adaptive Training: {universe} ---")
    tickers  = [c.replace("_ret", "") for c in returns.columns
                if c.endswith("_ret") and c not in config.MACRO_COLS]
    cp_date  = universe_adaptive_start_date(returns)
    print(f"  Adaptive window starts: {cp_date.date()}")

    end_date = returns.index[-1] - pd.Timedelta(days=config.MIN_TEST_DAYS)
    if end_date <= cp_date:
        end_date = returns.index[-1] - pd.Timedelta(days=10)

    train_mask = (returns.index >= cp_date) & (returns.index <= end_date)
    train_ret  = returns.loc[train_mask]
    test_ret   = returns.loc[returns.index > end_date]

    if len(train_ret) < config.MIN_TRAIN_DAYS:
        print("  Insufficient training days. Falling back to global.")
        return train_global(universe, returns)

    lookback = min(config.ADAPTIVE_MAX_LOOKBACK, (returns.index[-1] - cp_date).days)
    lookback = max(lookback, 20)
    n_basis  = min(15, lookback // config.N_BASIS_FACTOR)

    features, fpca_models, n_tr = _fit_and_build_features(
        train_ret, lookback, n_basis, refit=True)
    if features is None:
        return train_global(universe, returns)
    features = _attach_macro(features, train_ret, lookback, n_tr)

    # FIX Bug 1: correct target alignment
    y_train = _build_targets(train_ret, lookback, n_tr)

    predictors = {}
    for i, ticker in enumerate(tickers):
        valid = ~np.isnan(y_train[:, i])
        if valid.sum() < 10:
            continue
        pred = ShapePredictor()
        pred.fit(features.iloc[valid], y_train[valid, i], ticker=ticker)
        predictors[ticker] = pred

    pred_returns = _get_latest_prediction(returns, lookback, fpca_models, predictors, n_basis)
    if pred_returns:
        best_ticker      = max(pred_returns, key=pred_returns.get)
        best_pred_return = pred_returns[best_ticker]
    else:
        best_ticker      = tickers[0]
        best_pred_return = 0.0
        pred_returns     = {t: 0.0 for t in tickers}

    metrics = evaluate_etf(best_ticker, test_ret)
    print(f"  Selected ETF: {best_ticker}, Predicted Return: {best_pred_return * 100:.2f}%")
    return {
        "ticker": best_ticker,
        "pred_return": best_pred_return,
        "all_pred_returns": pred_returns,
        "optimal_window": lookback,
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d") if len(test_ret) else "",
        "test_end":   test_ret.index[-1].strftime("%Y-%m-%d") if len(test_ret) else "",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set")
        return

    df = load_master_data()
    df = prepare_data(df)

    results = {}
    for universe_id, universe_name in [("fi", "FI"), ("equity", "Equity"), ("combined", "Combined")]:
        returns = get_universe_returns(df, universe_id)
        if returns.empty:
            continue
        results[universe_id] = {
            "global":   train_global(universe_name, returns),
            "daily":    train_daily(universe_name, returns),
            "adaptive": train_adaptive(universe_name, returns),
        }

    push_daily_result(results)
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    main()
