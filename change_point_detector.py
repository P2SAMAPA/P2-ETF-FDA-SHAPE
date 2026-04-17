import numpy as np
import pandas as pd
import ruptures as rpt
import config

def detect_change_points_single(series: pd.Series) -> list:
    if len(series) < config.MIN_TRAIN_DAYS:
        return []
    values = series.values.reshape(-1, 1)
    algo = rpt.Pelt(model=config.CP_MODEL, min_size=config.CP_MIN_DAYS_BETWEEN).fit(values)
    return algo.predict(pen=config.CP_PENALTY)[:-1]

def get_most_recent_change_point(series: pd.Series) -> pd.Timestamp:
    cp_indices = detect_change_points_single(series)
    return series.index[0] if not cp_indices else series.index[cp_indices[-1]]

def universe_adaptive_start_date(returns: pd.DataFrame) -> pd.Timestamp:
    tickers = [col.replace("_ret", "") for col in returns.columns]
    change_dates = []
    for ticker in tickers:
        col = f"{ticker}_ret"
        if col in returns.columns:
            change_dates.append(get_most_recent_change_point(returns[col]))
    if not change_dates:
        return returns.index[0]
    from collections import Counter
    date_counts = Counter(change_dates)
    threshold = int(len(tickers) * config.CP_CONSENSUS_FRACTION)
    sorted_dates = sorted(date_counts.keys(), reverse=True)
    for date in sorted_dates:
        if date_counts[date] >= threshold:
            return date
    return date_counts.most_common(1)[0][0]
