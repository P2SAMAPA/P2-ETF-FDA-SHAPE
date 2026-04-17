import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import pandas as pd

def get_us_calendar():
    return mcal.get_calendar("NYSE")

def next_trading_day(date: datetime = None) -> datetime:
    if date is None:
        date = datetime.utcnow()
    nyse = get_us_calendar()
    ts = pd.Timestamp(date)
    start = ts - pd.Timedelta(days=5)
    end = ts + pd.Timedelta(days=10)
    schedule = nyse.schedule(start_date=start, end_date=end)
    if schedule.index.tz is not None:
        ts = ts.tz_localize('UTC') if ts.tz is None else ts.tz_convert('UTC')
    else:
        ts = ts.tz_localize(None) if ts.tz is not None else ts
    future = schedule.index[schedule.index > ts]
    if len(future) > 0:
        return future[0].to_pydatetime()
    return next_trading_day(date + timedelta(days=1))

def is_trading_day(date: datetime) -> bool:
    nyse = get_us_calendar()
    ts = pd.Timestamp(date)
    start = ts - pd.Timedelta(days=5)
    end = ts + pd.Timedelta(days=5)
    schedule = nyse.schedule(start_date=start, end_date=end)
    if schedule.index.tz is not None:
        ts = ts.tz_localize('UTC') if ts.tz is None else ts.tz_convert('UTC')
    else:
        ts = ts.tz_localize(None) if ts.tz is not None else ts
    return ts in schedule.index
