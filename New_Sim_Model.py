
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score
from xgboost import XGBRegressor
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
import joblib
pipe=joblib.load("model_and_pre_fin.pkl")
def make_ohlc_from_returns(dates, start_price, returns_log, base_vol=0.012):
    """dates: sequence of Timestamps; returns_log: array of log-returns"""
    ohlc = []
    open_p = float(start_price)
    for d, r in zip(dates, returns_log):
        close_p = open_p * np.exp(r)
        intra = abs(r) + base_vol*0.5  # wick size ~ move + base vol
        high_p = max(open_p, close_p) * (1 + np.random.uniform(0, intra))
        low_p  = min(open_p, close_p)  * (1 - np.random.uniform(0, intra))
        ohlc.append({"date": d, "open": open_p, "high": high_p, "low": low_p, "close": close_p})
        open_p = close_p
    return pd.DataFrame(ohlc)



def plot_candles(ohlc_df, title="Candlestick"):
    # ohlc_df: columns = ['date','open','high','low','close']
    ohlc_df = ohlc_df.sort_values('date').reset_index(drop=True)
    xs = np.arange(len(ohlc_df))
    w = 0.6

    fig, ax = plt.subplots(figsize=(10,4))
    for i, r in ohlc_df.iterrows():
        # wick
        ax.plot([xs[i], xs[i]], [r['low'], r['high']])
        # body
        lower = min(r['open'], r['close'])
        height = abs(r['close'] - r['open'])
        ax.add_patch(plt.Rectangle((xs[i]-w/2, lower), w, max(height, 1e-9)))
    ax.set_xticks(xs[::max(1, len(xs)//10)])
    ax.set_xticklabels(ohlc_df['date'].dt.strftime('%Y-%m-%d').iloc[::max(1, len(xs)//10)],
                       rotation=45, ha='right')
    ax.set_title(title)
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    plt.tight_layout()
    plt.show()




# --- 1) build future trading dates ---
def future_business_days(start_date, days=88):
    dts = pd.bdate_range(start_date + BDay(1), periods=days)
    return dts

# --- 2) expand a scenario into a feature panel ---
def make_future_features(
    company_meta,              # dict: {'company_id', 'sector', 'market_cap_bucket', 'company_size'}
    start_date, horizon=88,
    mode="hold",               # "hold" | "trajectory"
    controls=None,             # dict of scalars or per-day arrays
    events=None                # list of {'date':..., 'field':..., 'value':...}
):
    """Returns a DataFrame with the same feature columns your model expects."""
    dates = future_business_days(pd.Timestamp(start_date), horizon)
    n = len(dates)

    # defaults if user omits anything
    if (mode=="hold"):
        def _val(name, default):
            v = (controls or {}).get(name, default)
            if np.isscalar(v): return np.repeat(v, n)
            v = np.asarray(v)
            if len(v) != n: raise ValueError(f"{name} length {len(v)} != horizon {n}")
            return v
    else:
        def _val(name, default):
            v = (controls or {}).get(name, default)
            if np.isscalar(v): return np.repeat(v, n)
            v = np.asarray(v)
            if len(v) != n: raise ValueError(f"{name} length {len(v)} != horizon {n}")
            return v       


    # numeric features (daily scale)
    data = dict(
        overall_market_sentiment = _val("overall_market_sentiment", 0.0),
        fii_flows                 = _val("fii_flows", 0.0),         # crores per day
        dii_flows                 = _val("dii_flows", 0.0),
        global_market_cues        = _val("global_market_cues", 0.0),
        inr_usd_delta             = _val("inr_usd_delta", 0.0),     # e.g. +0.002 = +0.2%
        crude_oil_delta           = _val("crude_oil_delta", 0.0),   # e.g. -0.01 = -1%
        earnings_announcement     = _val("earnings_announcement", 0),    # 0/1
        analyst_rating_change     = _val("analyst_rating_change", 0)     # -2..+2
    )

    # categoricals (single choice each day; can be arrays)
    cat = dict(
        sector                = np.repeat(company_meta['sector'], n),
        market_cap_bucket     = np.repeat(company_meta['market_cap_bucket'], n),
        major_news            = _val("major_news", "none"),
        insider_activity      = _val("insider_activity", "none"),
        predefined_global_shock = _val("predefined_global_shock", "none"),
    )

    # optional event overrides (date-stamped mutations)
    if events:
        df_tmp = pd.DataFrame({**data, **cat}, index=dates).reset_index().rename(columns={'index':'date'})
        for ev in events:
            ix = df_tmp['date'] == pd.Timestamp(ev['date'])
            df_tmp.loc[ix, ev['field']] = ev['value']
        df_tmp['company_id'] = company_meta['company_id']
        df_tmp['company_name'] = company_meta.get('company_name', company_meta['company_id'])
        df_tmp['company_size'] = company_meta['company_size']
        return df_tmp

    # assemble
    out = pd.DataFrame({**data, **cat})
    out.insert(0, 'date', dates)
    out['company_id'] = company_meta['company_id']
    out['company_name'] = company_meta.get('company_name', company_meta['company_id'])
    out['company_size'] = company_meta['company_size']
    return out


def make_ohlc_from_returns(dates, start_price, returns_log, base_vol=0.012):
    ohlc, open_p = [], float(start_price)
    for d, r in zip(pd.to_datetime(dates), returns_log):
        close_p = open_p * np.exp(r)
        intra = abs(r) + base_vol*0.5
        high_p = max(open_p, close_p) * (1 + np.random.uniform(0, intra))
        low_p  = min(open_p, close_p)  * (1 - np.random.uniform(0, intra))
        ohlc.append({"date": d, "open": open_p, "high": high_p, "low": low_p, "close": close_p})
        open_p = close_p
    return pd.DataFrame(ohlc)

def simulate_path(pipe, company_meta, last_known_close, start_date, horizon=88, events=None, base_vol=0.012,mode="hold"):
    # Build the same columns the model expects:
    
    num_cols = [
      "overall_market_sentiment","fii_flows","dii_flows","global_market_cues",
      "inr_usd_delta","crude_oil_delta","company_size","analyst_rating_change","earnings_announcement"
    ]
    cat_cols = ["sector","market_cap_bucket","major_news","insider_activity","predefined_global_shock"]
    if (mode=="hold"):
        future_df = make_future_features(company_meta, start_date, horizon, mode="hold",controls=controls_hold, events=events)
    elif(mode=="trajectory"):
        future_df = make_future_features(company_meta, start_date, horizon, mode="trajectory",controls=controls_trajectory, events=events)
    # order columns for the pipeline
    X_fut = future_df[num_cols + cat_cols]
    # predict daily log-returns
    rhat = pipe.predict(X_fut)
    # turn into OHLC path
    ohlc = make_ohlc_from_returns(future_df['date'], last_known_close, rhat, base_vol=base_vol)
    return ohlc, future_df, rhat

company_meta = {
  "company_id": "APPL",
  "company_name": "Apple",
  "sector": "IT Services",
  "market_cap_bucket": "Large Cap",
  "company_size": 90
}

controls_hold = {
  "overall_market_sentiment": 0.4,
  "fii_flows": 800, "dii_flows": 300,
  "global_market_cues": 0.2,
  "inr_usd_delta": -0.001,         # +0.1% per day
  "crude_oil_delta": -0.004,      # -0.4% per day
  "earnings_announcement": 0,
  "analyst_rating_change": 0,
  "major_news": "none",
  "insider_activity": "none",
  "predefined_global_shock": "none"
}
n = 88
controls_trajectory = {
  "overall_market_sentiment": np.linspace(0.0, 0.6, n),  # ramps up
  "fii_flows": np.repeat(500, n),
  "global_market_cues": np.sin(np.linspace(0, 2*np.pi, n))*0.2,
  "major_news": ["none"]*n,
  "insider_activity": ["none"]*n,
  "predefined_global_shock": ["none"]*n,
  "earnings_announcement": [0]*n,
  "analyst_rating_change": [0]*n,
  "inr_usd_delta": [0.0]*n,
  "crude_oil_delta": [0.0]*n,
}



last_close   = 112.3           # from your latest historical row
start_date   = pd.Timestamp("2025-06-27")

ohlc, fut, rhat = simulate_path(pipe, company_meta, last_close, start_date, horizon=88, base_vol=0.012,mode="trajectory")
# -> plot ohlc
plot_candles(ohlc)