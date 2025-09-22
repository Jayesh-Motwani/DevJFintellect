# pip install yfinance pandas numpy
import yfinance as yf, pandas as pd, numpy as np

def fetch_prices(ticker, start):
    for i in range(3):
        try:
            df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
            if "Close" in df:
                return df["Close"].to_frame(ticker)
            return df.rename(columns={"close":"Close"})[["Close"]].rename(columns={"Close":ticker})
        except Exception as e:
            if i == 2: raise
            time.sleep(1 + i)
    raise RuntimeError("unreachable")

def download_sleeves(ticker_map, start="2014-01-01"):
    frames = []
    for sleeve, tkr in ticker_map.items():
        s = fetch_prices(tkr, start)
        s.columns = [sleeve]
        frames.append(s)
    # outer-join then align to common period where all exist
    prices = pd.concat(frames, axis=1)
    # trim to common availability (drop rows with any NaN at the ends)
    first_idx = max(prices[col].first_valid_index() for col in prices.columns)
    last_idx  = min(prices[col].last_valid_index()  for col in prices.columns)
    prices = prices.loc[first_idx:last_idx].dropna(how="any")
    return prices
def cagr(curve, periods_per_year=12):
    n_years = (curve.index[-1] - curve.index[0]).days/365.25
    return curve.iloc[-1]**(1/n_years)-1

def max_drawdown(curve):
    peak = curve.cummax()
    dd = curve/peak - 1.0
    return dd.min()
def time_to_recover(curve):
    # Longest number of months from any peak to when it’s reattained
    peaks = curve.cummax()
    longest = 0
    start_peak = None
    for i in range(1, len(curve)):
        if curve.iloc[i] < peaks.iloc[i-1]:
            # we are below a peak – start timing if not already
            if start_peak is None:
                start_peak = peaks.iloc[i-1]
                start_date = curve.index[i-1]
        else:
            # we recovered (>= previous peak)
            if start_peak is not None:
                months = (curve.index[i] - start_date).days // 30
                longest = max(longest, months)
                start_peak = None
    return longest if longest > 0 else None

if __name__=="__main__":
    tickers = {
    "equity": "NIFTYBEES.NS",     # Nifty 50
    "bonds":  "NETFLTGILT.NS",    # 8-13y G-Sec (long gilt)
    "cash":   "LIQUIDBEES.NS"     # cash proxy
    }
    weights = {"equity":0.60, "bonds":0.35, "cash":0.05}  # e.g., Balanced baseline

    prices = yf.download(list(tickers.values()), start="2007-01-01", auto_adjust=True)["Close"].dropna()
    prices.columns = {v:k for k,v in tickers.items()}  # rename to sleeves

# Monthly returns (use last business day)
    mclose = prices.resample("M").last()
    rets   = mclose.pct_change().dropna()

# Monthly rebalance to target weights (constant-mix)
    port_rets = (rets * pd.Series(weights)).sum(axis=1)

# Equity curve (₹1 start)
    curve = (1 + port_rets).cumprod()
