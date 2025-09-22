import json, requests 
from jsonschema import validate, ValidationError
import yfinance as yf, pandas as pd, numpy as np
import matplotlib.pyplot as plt 
from backtest import cagr, max_drawdown, time_to_recover, download_sleeves

SCHEMA = {
  "type":"object",
  "properties":{
    "goal":{"type":"string"},
    "timeline_years":{"type":"number"},
    "loss_aversion":{"type":"string","enum":["very_low","low","moderate","high","very_high"]},
    "liquidity_need":{"type":"string","enum":["low","moderate","high"]},
    "income_stability":{"type":"string","enum":["stable","variable","unstable"]},
    "knowledge_level":{"type":"string","enum":["novice","intermediate","advanced"]},
    "notes":{"type":"string"},
    "confidences":{
      "type":"object",
      "properties":{
        "timeline_years":{"type":"number"},
        "loss_aversion":{"type":"number"},
        "liquidity_need":{"type":"number"}
      }
    }
  },
  "required":["goal","timeline_years","loss_aversion","liquidity_need",
              "income_stability","knowledge_level"]
}


prompt = f"""
Return ONLY valid JSON. JSON Schema:
{json.dumps(SCHEMA)}

Conversation:
- If your portfolio dropped 20% in a year, what would you do?
  -> I’d hold but feel stressed.
- What excites you more: steady growth or high gains with volatility?
  -> Prefer steady growth, some risk okay.
- Any large cash needs next 2–3 years?
  -> Down payment in ~3 years.

Now produce the JSON object (no prose, no extra keys).
"""

def call_ollama(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "risk-profiler",
        "prompt": prompt,
        "format": "json",
        "options": { "temperature": 0.2 }
    }, timeout=120,stream=True)
    r.raise_for_status()
    # the /generate endpoint streams; concatenate 'response' chunks
    data = ""
    for line in r.iter_lines():
        if not line: continue
        obj = json.loads(line.decode())
        data += obj.get("response","")
        if obj.get("done"): break
    return data

raw = call_ollama(prompt)

def align_weights(w, cols):
    return pd.Series(w, dtype=float).reindex(cols).fillna(0.0)



def enum_map_loss(s):
    order = ["very_low","low","moderate","high","very_high"]
    return order.index(s) / (len(order)-1)  # 0..1 (higher = more loss averse)

def map_liq(s): return {"low":0.0,"moderate":0.5,"high":1.0}[s]
def map_income(s): return {"stable":0.0,"variable":0.5,"unstable":1.0}[s]
def map_knowledge(s): return {"novice":1.0,"intermediate":0.5,"advanced":0.0}[s]  # caution proxy
def map_horizon(y): return max(0.0, min(y/30.0, 1.0))

def composite(o):
    loss = enum_map_loss(o["loss_aversion"])
    liq  = map_liq(o["liquidity_need"])
    inc  = map_income(o["income_stability"])
    know = map_knowledge(o["knowledge_level"])
    time = map_horizon(o["timeline_years"])
    score = 100 * (0.35*(1-loss) + 0.20*(1-liq) + 0.20*(1-inc) + 0.15*(time) + 0.10*(1-know))
    if score < 35: label = "Cautious Explorer"
    elif score < 65: label = "Balanced Builder"
    else: label = "Ambitious Growth-Seeker"
    return round(score,1), label
POLICY = {
  "Cautious Explorer": {
    "baseline":  {"equity": 30, "bonds": 60, "cash": 10},
    "defensive": {"equity": 20, "bonds": 70, "cash": 10},
    "aggressive":{"equity": 40, "bonds": 50, "cash": 10},
  },
  "Balanced Builder": {
    "baseline":  {"equity": 60, "bonds": 35, "cash": 5},
    "defensive": {"equity": 50, "bonds": 45, "cash": 5},
    "aggressive":{"equity": 70, "bonds": 25, "cash": 5},
  },
  "Ambitious Growth-Seeker": {
    "baseline":  {"equity": 80, "bonds": 15, "cash": 5},
    "defensive": {"equity": 70, "bonds": 25, "cash": 5},
    "aggressive":{"equity": 90, "bonds": 5,  "cash": 5},
  }
}
def choose_weights(label:str, variant:str, axes:dict):
    # 1) start from policy (percent → weights)
    w = {k: v/100 for k,v in POLICY[label][variant].items()}

    # 2) guardrail nudges from axes
    if axes["liquidity"] >= 0.75:     # high liquidity need
        w["cash"]  = max(w["cash"], 0.10)
    if axes["loss_aversion"] >= 0.75: # very high loss aversion
        w["bonds"] = max(w["bonds"], 1 - w["cash"] - 0.50)  # cap equity at 50%

    # 3) renormalize
    s = sum(w.values())
    w = {k: v/s for k,v in w.items()}
    return w

def explain_mix(name, m):
    # m = {"CAGR_%":10.87,"Vol_ann_%":9.63,"MaxDD_%":-16.61,"Worst_12m_%":-10.16,"Recovery_m":12}
    risk = ("low" if m["Vol_ann_%"] < 7 else
            "moderate" if m["Vol_ann_%"] < 12 else
            "high")
    dd_severity = ("mild" if m["MaxDD_%"] > -12 else
                   "notable" if m["MaxDD_%"] > -20 else
                   "deep")
    rec = (f"About {m['Recovery_m']} months to recover from the worst dip."
           if m["Recovery_m"] else "No extended recovery periods observed.")

    return (
        f"**{name}**: Grew at ~{m['CAGR_%']:.1f}% per year with {risk} volatility "
        f"(~{m['Vol_ann_%']:.1f}%). Max drawdown was {m['MaxDD_%']:.1f}% ({dd_severity}). "
        f"Worst 12-month stretch was {m['Worst_12m_%']:.1f}%. {rec}"
    )

def compare_sentence(a_name, a, b_name, b):
    # Emphasize trade-offs
    d_cagr = a["CAGR_%"] - b["CAGR_%"]
    d_vol  = a["Vol_ann_%"] - b["Vol_ann_%"]
    d_dd   = a["MaxDD_%"] - b["MaxDD_%"]  # less negative = shallower
    bits = []
    if abs(d_cagr) >= 0.5: bits.append(f"{a_name} has ~{d_cagr:+.1f}pp higher CAGR")
    if abs(d_vol)  >= 0.5: bits.append(f"{d_vol:+.1f}pp change in vol")
    if abs(d_dd)   >= 2.0: bits.append(f"{'shallower' if d_dd>0 else 'deeper'} max drawdown by {abs(d_dd):.1f}pp")
    return f"{a_name} vs {b_name}: " + (", ".join(bits) if bits else "similar risk/return profile.") + "."
def drawdown(curve: pd.Series) -> pd.Series:
    return curve / curve.cummax() - 1.0
if __name__=="__main__":
    raw = call_ollama(prompt)
    try:
        obj = json.loads(raw)
        validate(instance=obj, schema=SCHEMA)
    except (json.JSONDecodeError, ValidationError) as e:
        raw = call_ollama(prompt + f"\nPrevious output failed schema validation: {e}. Return ONLY corrected JSON.")
        obj = json.loads(raw)
        validate(instance=obj, schema=SCHEMA)

    print("LLM JSON:", obj)

    score, label = composite(obj)
    print("Score/Label:", score, label)

    
    axes = {
        "time_horizon": map_horizon(obj["timeline_years"]),
        "loss_aversion": enum_map_loss(obj["loss_aversion"]),
        "liquidity": map_liq(obj["liquidity_need"]),
        "income_stability": map_income(obj["income_stability"]),
        "knowledge_caution": map_knowledge(obj["knowledge_level"]),
    }

    # Get user weights (baseline variant to start)
    w_user = choose_weights(label, "baseline", axes)
    print("User weights:", w_user)

    # Data
    tickers = {
        "equity": "NIFTYBEES.NS",
        "bonds":  "NETFLTGILT.NS",
        "cash":   "LIQUIDBEES.NS",
       
    }
    prices = download_sleeves(tickers, start="2014-01-01")
    mclose = prices.resample("ME").last()
    rets = mclose.pct_change().dropna()
    if rets.empty:
       raise ValueError("Empty returns after alignment-> check ticker ka start dates.")

    compare = {
        "Your Mix": w_user,
        "Defensive": choose_weights(label,"defensive",axes),
        "Aggressive": choose_weights(label,"aggressive",axes),
        "60/40": {"equity":0.60,"bonds":0.40,"cash":0.0},
        "All Equity": {"equity":1.0,"bonds":0.0,"cash":0.0}
    }

    # Backtest each strategy
    curves={}
    for name, wts in compare.items():
        wts_aligned = align_weights(wts, rets.columns)
        port_rets = rets.dot(wts_aligned)
        curve = (1 + port_rets).cumprod()
        curves[name]=curve
        # plt.figure(figsize=(10,4)) 
        # plt.plot(curve.index, curve.values) 
        # plt.title("Investment Growth (₹1 start)") 
        # plt.xlabel("Month") 
        # plt.ylabel("Portfolio Value (₹)") 
        # plt.tight_layout() 
        # plt.show()
        stats = {
            "CAGR_%": round(cagr(curve)*100,2),
            "Vol_ann_%": round(port_rets.std()*np.sqrt(12)*100,2),
            "MaxDD_%": round(max_drawdown(curve)*100,2),
            "Worst_12m_%": round(port_rets.rolling(12).apply(lambda x: np.prod(1+x)-1).min()*100,2),
            "Recovery_m": time_to_recover(curve)
        }

        text=explain_mix(name,stats)
        print(text)
        print(f"\n{name} stats:", stats)
    plt.figure(figsize=(11,4))
    for name, curve in curves.items():
        plt.plot(curve.index, curve.values, label=name)
    plt.title("Investment Growth (₹1 start)")
    plt.xlabel("Month"); plt.ylabel("Portfolio Value (₹)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 2) Drawdown plot ---
    plt.figure(figsize=(11,3.5))
    for name, curve in curves.items():
        dd = drawdown(curve)
        plt.plot(dd.index, (dd.values * 100), label=name)
    plt.title("Drawdown (%)")
    plt.xlabel("Month"); plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout() 
    plt.show()