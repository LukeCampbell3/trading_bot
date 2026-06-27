# adaptive_strategy_live.py – robust, tunable version
import argparse, sys, time, requests, joblib, numpy as np, cupy as cp, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from tensorflow.keras.models import load_model

API_KEY = ""                       # ← Polygon key
TICKER  = "TSLA"
MODEL_P = Path("HFT/model/instinct_model.keras")
SCALER_P= Path("HFT/model/scaler.pkl")

# ─────────────────────────────────────────────────────
@dataclass
class HParams:
    lookback:      int   = 100
    vol_win:       int   = 10
    dz_avg_win:    int   = 5
    trend_len:     int   = 20
    buy_thresh:    float = 0.00024
    trend_min:     float = -0.20
    pos_frac:      float = 0.10      # 10 % equity risked per trade
    stop_pct:      float = 0.02
    take_pct:      float = 0.002
    cooldown:      int   = 2
    print_every:   int   = 200

HP = HParams()                      # default hyper-params
# ─────────────────────────────────────────────────────

def fetch_polygon(tkr, start, end, limit=3000):
    url = (f"https://api.polygon.io/v2/aggs/ticker/{tkr}/range/1/minute/"
           f"{start}/{end}?adjusted=true&sort=asc&limit={limit}&apiKey={API_KEY}")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()
    if "results" not in j: raise RuntimeError(j)
    df = pd.DataFrame(j["results"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={"c":"price", "v":"volume"})
    return df[["timestamp","price","volume"]]

class Strategy:
    FEATS = ["z","dz","avg_dz","ma_slope","vol_surge","dev_score","trend"]
    def __init__(self,hp: HParams):
        self.hp=hp
        self.model  = load_model(MODEL_P)
        self.scaler = joblib.load(SCALER_P)
        if self.scaler.center_.shape[0]!=len(self.FEATS):
            raise ValueError("Scaler dimension mismatch.")
    # ───────────────────────────────────────────────
    def features(self, p, v, i):
        h=self.hp
        win_p, win_v = cp.asarray(p[i-h.lookback:i]), cp.asarray(v[i-h.lookback:i])
        vwap = (win_p*win_v).sum()/win_v.sum(); std=win_p.std()
        z  = ((win_p[-1]-vwap)/std).item() if std else 0
        dz = z-((win_p[-2]-vwap)/std).item() if std else 0
        ma_s=win_p[-20:].mean(); ma_p=win_p[-30:-10].mean(); ma_sl=(ma_s-ma_p).item()
        vol_s=(v[i]/v[i-h.vol_win:i].mean()-1) if i>=h.vol_win else 0
        dev_s=((vwap-p[i])/std).item() if std else 0
        trend=np.polyfit(range(h.trend_len),p[i-h.trend_len:i],1)[0]
        return np.array([z,dz,cp.mean(cp.asarray([dz]*(h.dz_avg_win))).item(),ma_sl,vol_s,dev_s,trend],dtype=np.float32)
    # ───────────────────────────────────────────────
    def run(self, df):
        hp=self.hp; p=df.price.values.astype(np.float32); v=df.volume.values
        pos=0; cash=10_000; last_buy=0; cd=0; log=[]
        feats_list=[]
        for i in range(hp.lookback,len(p)):
            feats = self.features(p,v,i); feats_list.append(feats)
        X = self.scaler.transform(np.vstack(feats_list))
        preds = self.model.predict(X,verbose=0).ravel()
        idx=0
        for i in range(hp.lookback,len(p)):
            price, ts = p[i], df.timestamp.iat[i]
            pnl = preds[idx]; idx+=1
            if (idx%hp.print_every)==0:
                print(ts, "pnl",f"{pnl:.4f}","pos",pos)
            if cd: cd-=1
            # enter
            if not pos and pnl>hp.buy_thresh and feats_list[i-hp.lookback][-1]>hp.trend_min and not cd:
                confidence = np.clip(pnl / 0.001, 15.0, 30.0)
                qty = max(1, int((cash * hp.pos_frac * confidence) // price))

                #qty=max(1,int((cash*hp.pos_frac)//price)); 
                pos=qty; cash-=qty*price; last_buy=price           
                log.append((ts,"BUY",cash+pos*price,price)); continue   
            # exit
            if pos:
                pct=(price-last_buy)/last_buy
                if pct<-hp.stop_pct or pct>hp.take_pct:
                    cash+=pos*price; pos=0; cd=hp.cooldown
                    log.append((ts,"EXIT",cash,price)); continue
            log.append((ts,"SIT",cash+pos*price,price))
        return pd.DataFrame(log,columns=["timestamp","action","equity","price"])

# ───── Optional: hyper-parameter tuning via Optuna ──────
def tune(df, trials=30):
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Skipping tune."); return None
    def objective(trial):
        hp=HParams(
            buy_thresh=trial.suggest_float("buy_thresh",0.001,0.01),
            trend_min =trial.suggest_float("trend_min",-0.3,0.1),
            pos_frac  =trial.suggest_float("pos_frac",0.05,0.3),
            stop_pct  =trial.suggest_float("stop_pct",0.01,0.05),
            take_pct  =trial.suggest_float("take_pct",0.001,0.01),
        )
        eq=Strategy(hp).run(df).equity.iloc[-1]
        return -eq   # minimise negative equity → maximise equity
    study=optuna.create_study(direction="minimize")
    study.optimize(objective,n_trials=trials,show_progress_bar=False)
    print("Best params",study.best_params); return study.best_params

# ----------------- CLI -----------------
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--tune",action="store_true",help="Run Optuna tune (30 trials)")
    args=parser.parse_args()

    st,ed="2025-06-02","2025-06-06"
    data=fetch_polygon(TICKER,st,ed)

    if args.tune and "optuna" in sys.modules:
        best=tune(data); HP.__dict__.update(best)

    strat=Strategy(HP); result=strat.run(data)
    print(result.head())
    plt.plot(result.timestamp,result.equity,label="Equity")
    plt.legend(); plt.tight_layout(); plt.show()
