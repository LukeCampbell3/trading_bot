# ============================================
#  adaptive_minute_hybrid.py   (swing version)
# ============================================
import os, time, threading, csv, queue, math, logging, requests
from collections import deque
import numpy as np, cupy as cp, pandas as pd, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
import joblib

# ---------- CONFIG ----------
API_KEY          = os.getenv("POLYGON_API_KEY", "")

TICKER           = "TSLA"
START_DATE       = "2024-11-05"
END_DATE         = "2025-03-08"
BAR_LIMIT        = 10_000          # pull more minutes for swing

MODEL_PATH       = "C:/Users/jcthi/Code/HFT/model/instinct_model.keras"
SCALER_PATH      = "C:/Users/jcthi/Code/HFT/model/scaler.pkl"

# swing‑trading risk settings
MAX_LEVERAGE        = 2.0
BASE_POS_FRACT      = 0.10
MAX_HOLD_TICKS      = 300           # ≈5 h on 1‑min bars
TRAIL_STOP_PCT      = 0.60          # give back 40 % of peak gain
RISK_STOP_PCT       = 0.05          # hard −5 % stop
DRAW_DOWN_CAP_PCT   = 0.25          # flatten if equity‑DD >25 %
COOLDOWN_ENTER      = 5             # min bars between new entries

# transaction cost
FEE_PCT          = 0.0005           # 5 bps round‑trip (commission+slip)

# fine‑tune
FINE_TUNE_BATCH  = 128
EARLY_STOP_ROUNDS= 6
DRIFT_TOL        = 0.002
LOG_FILE         = "trade_log.csv"

# ---------- logging ----------
logging.basicConfig(level=logging.INFO,
      format="%(asctime)s %(levelname)s | %(message)s",
      handlers=[logging.StreamHandler(),
                logging.FileHandler("runtime.log", mode="w")])

# ---------- Polygon fetch ----------
def fetch_polygon_data(ticker, start, end, bar_limit=10_000):
    url=(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/"
         f"{start}/{end}?adjusted=true&sort=asc&limit={bar_limit}&apiKey={API_KEY}")
    j=requests.get(url,timeout=10).json()
    if "results" not in j: raise RuntimeError(f"Polygon error: {j}")
    df=pd.DataFrame(j["results"])
    df["timestamp"]=pd.to_datetime(df["t"],unit="ms")
    df.rename(columns={"c":"price","v":"volume"},inplace=True)
    return df[["timestamp","price","volume"]]

# ---------- model objects ----------
scaler      = joblib.load(SCALER_PATH)
model_live  = load_model(MODEL_PATH)
model_tuner = clone_model(model_live); model_tuner.set_weights(model_live.get_weights())

err_q=queue.Queue(maxsize=4096); queue_lock=threading.Lock()

def fine_tune_daemon():
    global model_live
    patience,best,stalled=EARLY_STOP_ROUNDS,math.inf,False
    valX,valY=deque(maxlen=256),deque(maxlen=256)
    while True:
        if stalled:
            time.sleep(5)
            if len(valX)>=128 and model_live.evaluate(
                 scaler.transform(np.array(valX)),np.array(valY),verbose=0) > best+DRIFT_TOL:
                stalled=False; logging.info("Drift detected – retraining continues.")
            continue
        Xb,Yb=[],[]
        while len(Xb)<FINE_TUNE_BATCH:
            try: Xb.append(*err_q.get(timeout=2))
            except queue.Empty: break
        if not Xb: continue
        Xb=scaler.transform(np.array(Xb)); Yb=np.array(Yb)
        model_tuner.compile(optimizer="adam",loss="huber")
        model_tuner.fit(Xb,Yb,epochs=1,batch_size=32,verbose=0)
        loss=model_tuner.evaluate(Xb,Yb,verbose=0)
        valX.extend(Xb); valY.extend(Yb)
        if loss<best-1e-4:
            best,patience=loss,EARLY_STOP_ROUNDS
            with queue_lock: model_live.set_weights(model_tuner.get_weights())
        else:
            patience-=1
            if patience==0:
                stalled=True; patience=EARLY_STOP_ROUNDS; logging.info("Fine‑tune stalled.")

class SwingHybrid:
    def __init__(self,init_cap=10_000):
        self.init_cap=init_cap; self.reset()
    def reset(self):
        self.capital=self.init_cap; self.position=0
        self.last_buy=None; self.max_unreal=0; self.hold_ticks=0
        self.cooldown=0; self.equity_peak=self.init_cap
        self.sig_hist,deq=deque(maxlen=120),deque(maxlen=30); self.pnl_hist=deq
        self.tick_t,self.model_t,self.log=[],[],[]
    def adaptive_theta(self):
        if len(self.sig_hist)<80: return 0.010
        a=np.array(self.sig_hist); return a.mean()+1.2*a.std()
    def kelly_frac(self):
        if len(self.pnl_hist)<8: return 0.25
        p=sum(o>0 for o in self.pnl_hist)/len(self.pnl_hist)
        return 0.4*max(0,min(1,(p*2-1)))
    def run(self,df):
        p_arr,v_arr=df.price.values.astype(np.float32),df.volume.values.astype(np.float32)
        ts=df.timestamp.values
        for i in range(120,len(df)):
            t0=time.perf_counter()
            w_p=cp.asarray(p_arr[i-120:i]); w_v=cp.asarray(v_arr[i-120:i])
            price,stamp=float(p_arr[i]),ts[i]
            vwap=(w_p*w_v).sum()/w_v.sum(); std=cp.std(w_p)
            z=((w_p[-1]-vwap)/std).item() if std>0 else 0
            dz=z-((w_p[-2]-vwap)/std).item() if std>0 else 0
            avg_dz=float(np.mean(np.diff(p_arr[i-30:i])))
            ma50=cp.mean(w_p[-50:]); ma200=cp.mean(w_p)
            ma_slope=(ma50-ma200).item()/ma200.item()
            vol_now=float(v_arr[i]); vol_avg=float(np.mean(v_arr[i-30:i]))
            vol_surge=(vol_now-vol_avg)/vol_avg if vol_avg>0 else 0
            dev=((vwap-w_p[-1])/std).item() if std>0 else 0
            instinct=dev+1.5*avg_dz+0.8*ma_slope+0.5*vol_surge
            self.sig_hist.append(instinct)
            X=scaler.transform([[z,dz,avg_dz,ma_slope,vol_surge,dev]])
            m0=time.perf_counter()
            with queue_lock: pred=model_live.predict(X,verbose=0)[0][0]
            self.model_t.append(time.perf_counter()-m0)
            theta=self.adaptive_theta()
            buy_thr=theta*0.9 if sum(o<0 for o in list(self.pnl_hist)[-3:])>=2 else theta
            kfrac=self.kelly_frac(); pos_frac=BASE_POS_FRACT*kfrac
            action="SIT"
            if self.cooldown>0: self.cooldown-=1
            # ----- enter / add -----
            if self.position==0 and pred>buy_thr and self.cooldown==0:
                size=int((self.capital*pos_frac)//price)
                if size:
                    self.capital-=price*size*(1+FEE_PCT); self.position=size
                    self.last_buy=price; self.hold_ticks=0; self.max_unreal=0
                    self.cooldown=COOLDOWN_ENTER; action=f"BUY({size})"
            elif self.position and pred>theta and self.cooldown==0:
                add=int((self.capital*0.05)//price)
                if add:
                    self.capital-=price*add*(1+FEE_PCT); self.position+=add
                    self.cooldown=COOLDOWN_ENTER; action=f"ADD({add})"
            # ----- exit -----
            if self.position:
                self.hold_ticks+=1
                unreal=price-self.last_buy; pr=unreal/self.last_buy
                self.max_unreal=max(self.max_unreal,unreal)
                lg=abs(unreal)<=self.max_unreal*(1-TRAIL_STOP_PCT)
                stop=(pr<-RISK_STOP_PCT) or lg or self.hold_ticks>MAX_HOLD_TICKS
                if stop:
                    self.capital+=price*self.position*(1-FEE_PCT)
                    self.pnl_hist.append(pr)
                    if pr<0 and not err_q.full(): err_q.put_nowait((X[0],pr))
                    self.position=0; self.last_buy=None; self.hold_ticks=0
                    self.cooldown=COOLDOWN_ENTER; action="EXIT"
            # draw‑down
            self.equity_peak=max(self.equity_peak,self.capital)
            if self.capital<self.equity_peak*(1-DRAW_DOWN_CAP_PCT):
                self.position=0; logging.warning("DD >25 % — flattened.")
            port=self.capital+self.position*price
            self.log.append((stamp,action,port,price))
            self.tick_t.append(time.perf_counter()-t0)
        with open(LOG_FILE,"w",newline="") as f:
            csv.writer(f).writerows([["timestamp","action","portfolio","price"],*self.log])
        return pd.DataFrame(self.log,columns=["timestamp","action","portfolio","price"])

def main():
    df=fetch_polygon_data(TICKER,START_DATE,END_DATE,BAR_LIMIT)
    threading.Thread(target=fine_tune_daemon,daemon=True).start()
    strat=SwingHybrid(init_cap=1_000)
    t0=time.perf_counter(); res=strat.run(df); t1=time.perf_counter()
    print(f"\nRuntime {t1-t0:.1f}s | avg tick {np.mean(strat.tick_t)*1000:.2f} ms")
    plt.figure(figsize=(13,5)); plt.plot(res.timestamp,res.portfolio,color="navy")
    plt.title(f"{TICKER} – Swing Hybrid ($1 k)"); plt.grid(); plt.tight_layout(); plt.show()
    print(f"End equity ${res.portfolio.iloc[-1]:,.2f} | trades {len(res[res.action!='SIT'])}")

if __name__=="__main__":
    main()
