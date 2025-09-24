#!/usr/bin/env python3
"""
Live Stocks Paper-Trader — OP Edition (Yahoo minute bars) + PM2-friendly logging
v1.4-bmo

Additions:
  • Parallel live fetches via asyncio + threadpool (yfinance is blocking)
  • Optional market calendar (NYSE) if pandas_market_calendars installed
  • Robust fetch: retry with exponential backoff + jitter; soft cache last bar
  • Walk-forward retrain: --retrain_every_bars (0=never, N=retrain on each N-th closed bar)
  • Risk: --max_drawdown_pct, --daily_loss_limit_pct, --cooldown_bars_after_loss
  • Slippage: --slippage_bps; Cash reserve: --min_cash_reserve
  • JSONL trade log: logs/trades.jsonl (append-only, PM2-safe)
  • Model file prefix includes version tag and symbol; safe load

Everything else kept: backfill, lag stacks, horizon labels, regime filter, kelly sizing, stops, bandit/adaptive, PM2 shutdown.
"""

from __future__ import annotations
import argparse, asyncio, os, sys, warnings, signal, time, json, math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import ta  # pip install ta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import yfinance as yf

# Optional: holiday/half-day awareness (best-effort)
try:
    import pandas_market_calendars as pmc  # pip install pandas_market_calendars
    _HAS_CAL = True
    _CAL = pmc.get_calendar("XNYS")
except Exception:
    _HAS_CAL = False
    _CAL = None

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)
plt.switch_backend("Agg")

INTERVALS = {"1m":"1m","2m":"2m","5m":"5m","15m":"15m"}
VERSION_TAG = "v1_4"

# ======= PM2-friendly Logger =======
class TinyLogger:
    def __init__(self, log_path: Optional[str] = None, rollover_bytes: int = 5_000_000, backups: int = 3):
        self.log_path = log_path
        self.rollover_bytes = rollover_bytes
        self.backups = backups
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _write_file(self, line: str):
        if not self.log_path:
            return
        try:
            # poor man's size-based rollover
            if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > self.rollover_bytes:
                for i in range(self.backups, 0, -1):
                    src = f"{self.log_path}.{i}" if i > 1 else self.log_path
                    dst = f"{self.log_path}.{i+1}"
                    if os.path.exists(src):
                        try:
                            if i == self.backups:
                                os.remove(src)
                            else:
                                os.rename(src, dst)
                        except Exception:
                            pass
                try:
                    os.rename(self.log_path, f"{self.log_path}.1")
                except Exception:
                    pass
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def log(self, msg: str):
        line = msg
        print(line, flush=True)
        self._write_file(line)

LOGGER: TinyLogger = TinyLogger()  # set in main()

def L(msg: str):
    LOGGER.log(msg)

# ======= Utils / Time =======
LOCAL_TZ = ZoneInfo("America/Chicago")
ET_TZ    = ZoneInfo("America/New_York")

def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")

def fmt_local_from_utc(pd_ts: pd.Timestamp) -> str:
    if pd_ts.tzinfo is None:
        pd_ts = pd_ts.tz_localize("UTC")
    return pd_ts.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")

def ensure_dirs():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

# Better market-hours with optional holiday/half-day awareness
def is_market_open(now_utc: pd.Timestamp, rth_only: bool = True) -> bool:
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    now_et = now_utc.tz_convert(ET_TZ)

    if not rth_only:
        # 4am–8pm ET (pre/post)
        pre  = now_et.replace(hour=4,  minute=0, second=0, microsecond=0)
        post = now_et.replace(hour=20, minute=0, second=0, microsecond=0)
        return pre <= now_et <= post

    if _HAS_CAL:
        # RTH window from the calendar (handles holidays/half-days)
        sched = _CAL.schedule(start_date=now_et.date(), end_date=now_et.date())
        if sched.empty:
            return False
        open_ts  = sched.iloc[0]["market_open"].tz_convert(ET_TZ)
        close_ts = sched.iloc[0]["market_close"].tz_convert(ET_TZ)
        return open_ts <= now_et <= close_ts

    # Fallback: simple weekday 9:30–16:00 ET
    if now_et.weekday() >= 5:
        return False
    open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_et <= now_et <= close_et

# ======= Indicators / Features =======
BASE_FEATS = [
    "ret1","logret","rsi14","ema_ratio","macd","macd_sig","macd_hist",
    "vol_ratio","hl_spread","atr14"
]

def rsi(s: pd.Series, win=14) -> pd.Series:
    return ta.momentum.rsi(s, window=win)

def ema(s: pd.Series, span=12) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def macd(close: pd.Series):
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def add_features(df: pd.DataFrame, lag_k: int = 0, lag_only_feats: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    df["ret1"]   = df["close"].pct_change()
    df["logret"] = np.log(df["close"]).diff()
    df["rsi14"]  = rsi(df["close"], 14)
    df["ema12"]  = ema(df["close"], 12)
    df["ema26"]  = ema(df["close"], 26)
    df["ema_ratio"] = df["ema12"] / (df["ema26"] + 1e-9) - 1.0
    m, s, h = macd(df["close"])
    df["macd"]      = m
    df["macd_sig"]  = s
    df["macd_hist"] = h
    df["vol_ma20"]  = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_ma20"] + 1e-9)
    df["hl_spread"] = (df["high"] - df["low"]) / (df["close"].shift(1) + 1e-9)
    df["atr14"]     = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()

    if lag_k and lag_k > 0:
        if lag_only_feats is None:
            lag_only_feats = ["rsi14","ema_ratio","macd_hist","vol_ratio","ret1"]
        for feat in lag_only_feats:
            if feat in df.columns:
                for k in range(1, lag_k+1):
                    df[f"{feat}_l{k}"] = df[feat].shift(k)
    return df

def label_up(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    return (df["close"].shift(-horizon) / df["close"] - 1.0 > 0).astype(int)

def build_feature_list(lag_k: int, lag_only_feats: Optional[List[str]]) -> List[str]:
    feats = list(BASE_FEATS)
    if lag_k and lag_k > 0:
        if lag_only_feats is None:
            lag_only_feats = ["rsi14","ema_ratio","macd_hist","vol_ratio","ret1"]
        for feat in lag_only_feats:
            for k in range(1, lag_k+1):
                feats.append(f"{feat}_l{k}")
    return feats

def make_xy(df: pd.DataFrame, lag_k: int = 0, horizon: int = 1, lag_only_feats: Optional[List[str]] = None):
    df = add_features(df.copy(), lag_k=lag_k, lag_only_feats=lag_only_feats)
    if df.empty:
        return df, pd.Series(dtype=int), []
    df["y"] = label_up(df, horizon=horizon)
    feats = build_feature_list(lag_k, lag_only_feats)
    X = df[feats].astype(float)
    y = df["y"].astype(int)
    ok = np.isfinite(X).all(1) & y.notna()
    return X[ok], y[ok], feats

# ======= Yahoo fetch / Backfill (robust) =======
def _yahoo_to_df(hist: pd.DataFrame) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time"])
    df = hist.rename(columns=str.lower).copy()
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df["open_time"]  = idx
    df["close_time"] = df["open_time"]
    df = df.reset_index(drop=True)
    need = ["open","high","low","close","volume","open_time","close_time"]
    for n in need:
        if n not in df.columns:
            df[n] = np.nan
    return df[need]

def _apply_step_close_time(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty: return df
    step = {"1m":60, "2m":120, "5m":300, "15m":900, "30m":1800, "60m":3600}.get(interval, 60)
    df = df.copy()
    df["close_time"] = df["open_time"] + pd.to_timedelta(step, unit="s")
    return df

def _retry_fetch(fn, *args, **kwargs):
    max_tries = kwargs.pop("_max_tries", 4)
    base = kwargs.pop("_base_sleep", 0.8)
    for i in range(max_tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            sleep = base * (2 ** i) + random.uniform(0, 0.25)
            time.sleep(sleep)
    return pd.DataFrame()

def fetch_recent(symbol: str, interval: str) -> pd.DataFrame:
    hist = _retry_fetch(
        yf.Ticker(symbol).history,
        period="1d", interval=interval, prepost=True, auto_adjust=False,
        _max_tries=3
    )
    return _apply_step_close_time(_yahoo_to_df(hist), interval)

def fetch_backfill_chunked(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    small_intraday = {"1m","2m","5m","15m","30m"}
    now = datetime.utcnow()
    if interval in small_intraday:
        max_start = now - timedelta(days=59)
        if start < max_start:
            L(f"[{ts()}] NOTE: {symbol} {interval} backfill older than ~60d not available on Yahoo; clamping to {max_start.date()} -> {end.date()}.")
            start = max_start

    if interval in {"1m","2m"}:
        chunk_days = 6
    elif interval in {"5m","15m","30m"}:
        chunk_days = 58
    else:
        chunk_days = 179

    df_all = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        try:
            hist = _retry_fetch(
                yf.Ticker(symbol).history,
                start=cur, end=nxt, interval=interval, prepost=True, auto_adjust=False,
                _max_tries=3
            )
            part = _apply_step_close_time(_yahoo_to_df(hist), interval)
            if not part.empty:
                df_all.append(part)
        except Exception as e:
            L(f"[{ts()}] WARN: backfill chunk failed for {symbol} {interval} {cur.date()}->{nxt.date()}: {e}")
        cur = nxt + timedelta(days=1)

    if not df_all:
        L(f"[{ts()}] WARN: no {interval} backfill returned for {symbol}.")
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time"])

    df = pd.concat(df_all, ignore_index=True)
    df = df.sort_values("close_time").drop_duplicates(subset=["close_time"]).reset_index(drop=True)
    return df

# ======= Broker =======
@dataclass
class Position:
    qty: float = 0.0
    entry: float = 0.0
    entry_ct: Optional[pd.Timestamp] = None

@dataclass
class Trade:
    time: str
    symbol: str
    side: str   # BUY/SELL
    price: float
    qty: float
    fee: float
    slippage: float
    cash_after: float
    equity_after: float
    entry_price: float = 0.0
    pct_return: float = 0.0

class PaperBroker:
    def __init__(self, cash: float, fee_bps: float, slippage_bps: float, min_cash_reserve: float = 0.0):
        self.starting_cash = float(cash)
        self.cash = float(cash)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.min_cash_reserve = float(min_cash_reserve)
        self.pos: Dict[str, Position] = {}
        self.equity_history: List[Tuple[str, float]] = []
        self.trades: List[Trade] = []
        self._jsonl_path = os.path.join("logs", "trades.jsonl")

    def _fee(self, notional: float) -> float:
        return abs(notional) * (self.fee_bps / 1e4)

    def _slip_price(self, price: float, side: str) -> float:
        # simple bps-based slippage
        s = price * (self.slippage_bps / 1e4)
        return price + s if side == "BUY" else price - s

    def _append_jsonl(self, t: Trade):
        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(t.__dict__) + "\n")
        except Exception:
            pass

    def mark(self, when: str, prices: Dict[str, float]):
        eq = self.cash
        for sym, p in prices.items():
            if sym in self.pos:
                eq += self.pos[sym].qty * p
        self.equity_history.append((when, eq))

    def equity(self, prices: Dict[str, float]) -> float:
        eq = self.cash
        for sym, p in prices.items():
            pos = self.pos.get(sym)
            if pos and pos.qty:
                eq += pos.qty * p
        return eq

    def rebalance_to_target(self, when: str, symbol: str, price: float, target_frac: float, prices_snapshot: Dict[str, float]) -> Optional[Trade]:
        target_frac = float(np.clip(target_frac, 0.0, 1.0))
        if symbol not in self.pos:
            self.pos[symbol] = Position()
        pos = self.pos[symbol]

        equity_now = self.equity(prices_snapshot | {symbol: price})
        target_notional   = equity_now * target_frac
        current_notional  = pos.qty * price
        delta_notional    = target_notional - current_notional

        if abs(delta_notional) < 1e-10:
            return None

        if delta_notional > 0:
            buy_notional = min(self.cash - self.min_cash_reserve, delta_notional)
            if buy_notional <= 0:
                return None
            trade_px = self._slip_price(price, "BUY")
            qty = buy_notional / trade_px
            fee = self._fee(buy_notional)
            self.cash -= (buy_notional + fee)
            if pos.qty == 0:
                pos.entry = trade_px
                pos.entry_ct = utcnow()
            else:
                # VWAP-style update to entry if scaling in
                pos.entry = (pos.entry * pos.qty + trade_px * qty) / (pos.qty + qty)
            pos.qty += qty
            eq_after = self.cash + pos.qty * price
            t = Trade(when, symbol, "BUY", trade_px, qty, fee, trade_px - price, self.cash, eq_after, entry_price=pos.entry, pct_return=0.0)
            self.trades.append(t); self._append_jsonl(t); return t
        else:
            sell_notional = min(current_notional, -delta_notional)
            if sell_notional <= 0 or pos.qty <= 0:
                return None
            trade_px = self._slip_price(price, "SELL")
            qty = sell_notional / trade_px
            qty = min(qty, pos.qty)
            fee = self._fee(sell_notional)
            self.cash += (qty * trade_px - fee)
            pos.qty -= qty
            if pos.qty <= 1e-12:
                entry_price = pos.entry
                pct_ret = (trade_px - entry_price) / entry_price if entry_price > 0 else 0.0
                pos.qty = 0.0; pos.entry = 0.0; pos.entry_ct = None
                t = Trade(when, symbol, "SELL", trade_px, 0.0, fee, price - trade_px, self.cash, self.cash, entry_price=entry_price, pct_return=pct_ret)
            else:
                t = Trade(when, symbol, "SELL", trade_px, 0.0, fee, price - trade_px, self.cash, self.cash + pos.qty*price, entry_price=pos.entry, pct_return=0.0)
            self.trades.append(t); self._append_jsonl(t); return t

    def force_close(self, when: str, symbol: str, price: float) -> Optional[Trade]:
        pos = self.pos.get(symbol)
        if not pos or pos.qty <= 0: return None
        trade_px = self._slip_price(price, "SELL")
        notional = pos.qty * trade_px
        fee = self._fee(notional)
        entry_price = pos.entry
        pct_ret = (trade_px - entry_price) / entry_price if entry_price > 0 else 0.0
        self.cash += (notional - fee)
        pos.qty = 0.0; pos.entry = 0.0; pos.entry_ct = None
        t = Trade(when, symbol, "SELL", trade_px, 0.0, fee, price - trade_px, self.cash, self.cash, entry_price=entry_price, pct_return=pct_ret)
        self.trades.append(t); self._append_jsonl(t); return t

# ======= Persistence =======
def save_model(model, feats, path):
    try:
        joblib.dump({"model": model, "feats": feats}, path)
    except Exception as e:
        L(f"[{ts()}] WARN: could not save model to {path}: {e}")

def load_model(path):
    try:
        if os.path.exists(path):
            obj = joblib.load(path)
            return obj.get("model"), obj.get("feats")
    except Exception as e:
        L(f"[{ts()}] WARN: could not load model from {path}: {e}")
    return None, None

# ======= Ensemble =======
class EnsembleLearner:
    def __init__(self, model_dir: str, lag_k: int, horizon: int, lag_only_feats: Optional[List[str]] = None):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.models = {
            "gb": GradientBoostingClassifier(random_state=42),
            "rf": RandomForestClassifier(n_estimators=300, random_state=42),
            "et": ExtraTreesClassifier(n_estimators=400, random_state=42),
        }
        self.feats: List[str] = []
        self.cv_auc: Dict[str, float] = {k: float("nan") for k in self.models}
        self.live_score: Dict[str, float] = {k: 0.5 for k in self.models}
        self.lag_k = lag_k
        self.horizon = horizon
        self.lag_only_feats = lag_only_feats

    def _paths(self, symbol: str):
        base = os.path.join(self.model_dir, f"{VERSION_TAG}_{symbol}")
        return {k: f"{base}_{k}.pkl" for k in self.models}

    def fit(self, symbol: str, df: pd.DataFrame) -> float:
        X, y, feats = make_xy(df.copy(), lag_k=self.lag_k, horizon=self.horizon, lag_only_feats=self.lag_only_feats)
        if len(X) < 50 or y.nunique() < 2:
            for name, mdl in self.models.items():
                try:
                    if len(X) >= 2:
                        mdl.fit(X, y)
                    self.cv_auc[name] = np.nan
                except Exception:
                    pass
            self.feats = feats
            for name, mdl in self.models.items():
                save_model(mdl, self.feats, self._paths(symbol)[name])
            return float("nan")
        self.feats = feats
        tscv = TimeSeriesSplit(n_splits=5)
        for name, mdl in self.models.items():
            aucs = []
            for tr, te in tscv.split(X):
                mdl.fit(X.iloc[tr], y.iloc[tr])
                p = mdl.predict_proba(X.iloc[te])[:,1]
                aucs.append(roc_auc_score(y.iloc[te], p))
            mdl.fit(X, y)
            self.cv_auc[name] = float(np.mean(aucs)) if aucs else float("nan")
        for name, mdl in self.models.items():
            save_model(mdl, self.feats, self._paths(symbol)[name])
        valid = [v for v in self.cv_auc.values() if v == v]
        return float(np.mean(valid)) if valid else float("nan")

    def load_if_exists(self, symbol: str):
        for name in list(self.models.keys()):
            m, feats = load_model(self._paths(symbol)[name])
            if m is not None and feats:
                self.models[name] = m
                self.feats = feats

    def predict_proba(self, row: pd.Series) -> float:
        if not self.feats:
            raise RuntimeError("EnsembleLearner not fitted")
        X = pd.DataFrame([row[self.feats].astype(float)], columns=self.feats)
        preds = {name: float(mdl.predict_proba(X)[0,1]) for name, mdl in self.models.items()}
        w = {}
        for name in preds:
            base = self.cv_auc.get(name, 0.5)
            live = self.live_score.get(name, 0.5)
            w[name] = max(1e-6, (0.5*base if base == base else 0.0) + 0.5*live)
        s = sum(w.values()) or 1.0
        for k in w: w[k] /= s
        return sum(w[name]*preds[name] for name in preds)

    def update_live_score(self, y_true_up: int, last_row: pd.Series):
        try:
            p_pred = self.predict_proba(last_row)
        except Exception:
            return
        brier = (p_pred - y_true_up)**2
        score = 1.0 - brier
        for name in self.live_score:
            self.live_score[name] = 0.9*self.live_score[name] + 0.1*score

# ======= Threshold / Bandit =======
class AdaptiveThreshold:
    def __init__(self, base: float = 0.52, min_t: float = 0.45, max_t: float = 0.65, step: float = 0.01, window: int = 20):
        self.t = base; self.min_t = min_t; self.max_t = max_t; self.step = step; self.window = window
        self.returns: List[float] = []

    def record_trade(self, pct_return: float):
        self.returns.append(pct_return); self.returns = self.returns[-self.window:]
        avg = np.mean(self.returns) if self.returns else 0.0
        if   avg > 0.0: self.t = max(self.min_t, self.t - self.step)
        elif avg < 0.0: self.t = min(self.max_t, self.t + self.step)

    def current(self) -> float:
        return self.t

class ThresholdBandit:
    def __init__(self, candidates: List[float], eps: float = 0.1):
        self.cands = candidates; self.eps = eps
        self.counts = {c: 0 for c in candidates}
        self.values = {c: 0.0 for c in candidates}
        self.active: float = candidates[0]

    def select(self) -> float:
        if random.random() < self.eps:
            self.active = random.choice(self.cands)
        else:
            self.active = max(self.cands, key=lambda c: self.values[c])
        return self.active

    def update(self, thr: float, reward: float):
        self.counts[thr] += 1
        n = self.counts[thr]; v = self.values[thr]
        self.values[thr] = v + (reward - v) / n

# ======= Plotting =======
def plot_signals(symbol: str, df: pd.DataFrame, trades: List[Trade], out="plots", overlay_indicators=True):
    if df.empty: return
    if ("ema12" not in df.columns) or ("rsi14" not in df.columns):
        df = add_features(df.copy())
    plt.figure(figsize=(12,6))
    plt.title(f"{symbol.upper()} price & trades")
    plt.plot(df["close_time"], df["close"], label="Close")
    if overlay_indicators:
        plt.plot(df["close_time"], df["ema12"], label="EMA12", alpha=0.8)
    for t in trades:
        if t.symbol != symbol: continue
        tt = pd.to_datetime(t.time)
        plt.scatter([tt], [t.price], marker="^" if t.side=="BUY" else "v")
    plt.legend(); plt.tight_layout()
    os.makedirs(out, exist_ok=True)
    plt.savefig(os.path.join(out, f"{symbol}_price_signals.png"))
    plt.close()

def plot_equity(equity_hist: List[Tuple[str,float]], out="plots"):
    if not equity_hist: return
    t = [pd.to_datetime(x[0]) for x in equity_hist]; v = [x[1] for x in equity_hist]
    plt.figure(figsize=(12,4)); plt.title("Equity Curve"); plt.plot(t, v)
    plt.tight_layout(); os.makedirs(out, exist_ok=True)
    plt.savefig(os.path.join(out, "equity_curve.png")); plt.close()

# ======= Regime & Sizing / Risk =======
def regime_ok(row: pd.Series, need_bull_ema: bool, min_vol_ratio: float) -> bool:
    ema12 = float(row.get("ema12", np.nan))
    ema26 = float(row.get("ema26", np.nan))
    vr    = float(row.get("vol_ratio", np.nan))
    cond_ema = (ema12 > ema26) if need_bull_ema else True
    cond_vol = (vr >= min_vol_ratio) if min_vol_ratio is not None else True
    return cond_ema and cond_vol

def kelly_size(prob_up: float, cap: float, floor: float, scale: float) -> float:
    edge = max(0.0, prob_up - 0.5) * 2.0   # 0..1
    target = floor + (cap - floor) * (scale * edge)
    return float(np.clip(target, 0.0, cap))

# ======= Shutdown flag =======
_SHUTDOWN = False
def _signal_handler(signum, frame):
    global _SHUTDOWN
    _SHUTDOWN = True
    L(f"[{ts()}] Received signal {signum}. Shutting down gracefully...")

# ======= Helpers for drawdown / daily P&L =======
def max_drawdown(equity_series: List[Tuple[str,float]]) -> float:
    if not equity_series:
        return 0.0
    vals = [v for _, v in equity_series]
    peak = vals[0]
    maxdd = 0.0
    for v in vals:
        peak = max(peak, v)
        dd = (v/peak - 1.0)
        maxdd = min(maxdd, dd)
    return maxdd  # negative number

def today_pnl(equity_series: List[Tuple[str,float]]) -> float:
    if not equity_series:
        return 0.0
    df = pd.DataFrame(equity_series, columns=["time","equity"])
    df["dt"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.date
    today = pd.Timestamp.now(tz=LOCAL_TZ).date()
    dft = df[df["dt"] == today]
    if dft.empty:
        return 0.0
    return float(dft["equity"].iloc[-1] - dft["equity"].iloc[0])

# ======= Runner =======
async def run_loop(args, symbols: List[str], models: Dict[str, EnsembleLearner],
                   broker: PaperBroker, data_live: Dict[str, pd.DataFrame],
                   adapt: AdaptiveThreshold, bandit: ThresholdBandit):

    last_prices = {s: (float(df["close"].iloc[-1]) if not df.empty else np.nan) for s, df in data_live.items()}
    L(f"[{ts()}] Live poll started for: {', '.join(symbols)} (live_interval={args.live_interval})")

    # Soft cache of last close_time per symbol to prevent double-processing
    last_ct: Dict[str, pd.Timestamp] = {}
    for s, df in data_live.items():
        last_ct[s] = df["close_time"].iloc[-1] if not df.empty else pd.Timestamp(0, tz="UTC")

    # Per-symbol bar counters for retrain cadence & cooldown
    bars_since_retrain: Dict[str, int] = {s: 0 for s in symbols}
    cooldown_left: Dict[str, int] = {s: 0 for s in symbols}

    loop = asyncio.get_event_loop()
    executor = None  # default
    async def fetch_one(sym: str):
        return await loop.run_in_executor(executor, fetch_recent, sym, args.live_interval)

    while not _SHUTDOWN:
        try:
            now_utc = utcnow()
            # Risk kill-switch checks
            eqdd = max_drawdown(broker.equity_history)
            if args.max_drawdown_pct > 0 and eqdd <= -abs(args.max_drawdown_pct)/100.0:
                L(f"[{ts()}] Max drawdown reached ({eqdd*100:.2f}%). Flattening and exiting.")
                when_str = fmt_local_from_utc(now_utc)
                for sym in symbols:
                    price = last_prices.get(sym, np.nan)
                    if np.isfinite(price):
                        broker.force_close(when_str, sym, price)
                break

            if args.daily_loss_limit_pct > 0:
                pnl_today = today_pnl(broker.equity_history)
                if broker.starting_cash > 0 and (pnl_today / broker.starting_cash) <= -abs(args.daily_loss_limit_pct)/100.0:
                    L(f"[{ts()}] Daily loss limit hit ({pnl_today:.2f}). Flattening and pausing new entries for today.")
                    when_str = fmt_local_from_utc(now_utc)
                    for sym in symbols:
                        price = last_prices.get(sym, np.nan)
                        if np.isfinite(price):
                            broker.force_close(when_str, sym, price)
                    await asyncio.sleep(30)
                    continue

            if args.rth_only and not is_market_open(now_utc, rth_only=True):
                when_str = fmt_local_from_utc(now_utc)
                broker.mark(when_str, {k: v for k, v in last_prices.items() if np.isfinite(v)})
                await asyncio.sleep(20)
                continue

            # Parallel fetch all symbols
            recents = await asyncio.gather(*[fetch_one(sym) for sym in symbols])

            for sym, recent in zip(symbols, recents):
                if _SHUTDOWN: break
                if recent is None or recent.empty:
                    continue

                closed = recent[recent["close_time"] < (now_utc - pd.Timedelta(seconds=1))]
                if closed.empty:
                    continue

                df_prev = data_live[sym]
                last_known_ct = last_ct.get(sym, pd.Timestamp(0, tz="UTC"))
                new_bars = closed[closed["close_time"] > last_known_ct]
                if new_bars.empty:
                    continue

                # append & cap
                df = pd.concat([df_prev, new_bars], ignore_index=True)
                if args.max_bars and len(df) > args.max_bars:
                    df = df.iloc[-args.max_bars:].reset_index(drop=True)
                data_live[sym] = df
                last_ct[sym] = df["close_time"].iloc[-1]
                bars_since_retrain[sym] += len(new_bars)

                # walk-forward retrain
                if args.retrain_every_bars > 0 and bars_since_retrain[sym] >= args.retrain_every_bars:
                    auc = models[sym].fit(sym.upper(), df.copy())
                    L(f"[{ts()}] {sym.upper()} retrain AUC ~ {auc if auc==auc else float('nan'):.3f}")
                    bars_since_retrain[sym] = 0

                feats_df = add_features(df.copy(), lag_k=models[sym].lag_k)
                last_row = feats_df.iloc[-1]
                price = float(last_row["close"])
                last_prices[sym] = price
                when_str = fmt_local_from_utc(last_row["close_time"])

                # threshold selection
                thr_base = bandit.select()
                thr_live = adapt.current()
                thr = float(np.clip(0.5*(thr_base + thr_live), 0.40, 0.70))

                # regime gate
                if args.regime_filter and not regime_ok(last_row, need_bull_ema=True, min_vol_ratio=args.regime_min_vol_ratio):
                    broker.mark(when_str, {k: v for k, v in last_prices.items() if np.isfinite(v)})
                    if args.exit_on_regime_break:
                        t = broker.force_close(when_str, sym, price)
                        if t:
                            L(f"[{when_str}] {sym.upper()} REGIME EXIT @ {t.price:.2f}")
                            adapt.record_trade(t.pct_return); bandit.update(thr, t.pct_return)
                    continue

                # probability
                try:
                    proba_up = models[sym].predict_proba(last_row)
                except Exception:
                    proba_up = float("nan")

                broker.mark(when_str, {k: v for k, v in last_prices.items() if np.isfinite(v)})
                pos = broker.pos.get(sym); has_pos = pos is not None and pos.qty > 0

                # STOPS first
                stop_triggered = False
                if has_pos:
                    if args.atr_stop_mult > 0:
                        atr = float(last_row.get("atr14", np.nan))
                        if atr == atr:
                            stop_price = pos.entry - args.atr_stop_mult * atr
                            if price <= stop_price:
                                t = broker.force_close(when_str, sym, price)
                                if t:
                                    L(f"[{when_str}] {sym.UPPER()} ATR STOP -> SELL @ {t.price:.2f} (stop≈{stop_price:.2f})")
                                    adapt.record_trade(t.pct_return); bandit.update(thr, t.pct_return)
                                    cooldown_left[sym] = max(cooldown_left[sym], args.cooldown_bars_after_loss)
                                stop_triggered = True

                    if (not stop_triggered) and args.time_stop_bars > 0 and pos.entry_ct is not None:
                        bars_held = (df["close_time"] > pos.entry_ct).sum()
                        unrl = (price - pos.entry) / pos.entry if pos.entry > 0 else 0.0
                        if bars_held >= args.time_stop_bars and unrl <= 0:
                            t = broker.force_close(when_str, sym, price)
                            if t:
                                L(f"[{when_str}] {sym.upper()} TIME STOP -> SELL @ {t.price:.2f} (bars={bars_held})")
                                adapt.record_trade(t.pct_return); bandit.update(thr, t.pct_return)
                                cooldown_left[sym] = max(cooldown_left[sym], args.cooldown_bars_after_loss)
                            stop_triggered = True

                if stop_triggered:
                    continue

                # Cooldown after loss
                if cooldown_left[sym] > 0:
                    cooldown_left[sym] -= 1
                    continue

                # Sizing
                target_frac = 0.0
                if np.isfinite(proba_up) and proba_up >= thr:
                    if args.sizing == "allin":
                        target_frac = args.max_pos_frac
                    else:
                        target_frac = kelly_size(proba_up, args.max_pos_frac, args.min_pos_frac, args.kelly_scale)

                prices_snapshot = {k: v for k, v in last_prices.items() if np.isfinite(v)}
                has_pos = broker.pos.get(sym, Position()).qty > 0

                if target_frac == 0.0 and has_pos:
                    t = broker.force_close(when_str, sym, price)
                    if t:
                        L(f"[{when_str}] {sym.upper()} EXIT -> SELL @ {t.price:.2f}")
                        adapt.record_trade(t.pct_return); bandit.update(thr, t.pct_return)
                        if t.pct_return < 0:
                            cooldown_left[sym] = max(cooldown_left[sym], args.cooldown_bars_after_loss)

                elif target_frac > 0.0:
                    equity_now = broker.equity(prices_snapshot | {sym: price})
                    current_notional = broker.pos.get(sym, Position()).qty * price
                    desired_notional = equity_now * target_frac
                    if abs(desired_notional - current_notional) >= args.min_trade_notional:
                        t = broker.rebalance_to_target(when_str, sym, price, target_frac, prices_snapshot)
                        if t:
                            L(f"[{when_str}] {sym.upper()} {t.side} -> target {target_frac:.2f} | p={proba_up:.3f} thr={thr:.2f}")

                # Save artifacts periodically
                if (len(df) % args.save_every == 0):
                    plot_signals(sym, df, broker.trades)
                    plot_equity(broker.equity_history)
                    pd.DataFrame([t.__dict__ for t in broker.trades]).to_csv("trades.csv", index=False)
                    pd.DataFrame(broker.equity_history, columns=["time","equity"]).to_csv("equity.csv", index=False)

            await asyncio.sleep(args.poll_secs)

        except Exception as e:
            L(f"[{ts()}] Loop error: {e}. Cooling 5s...")
            await asyncio.sleep(5)

    L(f"[{ts()}] Loop terminated. Final cash={broker.cash:.2f}")

# ======= Optional immediate entry =======
def initial_entry_if_requested(args, models: Dict[str, EnsembleLearner], data_live: Dict[str, pd.DataFrame],
                               broker: PaperBroker, adapt: AdaptiveThreshold, bandit: ThresholdBandit):
    if not args.enter_on_start: return
    last_prices = {s: (float(df["close"].iloc[-1]) if not df.empty else np.nan) for s, df in data_live.items()}
    recent_cts = [df["close_time"].iloc[-1] for df in data_live.values() if not df.empty]
    if not recent_cts: return
    when = max(recent_cts); when_str = fmt_local_from_utc(when)
    broker.mark(when_str, {k: v for k, v in last_prices.items() if np.isfinite(v)})

    for sym, df in data_live.items():
        if df.empty: continue
        feats_df = add_features(df.copy(), lag_k=models[sym].lag_k)
        last_row = feats_df.iloc[-1]
        try:
            p = models[sym].predict_proba(last_row)
        except Exception:
            p = float("nan")
        thr = float(np.clip(0.5*(bandit.select() + adapt.current()), 0.40, 0.70))
        price = float(df["close"].iloc[-1])

        if np.isfinite(p) and p >= thr:
            if args.sizing == "allin":
                target = args.max_pos_frac
            else:
                target = kelly_size(p, args.max_pos_frac, args.min_pos_frac, args.kelly_scale)
            if target > 0:
                t = broker.rebalance_to_target(when_str, sym, price, target, {sym: price})
                if t:
                    L(f"[{when_str}] START {sym.upper()} BUY -> target {target:.2f} | p={p:.3f} thr={thr:.2f}")
                    break
    L(f"[{when_str}] Starting cash={broker.cash:.2f}")

# ======= Auto Mode helpers =======

MODEL_KEYS = ["gb", "rf", "et"]

def models_present(symbols: List[str], model_dir: str) -> bool:
    os.makedirs(model_dir, exist_ok=True)
    for sym in [s.upper() for s in symbols]:
        for k in MODEL_KEYS:
            p = os.path.join(model_dir, f"{VERSION_TAG}_{sym}_{k}.pkl")
            if not os.path.exists(p):
                return False
    return True

def train_on_backfill(args, symbols: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, EnsembleLearner]]:
    start_dt = datetime.utcnow() - timedelta(days=365*args.backfill_years)
    end_dt   = datetime.utcnow()
    L(f"[{ts()}] Backfilling {args.train_interval} for ~{args.backfill_years}y: {start_dt.date()} -> {end_dt.date()}")

    data_train: Dict[str, pd.DataFrame] = {}
    for s in [x.upper() for x in symbols]:
        df_hist = fetch_backfill_chunked(s, args.train_interval, start_dt, end_dt)
        if df_hist.empty:
            L(f"[{ts()}] WARN: no backfill for {s} (Yahoo may rate-limit).")
        data_train[s] = df_hist

    models: Dict[str, EnsembleLearner] = {}
    for s in data_train:
        L(f"[{ts()}] Training ensemble for {s} on {args.train_interval} backfill ...")
        ens = EnsembleLearner(model_dir=args.model_dir, lag_k=args.lag_k, horizon=args.horizon)
        ens.load_if_exists(s)
        base_df = data_train[s]
        if base_df.empty:
            L(f"[{ts()}] WARN: {s} backfill empty; attempting to warm with live for feature shape.")
            live_df = fetch_recent(s, args.live_interval)
            base_df = live_df
        auc = ens.fit(s, base_df.copy())
        models[s] = ens
        L(f"[{ts()}] {s} backfill AUC ~ {auc if auc==auc else float('nan'):.3f}")
    return data_train, models

def warmup_live(args, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    data_live: Dict[str, pd.DataFrame] = {}
    for s in [x.upper() for x in symbols]:
        L(f"[{ts()}] Warmup {s} {args.live_interval} ...")
        df_live = fetch_recent(s, args.live_interval)
        if args.lookback_live and len(df_live) > args.lookback_live:
            df_live = df_live.iloc[-args.lookback_live:].reset_index(drop=True)
        data_live[s] = df_live
    return data_live

def start_artifacts(broker: PaperBroker, data_live: Dict[str, pd.DataFrame], lag_k: int):
    now = ts()
    last_prices = {s: (float(df["close"].iloc[-1]) if not df.empty else np.nan) for s, df in data_live.items()}
    broker.mark(now, {k:v for k,v in last_prices.items() if np.isfinite(v)})
    for s, df in data_live.items():
        plot_signals(s, add_features(df.copy(), lag_k=lag_k), trades=[])
    plot_equity(broker.equity_history)

# ======= Main (with Auto Mode) =======
def main():
    global LOGGER
    p = argparse.ArgumentParser(description="OP Live Stocks ML Paper-Trader (Yahoo) — Auto Mode")

    # Symbols / intervals
    p.add_argument("--symbols", nargs="+", default=["AAPL","MSFT"])
    p.add_argument("--live_interval", choices=["1m","2m","5m"], default="1m")
    p.add_argument("--train_interval", choices=["5m","15m","30m","60m"], default="60m",
                   help="interval for backfill training (Yahoo: <60m limited to ~60 days)")
    p.add_argument("--backfill_years", type=int, default=2)
    p.add_argument("--lookback_live", type=int, default=1200)

    # Loop / hours
    p.add_argument("--poll_secs", type=int, default=10)
    p.add_argument("--rth_only", action="store_true", default=True)
    p.add_argument("--exit_on_regime_break", action="store_true", default=False)

    # Money / fees
    p.add_argument("--cash", type=float, default=100.0)
    p.add_argument("--fee_bps", type=float, default=5.0)
    p.add_argument("--slippage_bps", type=float, default=1.0)
    p.add_argument("--min_cash_reserve", type=float, default=0.0)
    p.add_argument("--min_trade_notional", type=float, default=5.0)

    # ML controls
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--lag_k", type=int, default=3)
    p.add_argument("--prob_buy", type=float, default=0.52)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--enter_on_start", action="store_true")
    p.add_argument("--train_once", action="store_true", default=True)
    p.add_argument("--retrain_every_bars", type=int, default=60)
    p.add_argument("--adaptive_threshold", action="store_true", default=True)
    p.add_argument("--bandit_thresholds", default="0.50,0.52,0.54,0.56")
    p.add_argument("--bandit_eps", type=float, default=0.1)

    # Regime
    p.add_argument("--regime_filter", action="store_true", default=True)
    p.add_argument("--regime_min_vol_ratio", type=float, default=1.0)

    # Sizing
    p.add_argument("--sizing", choices=["allin","kelly"], default="kelly")
    p.add_argument("--kelly_scale", type=float, default=1.0)
    p.add_argument("--max_pos_frac", type=float, default=1.0)
    p.add_argument("--min_pos_frac", type=float, default=0.25)

    # Stops
    p.add_argument("--atr_stop_mult", type=float, default=1.5)
    p.add_argument("--time_stop_bars", type=int, default=10)

    # Risk kill-switches
    p.add_argument("--max_drawdown_pct", type=float, default=0.0)
    p.add_argument("--daily_loss_limit_pct", type=float, default=0.0)
    p.add_argument("--cooldown_bars_after_loss", type=int, default=5)

    # Caps
    p.add_argument("--max_bars", type=int, default=3000)

    # Models
    p.add_argument("--model_dir", type=str, default="./models")

    # PM2 logging
    p.add_argument("--log_dir", type=str, default="./logs")
    p.add_argument("--pm2_tag", type=str, default="")

    # NEW: Mode controls (defaults to auto)
    p.add_argument("--mode", choices=["auto","train","live"], default="auto",
                   help="auto (default): decide based on market hours + model presence; 'train' or 'live' to force.")
    p.add_argument("--auto_refresh_retrain", action="store_true", default=True,
                   help="In auto mode, if market closed and models exist, refresh-train then exit.")

    args = p.parse_args()
    ensure_dirs()

    # Init logger
    log_filename = "op_stocks_bot.log" if not args.pm2_tag else f"op_stocks_bot_{args.pm2_tag}.log"
    log_path = os.path.join(args.log_dir, log_filename) if args.log_dir else None
    LOGGER = TinyLogger(log_path=log_path)

    # Hook signals
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT,  _signal_handler)

    symbols = [x.upper() for x in args.symbols]
    now_utc = utcnow()
    open_now = (not args.rth_only) or is_market_open(now_utc, rth_only=True)
    have_models = models_present(symbols, args.model_dir)

    # Decide mode
    decided = args.mode
    if args.mode == "auto":
        if not have_models:
            decided = "train"  # need at least one training pass
        else:
            decided = "live" if open_now else "train"  # refresh-train when closed

    L(f"[{ts()}] Mode decision: {decided.upper()} | market_open={open_now} | models_present={have_models}")

    # === TRAIN mode (also used by AUTO when closed or models missing) ===
    if decided == "train":
        data_train, models = train_on_backfill(args, symbols)
        if open_now and have_models:
            # Warm-up & proceed straight to live (AUTO-refresh + live)
            data_live = warmup_live(args, symbols)
            broker = PaperBroker(cash=args.cash, fee_bps=args.fee_bps,
                                 slippage_bps=args.slippage_bps, min_cash_reserve=args.min_cash_reserve)
            start_artifacts(broker, data_live, lag_k=args.lag_k)

            # Build model map aligned to symbols (use freshly trained if available)
            ens_map = {}
            for s in symbols:
                ens = EnsembleLearner(model_dir=args.model_dir, lag_k=args.lag_k, horizon=args.horizon)
                ens.load_if_exists(s)  # load the just-saved models
                ens_map[s] = ens

            adapt = AdaptiveThreshold(base=args.prob_buy)
            cands = [float(x) for x in args.bandit_thresholds.split(",")]
            bandit = ThresholdBandit(candidates=cands, eps=args.bandit_eps)
            if args.enter_on_start:
                initial_entry_if_requested(args, ens_map, data_live, broker, adapt, bandit)
            try:
                asyncio.run(run_loop(args, symbols, ens_map, broker, data_live, adapt, bandit))
            finally:
                # Save artifacts on exit
                try:
                    for s, df in data_live.items():
                        if not df.empty:
                            plot_signals(s, df, broker.trades)
                    plot_equity(broker.equity_history)
                    pd.DataFrame([t.__dict__ for t in broker.trades]).to_csv("trades.csv", index=False)
                    pd.DataFrame(broker.equity_history, columns=["time","equity"]).to_csv("equity.csv", index=False)
                    L(f"[{ts()}] Artifacts saved. Bye.")
                except Exception as e:
                    L(f"[{ts()}] WARN: failed to save artifacts on exit: {e}")
        else:
            if args.mode == "auto" and not open_now and args.auto_refresh_retrain:
                L(f"[{ts()}] Market closed. Refreshed models and exiting (auto).")
            else:
                L(f"[{ts()}] Training complete. Exiting.")
        return

    # === LIVE mode (manual or auto when open & models present) ===
    if decided == "live":
        # Ensure models exist (if not, do a quick train pass)
        if not have_models:
            L(f"[{ts()}] No models found; performing initial backfill training first ...")
            train_on_backfill(args, symbols)

        data_live = warmup_live(args, symbols)
        broker = PaperBroker(cash=args.cash, fee_bps=args.fee_bps,
                             slippage_bps=args.slippage_bps, min_cash_reserve=args.min_cash_reserve)
        start_artifacts(broker, data_live, lag_k=args.lag_k)

        ens_map: Dict[str, EnsembleLearner] = {}
        for s in symbols:
            ens = EnsembleLearner(model_dir=args.model_dir, lag_k=args.lag_k, horizon=args.horizon)
            ens.load_if_exists(s)
            ens_map[s] = ens

        adapt = AdaptiveThreshold(base=args.prob_buy) if args.adaptive_threshold else AdaptiveThreshold(base=args.prob_buy)
        cands = [float(x) for x in args.bandit_thresholds.split(",")]
        bandit = ThresholdBandit(candidates=cands, eps=args.bandit_eps)

        if args.enter_on_start:
            initial_entry_if_requested(args, ens_map, data_live, broker, adapt, bandit)

        try:
            asyncio.run(run_loop(args, symbols, ens_map, broker, data_live, adapt, bandit))
        finally:
            try:
                for s, df in data_live.items():
                    if not df.empty:
                        plot_signals(s, df, broker.trades)
                plot_equity(broker.equity_history)
                pd.DataFrame([t.__dict__ for t in broker.trades]).to_csv("trades.csv", index=False)
                pd.DataFrame(broker.equity_history, columns=["time","equity"]).to_csv("equity.csv", index=False)
                L(f"[{ts()}] Artifacts saved. Bye.")
            except Exception as e:
                L(f"[{ts()}] WARN: failed to save artifacts on exit: {e}")

if __name__ == "__main__":
    main()
