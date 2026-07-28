"""
AIROS V3 — Inference API
FastAPI wrapper around AIROSEngine.
Loads model from airos_v3_best.pt and feature_contract_3.0.0.json at startup.

Endpoints:
  GET  /health          — liveness + model metadata
  POST /predict         — single inference call
  POST /predict/batch   — multiple assets / sequences (optional)

Environment variables (override CONFIG defaults):
  MODEL_PATH            — path to .pt checkpoint  (default: airos_v3_best.pt)
  CONTRACT_PATH         — path to feature_contract JSON
  CONFIDENCE_THRESHOLD  — float override (default: from checkpoint CONFIG)
  DEVICE                — "cpu" or "cuda"  (default: auto-detect)
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Model definition  (must match notebook cells 7 exactly)
# ──────────────────────────────────────────────────────────────────────────────

class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_len: int):
        super().__init__()
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos   = torch.arange(max_len).float()
        freqs = torch.outer(pos, theta)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T    = x.size(2)
        c, s = self.cos[:T], self.sin[:T]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d: int, H: int, dff: int, drop: float, rope: RoPE):
        super().__init__()
        self.n1   = nn.LayerNorm(d)
        self.n2   = nn.LayerNorm(d)
        self.rope  = rope
        self.Wqkv = nn.Linear(d, d * 3, bias=False)
        self.Wo   = nn.Linear(d, d,     bias=False)
        self.ff   = nn.Sequential(
            nn.Linear(d, dff), nn.GELU(), nn.Dropout(drop), nn.Linear(dff, d)
        )
        self.drop = nn.Dropout(drop)
        self.H    = H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        h = d // self.H
        q, k, v = self.Wqkv(self.n1(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.H, h).transpose(1, 2)
        k = k.view(B, T, self.H, h).transpose(1, 2)
        v = v.view(B, T, self.H, h).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + self.drop(self.Wo(a))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class AttnPool(nn.Module):
    def __init__(self, d: int, P: int):
        super().__init__()
        self.q    = nn.Parameter(torch.randn(1, P, d) * 0.02)
        self.kp   = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(P * d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        k = self.kp(x)
        w = F.softmax(
            torch.bmm(self.q.expand(B, -1, -1), k.transpose(1, 2)) / x.size(-1) ** 0.5,
            dim=-1,
        )
        out = torch.bmm(w, x).reshape(B, -1)
        return self.norm(self.proj(out))


class AIROSV3(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, H, L, dff = cfg["d_model"], cfg["n_heads"], cfg["n_layers"], cfg["d_ff"]
        drop, P, n_h = cfg["dropout"], cfg["pool_heads"], len(cfg["horizons"])

        self.in_proj    = nn.Sequential(nn.Linear(cfg["token_dim"], d), nn.LayerNorm(d))
        self.state_proj = nn.Sequential(nn.Linear(cfg["token_dim"] * 2, d), nn.GELU(), nn.LayerNorm(d))
        rope            = RoPE(d // H, cfg["max_seq_len"])
        self.blocks     = nn.ModuleList([Block(d, H, dff, drop, rope) for _ in range(L)])
        self.norm       = nn.LayerNorm(d)
        self.pool       = AttnPool(d, P)
        self.fusion     = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(d))
        self.heads      = nn.ModuleList([nn.Linear(d, 2) for _ in range(n_h)])
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict:
        B, T, _ = x.shape
        st = self.state_proj(
            torch.cat([x[:, 1:].mean(1), x[:, 1:].std(1)], -1)
        ).unsqueeze(1)
        x = torch.cat([self.in_proj(x), st], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        seq, state = x[:, :T], x[:, T]
        fused  = self.fusion(torch.cat([self.pool(seq), state], -1))
        logits = torch.stack([h(fused) for h in self.heads], dim=1)  # (B,4,2)
        conf   = F.softmax(logits, dim=-1).max(-1).values             # (B,4)
        return {"logits": logits, "confidence": conf}


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering  (verbatim from notebook cells 3 and 4)
# ──────────────────────────────────────────────────────────────────────────────

def _ema_fast(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy(np.float32)


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d
    xd    = x.astype(np.float64)
    mean  = uniform_filter1d(xd,     size=w, mode="nearest")
    mean2 = uniform_filter1d(xd**2,  size=w, mode="nearest")
    return np.sqrt(np.maximum(mean2 - mean**2, 0.0)).astype(np.float32)


def _rolling_max(x: np.ndarray, w: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(x.astype(np.float32), size=w, mode="nearest")


def _rolling_min(x: np.ndarray, w: int) -> np.ndarray:
    from scipy.ndimage import minimum_filter1d
    return minimum_filter1d(x.astype(np.float32), size=w, mode="nearest")


def build_candle_features(ohlc_df: pd.DataFrame) -> np.ndarray:
    """Returns (N, 14) float32. Contract preserved from V2.2."""
    o = ohlc_df["open"].to_numpy(np.float32)
    h = ohlc_df["high"].to_numpy(np.float32)
    l = ohlc_df["low"].to_numpy(np.float32)
    c = ohlc_df["close"].to_numpy(np.float32)
    N   = len(c)
    eps = np.float32(1e-7)

    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr     = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14  = _ema_fast(tr, 14) + eps

    rng        = np.maximum(h - l, eps)
    body       = np.abs(c - o)
    body_ratio = body / rng
    upper_wick = (h - np.maximum(o, c)) / rng
    lower_wick = (np.minimum(o, c) - l) / rng
    close_pos  = (c - l) / rng
    direction  = np.sign(c - o).astype(np.float32)
    norm_range = rng / atr14
    atr_ratio  = atr14 / (np.abs(c) + eps)

    ret1      = np.empty(N, np.float32); ret1[0] = 0
    ret1[1:]  = (c[1:] - c[:-1]) / (np.abs(c[:-1]) + eps)

    mom5      = np.empty(N, np.float32); mom5[:5] = 0
    mom5[5:]  = (c[5:] - c[:-5]) / (np.abs(c[:-5]) + eps)

    mom14     = np.empty(N, np.float32); mom14[:14] = 0
    mom14[14:]= (c[14:] - c[:-14]) / (np.abs(c[:-14]) + eps)

    vol_std5  = _rolling_std(ret1, 5)

    from scipy.ndimage import uniform_filter1d
    avg_rng20   = uniform_filter1d(rng.astype(np.float64), size=20, mode="nearest").astype(np.float32)
    compression = rng / (avg_rng20 + eps)

    ema8        = _ema_fast(c, 8)
    ema21       = _ema_fast(c, 21)
    trend_state = (ema8 - ema21) / (atr14 + eps)

    swing_h      = _rolling_max(h, 20)
    swing_l      = _rolling_min(l, 20)
    swing_h_dist = (swing_h - c) / (atr14 + eps)
    swing_l_dist = (c - swing_l)  / (atr14 + eps)

    feats = np.stack([
        body_ratio, upper_wick, lower_wick, close_pos,
        direction,  norm_range, atr_ratio,
        mom5,       mom14,      vol_std5,
        compression, trend_state,
        swing_h_dist, swing_l_dist,
    ], axis=1).astype(np.float32)

    feats = np.clip(feats, -5.0, 5.0)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def build_regime_token(candle_feats_window: np.ndarray) -> np.ndarray:
    """Returns (22,) float32 regime token."""
    assert candle_feats_window.shape[1] >= 14
    recent = candle_feats_window[-20:]

    trend_strength  = float(np.mean(recent[:, 11]))
    compression_avg = float(np.mean(recent[:, 10]))
    vol_avg         = float(np.mean(recent[:, 6]))
    swing_h_prox    = float(recent[-1, 12])
    swing_l_prox    = float(recent[-1, 13])

    is_trending   = float(np.tanh(trend_strength))
    is_compressed = float(1.0 - np.clip(compression_avg, 0.0, 2.0) / 2.0)
    is_high_vol   = float(np.clip(vol_avg * 10.0, 0.0, 1.0))
    near_res      = float(1.0 / (1.0 + max(swing_h_prox, 0.0)))
    near_sup      = float(1.0 / (1.0 + max(swing_l_prox, 0.0)))
    trend_bias    = float(is_trending * (1.0 - is_compressed))
    breakout_risk = float(is_compressed * is_high_vol)

    regime_vals = np.array([
        is_trending, is_compressed, is_high_vol,
        near_res, near_sup, trend_bias, breakout_risk,
    ], dtype=np.float32)

    token = np.zeros(22, dtype=np.float32)
    token[:len(regime_vals)] = regime_vals
    return token


def build_tick_features(
    tick_store_df: pd.DataFrame,
    ohlc_df: pd.DataFrame,
    bucket_minutes: float,
    tick_max_per_bucket: int,
) -> np.ndarray:
    """Returns (N_bars, 8) float32."""
    eps    = np.float32(1e-7)
    N_bars = len(ohlc_df)
    ts_ns  = tick_store_df["timestamp"].astype(np.int64).to_numpy()
    prices = tick_store_df["price"].to_numpy(np.float64)

    diff    = np.diff(prices, prepend=prices[0])
    is_buy  = (diff > 0).astype(np.float32)
    is_sell = (diff < 0).astype(np.float32)
    ret     = np.where(prices[:-1] != 0, diff[1:] / prices[:-1], 0.0)
    ret     = np.concatenate([[0.0], ret]).astype(np.float32)

    bar_ns  = ohlc_df.index.astype(np.int64)
    bar_idx = np.searchsorted(bar_ns, ts_ns, side="right") - 1
    valid   = (bar_idx >= 0) & (bar_idx < N_bars)
    bar_idx = bar_idx[valid]
    is_buy  = is_buy[valid]
    is_sell = is_sell[valid]
    ret_v   = ret[valid]
    ts_ns_v = ts_ns[valid]

    count    = np.bincount(bar_idx, minlength=N_bars).astype(np.float32)
    buy_sum  = np.bincount(bar_idx, weights=is_buy,   minlength=N_bars).astype(np.float32)
    sell_sum = np.bincount(bar_idx, weights=is_sell,  minlength=N_bars).astype(np.float32)
    ret_sum  = np.bincount(bar_idx, weights=ret_v,    minlength=N_bars).astype(np.float32)
    ret2_sum = np.bincount(bar_idx, weights=ret_v**2, minlength=N_bars).astype(np.float32)

    _ts_f    = ts_ns_v.astype(np.float64)
    _sort_bi = np.argsort(bar_idx, kind="stable")
    _sbi     = bar_idx[_sort_bi]
    _sts     = _ts_f[_sort_bi]
    _uniq, _starts = np.unique(_sbi, return_index=True)
    max_ts_bar = np.full(N_bars, -np.inf, np.float64)
    min_ts_bar = np.full(N_bars,  np.inf, np.float64)
    max_ts_bar[_uniq] = np.maximum.reduceat(_sts, _starts)
    min_ts_bar[_uniq] = np.minimum.reduceat(_sts, _starts)
    duration_s = np.maximum((max_ts_bar - min_ts_bar) / 1e9, 1.0).astype(np.float32)
    duration_s[~np.isfinite(duration_s)] = 1.0

    total           = count + eps
    buyer_pressure  = buy_sum  / total
    seller_pressure = sell_sum / total
    delta           = buyer_pressure - seller_pressure
    imbalance       = np.clip((buy_sum - sell_sum) / total, -1.0, 1.0)
    tick_speed      = np.clip(count / (bucket_minutes * 15.0), 0.0, 1.0)
    tick_accel      = np.empty(N_bars, dtype=np.float32)
    tick_accel[0]   = 0.0
    tick_accel[1:]  = np.diff(tick_speed)
    e_r             = ret_sum  / total
    e_r2            = ret2_sum / total
    micro_vol       = np.clip(np.sqrt(np.maximum(e_r2 - e_r**2, 0.0)) * 1000.0, 0.0, 1.0)
    effective_max   = np.maximum(1.0, tick_max_per_bucket * (bucket_minutes / 14.0))
    density         = np.clip(count / effective_max, 0.0, 1.0)

    features = np.stack([
        buyer_pressure, seller_pressure, delta,
        tick_speed, tick_accel, imbalance,
        micro_vol, density,
    ], axis=1).astype(np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Inference engine  (from notebook cell 10)
# ──────────────────────────────────────────────────────────────────────────────

class AIROSEngine:
    def __init__(self, model: AIROSV3, cfg: dict, device: torch.device, asset: str):
        self.model  = model.eval()
        self.cfg    = cfg
        self.device = device
        self.asset  = asset
        self.seq    = cfg["sequence_length"]
        self.mod_seq= cfg["model_seq_len"]
        self.thresh = cfg["confidence_threshold"]
        self.ph     = cfg["primary_horizon"]
        self.hnames = [f"T+{h}" for h in cfg["horizons"]]
        self.bk_min = cfg["tick_bucket_min_minutes"]
        self.bk_max = cfg["tick_bucket_max_minutes"]
        self.bk_max_ticks = cfg["tick_max_per_bucket"]

        n_h   = len(cfg["horizons"])
        temps = cfg.get("calibration_temperature", [1.0] * n_h)
        self.temperature = torch.tensor(temps, dtype=torch.float32).view(1, n_h, 1).to(device)

    def _clamp_bucket_minutes(self, bucket_minutes: float) -> float:
        return float(max(self.bk_min, min(self.bk_max, bucket_minutes)))

    @torch.no_grad()
    def predict(
        self,
        ohlc_df: pd.DataFrame,
        tick_df: Optional[pd.DataFrame] = None,
        tick_bucket_minutes: Optional[float] = None,
    ) -> dict:
        warmup     = 0
        min_needed = self.seq
        if len(ohlc_df) < min_needed:
            return {
                "direction": "NO_TRADE",
                "reason": f"Need {min_needed} bars ({self.seq} + {warmup} warm-up), got {len(ohlc_df)}",
            }

        df_full = ohlc_df.iloc[-min_needed:].copy()
        cf_full = build_candle_features(df_full)
        cf      = cf_full[-self.seq:]
        df      = df_full.iloc[-self.seq:]

        if tick_df is not None and len(tick_df) >= 10:
            bk = self._clamp_bucket_minutes(
                tick_bucket_minutes if tick_bucket_minutes is not None else float(self.bk_min)
            )
            tick_trimmed = tick_df[tick_df["timestamp"] >= df.index[0]].copy()
            if len(tick_trimmed) >= 5:
                tf = build_tick_features(
                    tick_store_df       = tick_trimmed,
                    ohlc_df             = df,
                    bucket_minutes      = bk,
                    tick_max_per_bucket = self.bk_max_ticks,
                )
            else:
                tf = np.zeros((self.seq, 8), np.float32)
        else:
            tf = np.zeros((self.seq, 8), np.float32)

        market_bars = np.nan_to_num(
            np.concatenate([cf, tf], axis=1), 0.0, 0.0, 0.0
        )
        regime_tok = build_regime_token(cf)
        tok_full   = np.vstack([regime_tok[None, :], market_bars])
        inp = torch.from_numpy(tok_full).unsqueeze(0).to(self.device)

        t0  = time.perf_counter()
        out = self.model(inp)
        lat = (time.perf_counter() - t0) * 1000

        calibrated_logits = out["logits"] / self.temperature
        probs   = F.softmax(calibrated_logits[0], -1).cpu().numpy()
        conf    = probs.max(-1)
        ph_conf = float(conf[self.ph])
        ph_dir  = "BUY" if probs[self.ph, 1] >= 0.5 else "SELL"
        decision= ph_dir if ph_conf >= self.thresh else "NO_TRADE"

        return {
            "asset":      self.asset,
            "direction":  decision,
            "confidence": round(ph_conf, 4),
            "horizon":    self.hnames[self.ph],
            "regime": {
                "trend":         round(float(regime_tok[0]), 4),
                "compressed":    round(float(regime_tok[1]), 4),
                "high_vol":      round(float(regime_tok[2]), 4),
                "near_res":      round(float(regime_tok[3]), 4),
                "near_sup":      round(float(regime_tok[4]), 4),
                "trend_bias":    round(float(regime_tok[5]), 4),
                "breakout_risk": round(float(regime_tok[6]), 4),
            },
            "horizons": {
                hn: {
                    "direction":  "BUY" if probs[h, 1] >= 0.5 else "SELL",
                    "buy_prob":   round(float(probs[h, 1]), 4),
                    "confidence": round(float(conf[h]), 4),
                }
                for h, hn in enumerate(self.hnames)
            },
            "latency_ms": round(lat, 2),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Startup: load checkpoint and build engine
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH    = Path(os.getenv("MODEL_PATH",    "airos_v3_best.pt"))
CONTRACT_PATH = Path(os.getenv("CONTRACT_PATH", "feature_contract_3.0.0.json"))

_device_str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEVICE      = torch.device(_device_str)

# Load checkpoint
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
CONFIG: dict = ckpt["config"]

# Allow threshold override via env
_thresh_override = os.getenv("CONFIDENCE_THRESHOLD")
if _thresh_override:
    CONFIG["confidence_threshold"] = float(_thresh_override)

# Derived key must be present
CONFIG.setdefault("model_seq_len", CONFIG["sequence_length"] + 1)

# Build and load model
model = AIROSV3(CONFIG).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Load feature contract for metadata
contract_meta: dict = {}
if CONTRACT_PATH.exists():
    with open(CONTRACT_PATH) as f:
        contract_meta = json.load(f)

# Build engine (asset label from contract or checkpoint, fallback to "UNKNOWN")
_asset = contract_meta.get("asset") or ckpt.get("config", {}).get("project_name", "UNKNOWN")
engine = AIROSEngine(model, CONFIG, DEVICE, _asset)

gc.collect()
print(f"[AIROS] v{CONFIG['version']}  device={DEVICE}  asset={_asset}  threshold={CONFIG['confidence_threshold']}")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AIROS V3 Inference API",
    version=CONFIG["version"],
    description="Transformer-based FX direction model. POST OHLC + tick data, receive signal.",
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class OHLCBar(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 or Unix ms")
    open:      float
    high:      float
    low:       float
    close:     float


class TickRecord(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 or Unix ms")
    price:     float


class PredictRequest(BaseModel):
    ohlc:                List[OHLCBar]
    ticks:               Optional[List[TickRecord]] = None
    tick_bucket_minutes: Optional[float]            = Field(
        default=None,
        description="Actual broker bucket width in minutes (7–14). "
                    "Pass null to skip tick features.",
    )
    asset:               Optional[str]              = None


class HorizonResult(BaseModel):
    direction:  str
    buy_prob:   float
    confidence: float


class RegimeInfo(BaseModel):
    trend:         float
    compressed:    float
    high_vol:      float
    near_res:      float
    near_sup:      float
    trend_bias:    float
    breakout_risk: float


class PredictResponse(BaseModel):
    asset:      str
    direction:  str
    confidence: float
    horizon:    str
    regime:     RegimeInfo
    horizons:   Dict[str, HorizonResult]
    latency_ms: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_ts(ts: str) -> pd.Timestamp:
    """Accept ISO 8601 strings or Unix millisecond integers."""
    try:
        return pd.Timestamp(float(ts), unit="ms", tz="UTC")
    except (ValueError, TypeError):
        return pd.to_datetime(ts, utc=True)


def _ohlc_to_df(bars: List[OHLCBar]) -> pd.DataFrame:
    rows = [
        {"open": b.open, "high": b.high, "low": b.low, "close": b.close,
         "timestamp": _parse_ts(b.timestamp)}
        for b in bars
    ]
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df.index = df.index.tz_convert("UTC")
    return df


def _ticks_to_df(ticks: List[TickRecord]) -> pd.DataFrame:
    rows = [{"timestamp": _parse_ts(t.timestamp), "price": t.price} for t in ticks]
    df   = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "version":   CONFIG["version"],
        "asset":     engine.asset,
        "device":    str(DEVICE),
        "threshold": CONFIG["confidence_threshold"],
        "model_seq_len": CONFIG["model_seq_len"],
        "token_dim":     CONFIG["token_dim"],
        "checkpoint":    str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.ohlc) < CONFIG["sequence_length"]:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {CONFIG['sequence_length']} OHLC bars, "
                   f"got {len(req.ohlc)}.",
        )

    ohlc_df = _ohlc_to_df(req.ohlc)
    tick_df = _ticks_to_df(req.ticks) if req.ticks else None

    # Asset override per-request
    _eng_asset = req.asset or engine.asset
    if _eng_asset != engine.asset:
        engine.asset = _eng_asset

    result = engine.predict(ohlc_df, tick_df, req.tick_bucket_minutes)

    if "reason" in result:
        raise HTTPException(status_code=422, detail=result["reason"])

    return PredictResponse(**result)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point (for local dev)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
