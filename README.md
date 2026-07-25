# i90 — AIROS V5 Trading Engine
**Browser Extension · v60001**

A Chrome extension that intercepts live broker tick data, builds the AIROS V3 Transformer input sequence entirely in the browser, and calls a hosted inference API to produce trade signals.

---

## Architecture

```
Broker WebSocket
      │
      ▼
ws_hook.js          — Intercepts WS frames, extracts raw ticks + history
      │  postMessage(NEW_TICK / TICK_HISTORY_RESULT)
      ▼
content.js          — Bridge: page ↔ background  |  injects floating panel iframe
      │  chrome.runtime.sendMessage
      ▼
background.js       — Full preprocessing pipeline + Render API call
      │
      ├─ Stage 1: Resample ticks → 1-min OHLC (40 bars)
      ├─ Stage 2: Candle features (14 per bar)   ← Cell 3
      ├─ Stage 3: Tick microstructure (8 per bar) ← Cell 4
      ├─ Stage 4: Regime token (7 values, zero-padded to 22) ← Cell 3
      ├─ Stage 5: Sequence assembly → (41, 22) float tensor ← Cell 5
      │
      ▼
Render API          — AIROS V3 Transformer inference
https://i90-transformer-api.onrender.com/predict
      │
      ▼
panel.js / panel.html  — Signal display UI (floating iframe)
```

---

## Files

| File | Role |
|---|---|
| `manifest.json` | Extension manifest (MV3) |
| `background.js` | Service worker — full pipeline + API |
| `ws_hook.js` | WebSocket interceptor (MAIN world) |
| `content.js` | Bridge + overlay injector (ISOLATED world) |
| `panel.html` | UI shell |
| `panel.js` | UI logic + scanner |
| `icons/` | Extension icons (16 / 48 / 128 px) |

---

## Training Contract

All feature engineering in `background.js` mirrors the AIROS V3 training notebook exactly.

| Parameter | Value | Source |
|---|---|---|
| `sequence_length` | 40 market bars | Cell 1 CONFIG |
| `model_seq_len` | 41 (40 + regime token at pos 0) | Cell 1 CONFIG |
| `token_dim` | 22 (14 candle + 8 tick) | Cell 1 CONFIG |
| `ohlc_resample` | 1 minute | Cell 2 |
| `tick_max_per_bucket` | 210 | Cell 1 CONFIG |
| `clip range` | [−5, 5] | Cell 3 |
| `bucket_minutes` (training) | 1.0 | Cell 4 |
| `bucket_minutes` (inference) | clamped [7, 14] | Cell 4 |

### Feature Columns

**Candle features (cols 0–13):**

| Col | Name | Formula |
|---|---|---|
| 0 | body_ratio | `|close - open| / range` |
| 1 | upper_wick | `(high - max(open,close)) / range` |
| 2 | lower_wick | `(min(open,close) - low) / range` |
| 3 | close_pos | `(close - low) / range` |
| 4 | direction | `sign(close - open)` |
| 5 | norm_range | `range / ATR14` |
| 6 | atr_ratio | `ATR14 / |close|` |
| 7 | mom5 | `(close - close[i-5]) / |close[i-5]|` |
| 8 | mom14 | `(close - close[i-14]) / |close[i-14]|` |
| 9 | vol_std5 | rolling std of 1-bar returns, window 5 |
| 10 | compression | `range / rolling_mean(range, 20)` |
| 11 | trend_state | `(EMA8 - EMA21) / ATR14` |
| 12 | swing_h_dist | `(rolling_max(high, 20) - close) / ATR14` |
| 13 | swing_l_dist | `(close - rolling_min(low, 20)) / ATR14` |

**Tick microstructure features (cols 14–21):**

| Col | Name | Formula |
|---|---|---|
| 14 | buyer_pressure | `buy_ticks / total` |
| 15 | seller_pressure | `sell_ticks / total` |
| 16 | delta | `buyer_pressure - seller_pressure` |
| 17 | tick_speed | `count / (bucket_minutes × 10.0)`, clipped [0,1] |
| 18 | tick_accel | `tick_speed[i] - tick_speed[i-1]` |
| 19 | tick_imbalance | `(buys - sells) / total`, clipped [−1,1] |
| 20 | micro_vol | `sqrt(E[r²] - E[r]²) × 1000`, clipped [0,1] |
| 21 | tick_density | `count / tick_max_per_bucket`, clipped [0,1] |

**Regime token (position 0 of the (41,22) sequence):**

| Index | Name | Formula |
|---|---|---|
| 0 | is_trending | `tanh(mean(trend_state, last 20 bars))` |
| 1 | is_compressed | `1 - clip(mean(compression, last 20), 0, 2) / 2` |
| 2 | is_high_vol | `clip(mean(atr_ratio, last 20) × 10, 0, 1)` |
| 3 | near_res | `1 / (1 + max(swing_h_dist[last], 0))` |
| 4 | near_sup | `1 / (1 + max(swing_l_dist[last], 0))` |
| 5 | trend_bias | `is_trending × (1 - is_compressed)` |
| 6 | breakout_risk | `is_compressed × is_high_vol` |
| 7–21 | — | zero-padded |

### Leak Zeroing (Final Bar)

The last market bar (position 40) has the following columns zeroed before sending to the API — matching training Cell 5 `LEAK_COLS`:

```
[0, 3, 4, 14, 15, 16, 17, 18, 19, 20, 21]
```

Cols 0/3/4 (body_ratio, close_pos, direction) and all 8 tick features encode realized close direction at the reference bar. The model was trained to never see these at the final position.

---

## API

**Endpoint:** `POST https://i90-transformer-api.onrender.com/predict`

**Request:**
```json
{
  "token_sequence": [[...], ...],
  "asset": "EURUSD",
  "timeframe": 60
}
```
`token_sequence` is `(41, 22)` — a JSON array of 41 rows × 22 floats.

**Response:**
```json
{
  "action": "BUY" | "SELL" | "NO_TRADE",
  "confidence": 0.78,
  "p_up": 0.78,
  "p_down": 0.22,
  "reason": "..."
}
```

**Retry policy:** 3 attempts, 5-second delay between retries. Render cold-start can add up to ~30s on first request after inactivity.

---

## Supported Brokers

- `qxbroker.com` / `*.qxbroker.com`
- `quotex.com` / `*.quotex.com`
- `gxwbroker.com` / `*.gxwbroker.com`

---

## Data Flow Detail

### WebSocket Hook (`ws_hook.js`)

- Patches `window.WebSocket` constructor and `WebSocket.prototype.send` at `document_start` in the `MAIN` world before the broker page loads.
- Sets `binaryType = 'arraybuffer'` immediately on socket open — required before any binary frame arrives.
- **Asset theft:** intercepts the platform's outgoing `history/load` messages to capture the exact broker-side asset string (`eurusd_otc`, `EURUSD`, etc.) — more reliable than DOM scraping.
- **History request:** 3-step sequence (`instruments/follow` → `instruments/update` → `chart_notification/get`), 10s timeout.
- **Largest-frame selection:** broker sends multiple binary frames per response batch. Tick noise frames (< 200 bytes) are ignored. The history payload is always the largest frame. A 350ms debounce timer commits the largest frame seen in the batch.
- **Format A** (pre-built candles): expands each candle to 5 price points (O / extreme1 / extreme2 / C / inter-candle bridge) for ZigZag-compatible swing detection.
- **Format B** (raw ticks): passes all ticks directly, keeps the last `CANDLE_TARGET × 60` seconds.

### Preprocessing Pipeline (`background.js`)

Runs inside the service worker on every `requestSignal` action:

1. **Merge** — history ticks + live tick buffer, deduplicated by `timestamp:price` key.
2. **Resample** — bucket ticks into 1-minute OHLC bars, forward-fill gaps.
3. **Candle features** — 14-column matrix over all available bars for correct rolling lookback.
4. **Tick features** — 8-column matrix, bucket_minutes clamped to [7, 14].
5. **Regime token** — deterministic 7-value vector from the 40-bar sequence window.
6. **Sequence** — (41, 22) float array, leak columns zeroed on final bar.
7. **API call** — POST to Render endpoint, 3 retries.

### Overlay Panel (`content.js` + `panel.html` + `panel.js`)

- An `<iframe>` loaded from the extension origin is injected as a fixed overlay over the broker page. `frame-ancestors` CSP does not block extension-origin frames.
- Draggable (touch + mouse). Toggle button collapses/expands.
- A startup `BROKER_TAB_PING` message pins `brokerTabId` in background immediately — prevents the first signal request failing before any tick arrives.
- Extension invalidation guard: polls every 5s, shows a reload toast if the runtime context is invalidated after an extension update.

---

## Changelog

### v60001 (current)
- **Full pipeline moved to browser** — no more server-side feature engineering. Background.js now builds the complete (41, 22) AIROS V3 tensor and sends raw floats to the inference API.
- **AIROS V5 label** for the in-browser preprocessing stage (distinct from server-side AIROS V3 model weights).
- **Regime token** (V3 feature) prepended to every sequence at position 0.
- **Leak zeroing** on final bar cols [0, 3, 4, 14–21] — matches training Cell 5.
- **mom5 / mom14 / vol_std5** leak fix: removed ATR ratio division that V4 incorrectly applied (raw return, no normalization, matches Cell 3).
- **Tick bucket awareness** — tick_speed and tick_density normalized by actual bucket width.
- **Largest-frame binary selection** in ws_hook — prevents stale small frames from masking the history payload.
- **5-point OHLC expansion** (Format A) with inter-candle bridge — improves swing detection on candle-only data.
- **BROKER_TAB_PING** on content.js load — pins broker tab before first tick.
- Fixed: syntax error (double comma) on `engine_version` field.
- Fixed: `tick_speed` divisor corrected from `bucket_minutes × 15` to `bucket_minutes × 10` (matches Python Cell 4 exactly).
- Fixed: `tick_density` formula corrected from scaled `effective_max` to flat `tick_max_per_bucket` (matches Python Cell 4 exactly).
- Fixed: `rollingMax`, `rollingMin`, `rollingMean`, `rollingStd` now implement scipy `mode='nearest'` centered windows — eliminates distribution shift on `swing_h_dist`, `swing_l_dist`, and `compression` vs training.
- Fixed: version strings unified to `60001` across `manifest.json`, `background.js` console log, and `panel.html` title.

### v60000
- Initial AIROS V3 pipeline port to browser.
- Replaced ZigZag/structure token pipeline (v50000) with OHLC resample → candle/tick/regime feature stack.

### v50000
- ZigZag-based token pipeline (deprecated).

---

## Installation (Developer Mode)

1. Clone or unzip the extension folder.
2. Navigate to `chrome://extensions`.
3. Enable **Developer mode**.
4. Click **Load unpacked** and select the extension folder.
5. Navigate to a supported broker. The `⚡` overlay appears bottom-right.

## Notes

- The Render API server may cold-start (free tier). The first signal after a period of inactivity can take up to 30–45 seconds. Subsequent calls are fast.
- `tick_max_per_bucket = 210` assumes ~15 ticks/min × 14-min max bucket. Tune in `AIROS_CONFIG` if your instrument has significantly different tick density.
- `CANDLE_TARGET = 60` in `ws_hook.js` requests 1 hour of history. The pipeline uses only the last 40 bars; extra bars provide rolling-window warmup for ATR14, EMA21, and compression.
