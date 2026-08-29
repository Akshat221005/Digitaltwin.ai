# DigitalTwin.ai — Prototype Build Specification

**Audience:** an AI coding agent building this from scratch.
**Goal:** a working proof-of-concept for a hackathon Round 2 submission. Not production code.
**Read this whole file before writing anything.**

---

## 0. What we are building, in one paragraph

A digital twin of a mixed-model vehicle assembly line that unifies two problems plants
currently treat separately: bottlenecks (timing) and defects (quality). Both come from the
same root cause — a station behaving abnormally. The twin watches a small number of
richly-instrumented stations against manufacturer specification, senses the rest of the line
through barcode scan timestamps, **forecasts when a drifting signal will cross its spec
limit**, identifies which vehicles already on the line are carrying a latent defect, and
simulates the downstream consequences of doing nothing.

The prototype must demonstrate this on simulated data with known ground truth, and must
report honest validation metrics.

---

## 1. Terminology discipline — this is not optional

The system does four distinct things. The code, the UI labels, and the README must use these
words precisely and never blur them. This distinction is a graded part of the submission.

| Term | Meaning | Example output |
|---|---|---|
| **Detection** | An anomaly is happening *now*; we notice it. | "Weld current at S12 is 3.2% above nominal." |
| **Forecasting** | A signal is trending; we project when it crosses a limit. | "S12 current breaches tolerance in 4.5 h (95% CI: 3.1–6.8 h)." |
| **Inference** | A latent condition already exists but has not surfaced yet. | "63 vehicles passed S12 during the drift window and carry elevated leak risk." |
| **Simulation** | Running line dynamics forward to project consequences. | "If unfixed, buffer B3 starves at 11:15; projected stoppage 42 min." |

Never label detection as prediction. Never label inference as prediction. **Degradation
forecasting is the only true prediction in this system, and it is the centerpiece.**

---

## 2. Hard constraints

1. **Read-only.** The twin never writes to a PLC or line controller. It emits
   recommendations; a human executes them. Model this explicitly: recommendations are
   objects with a `status` field (`proposed` → `acknowledged` → `executed_by_human`).
   Enforced in code, stated in the README.
2. **No new hardware assumed.** Hard sensors exist only at designated stations. Everything
   else is inferred from barcode scan timestamps.
3. **Day-one value.** Spec-limit detection and degradation forecasting must work with zero
   historical data. Any learned model is a second-stage enhancement, never a prerequisite.
4. **Every prediction carries a confidence score** derived from the sensor coverage of the
   stations that contributed to it.
5. **Simulated, not hand-authored, data.** We need ground truth to validate against.

---

## 3. Anti-over-engineering rules

Follow these strictly. Scope discipline matters more than feature count here.

- Standard library and already-chosen dependencies before anything new.
- No database. Parquet and JSON files on disk.
- No Kafka, no message broker, no async framework. A time-stepped loop is correct.
- No deep learning. Gradient boosting is the ceiling, and it is optional (Stage 2).
- No authentication, no multi-tenancy, no Docker orchestration.
- No real OPC-UA client. A stub class that documents the interface is sufficient.
- No speculative abstraction. If there is one implementation, do not write an interface for
  it — the one exception is `DataSource` (§9), where the interface *is* the deliverable.

Dependencies, total: `numpy`, `pandas`, `scipy`, `pyarrow`, `streamlit`, `plotly`,
`scikit-learn` (Stage 2 only). Nothing else without a stated reason.

---

## 4. Repository layout

```
digitaltwin-ai/
├── README.md
├── ASSUMPTIONS.md          # every number we invented, with justification
├── requirements.txt
├── config/
│   ├── line.yaml           # station definitions
│   └── scenarios.yaml      # named demo scenarios
├── simulator/
│   ├── line.py             # station + buffer mechanics, time-stepped loop
│   ├── degradation.py      # wear processes that produce drift
│   ├── defects.py          # latent defect injection + delayed surfacing
│   └── run.py              # CLI: generate N days of data
├── twin/
│   ├── state.py            # live line state
│   ├── genealogy.py        # per-vehicle station history
│   └── confidence.py       # coverage-based confidence scoring
├── models/
│   ├── detect.py           # spec limits + EWMA control charts
│   ├── forecast.py         # ★ DEGRADATION FORECASTING — the centerpiece
│   ├── bottleneck.py       # active-period method
│   ├── infer.py            # latent defect inference via genealogy
│   ├── cascade.py          # forward simulation of consequences
│   └── learned.py          # OPTIONAL Stage 2 gradient boosting
├── validation/
│   ├── backtest.py
│   └── report.py           # writes metrics + plots to /docs
├── integration/
│   └── sources.py          # DataSource interface + Simulated/OPCUA stub
├── app/
│   ├── main.py             # streamlit entry, three tabs
│   ├── view_supervisor.py
│   ├── view_manager.py
│   └── view_leadership.py
├── data/                   # generated parquet (gitignored except a small sample)
└── docs/                   # architecture diagram, validation plots
```

---

## 5. The simulator

### 5.1 Line configuration (`config/line.yaml`)

40 stations across three zones:

- `body` — stations 1–15
- `paint` — stations 16–25
- `assembly` — stations 26–40

Each station:

```yaml
- id: S12
  name: "Body-shop weld station 3"
  zone: body
  sensor_tier: A          # A | B | C
  base_cycle_time_s: 58
  cycle_time_cv: 0.06     # coefficient of variation
  buffer_before: 8        # units
  signals:                # only for tier A/B
    weld_current_a:
      nominal: 240.0
      spec_low: 228.0
      spec_high: 252.0
      noise_sd: 1.8
  variant_cycle_multiplier:
    sedan: 1.0
    suv: 1.18
    van: 1.27
```

**Sensor tiers** — this distribution matters, it is what makes the "uneven coverage" story real:

- **Tier A (rich)** — 3 stations only: `S12` weld, `S19` paint, `S31` marriage. Full signal
  telemetry at every cycle. These are the stations the pitch models "deeply."
- **Tier B (partial)** — ~12 stations. One coarse signal each, sampled every 5th cycle.
- **Tier C (dark)** — the remaining ~25 stations. **Barcode entry/exit timestamps only.**
  No signal columns at all — they must be genuinely absent from the dataframe, not NaN-filled,
  so that any code path assuming they exist fails loudly.

Three vehicle variants (`sedan`, `suv`, `van`) in a repeating build sequence with
station-specific cycle multipliers. This is what makes the bottleneck *move* — a van-heavy
run shifts the constraint to a different station than a sedan-heavy run. Do not skip this;
"the bottleneck that shifts hour to hour" is a core claim of the pitch.

### 5.2 Line mechanics (`simulator/line.py`)

Time-stepped loop, 1-second ticks. Each station is `working` / `blocked` (downstream buffer
full) / `starved` (upstream buffer empty) / `down`. Standard buffer transfer. Log per-tick
station status — the bottleneck detector needs it.

Emit two streams:

**`events.parquet`** — one row per vehicle-station visit:

```
vin, station_id, variant, entry_ts, exit_ts, dwell_s,
operator_id, shift_id,
<signal columns, present only for tier A/B stations>
```

**`ground_truth.parquet`** — hidden from all models, used only by validation:

```
vin, defect_type, injected_at_station, injected_ts,
root_cause, surfaced_at_station, surfaced_ts, reworked
```

Plus **`degradation_truth.parquet`** — the ground truth for forecasting:

```
episode_id, station_id, signal, start_ts, drift_rate_per_hour,
actual_spec_crossing_ts, maintenance_ts, episode_type
```

### 5.3 Degradation processes (`simulator/degradation.py`)

This module produces the drift that §6 forecasts. It must generate a mix of episode types so
the forecaster's honesty is actually tested:

| Episode type | Behaviour | Purpose |
|---|---|---|
| `linear_wear` | Steady drift, e.g. weld electrode wearing at 0.4%/h | The easy case. Forecaster should nail it. |
| `accelerating_wear` | Quadratic drift, e.g. bearing degradation | Linear extrapolation will over-estimate remaining time. Must be visible in validation. |
| `step_change` | Sudden offset, e.g. a nozzle partially blocking | Not a degradation at all. The forecaster **must refuse to forecast** here (§6.5). |
| `noise_only` | No trend, just noise | The false-positive test. Forecaster must emit nothing. |
| `recovering` | Drift then maintenance reset | Tests changepoint reset. |

Also model, layered on top of station signals:
- Operator variation (per-operator dwell offset, fatigue increasing dwell late in shift)
- Incoming part batch quality (a batch-level quality scalar affecting defect probability)
- Ambient temperature/humidity cycles affecting paint stations

### 5.4 Latent defects (`simulator/defects.py`)

The defining mechanic of the whole demo:

- When a station's signal is out of spec — **or in the marginal band just inside spec** — a
  vehicle passing through gets a defect flag with some probability.
- The defect is **invisible** until the vehicle reaches its designated inspection station.
- Inspection stations: `S28` (body geometry), `S38` (paint finish), `S41` (leak/electrical
  end-of-line). A defect born at S12 does not surface until S41 — ~29 stations and roughly
  90 minutes later.
- The marginal-band case is important: it is the "each car individually passes inspection but
  the defect compounds silently" claim from the pitch. Implement it.

### 5.5 Scenarios (`config/scenarios.yaml`)

Named, seeded, reproducible runs:

- `nominal` — 30 days, background rates only. Used for training/baseline.
- `weld_drift_demo` — the video scenario. Clean start, `linear_wear` begins at S12 at
  09:00, crosses spec ~13:30, defects surface at S41 from ~10:30. **This must be seeded and
  byte-reproducible** so the demo video and the dashboard agree exactly.
- `noisy_line` — elevated noise and more `step_change`/`noise_only` episodes, for
  false-alarm-rate testing.

---

## 6. ★ Degradation forecasting (`models/forecast.py`)

**This is the most important module in the project. Build it carefully and give it the most
attention.** It is the only component that predicts an event rather than reporting one.

### 6.1 What it outputs

For each (station, signal) pair, on each update:

```python
@dataclass
class DegradationForecast:
    station_id: str
    signal: str
    current_value: float
    slope_per_hour: float
    slope_ci: tuple[float, float]      # 95% CI on the slope
    spec_limit: float                  # whichever limit it is heading toward
    hours_to_limit: float | None       # None if not forecastable
    hours_to_limit_ci: tuple[float, float] | None
    confidence: float                  # 0..1, see §8
    status: str                        # stable | drifting | forecastable | crossed | unstable
    reason: str                        # human-readable, shown in the UI
    n_samples: int
    window_start_ts: datetime
```

### 6.2 Algorithm

Keep it simple and statistically defensible. Do not reach for ML here — a robust linear fit
with an honest confidence interval is both stronger and easier to explain to a plant engineer.

1. **Buffer.** Rolling window of the last N readings per (station, signal). N = 300 cycles,
   or 4 hours, whichever is shorter. Ring buffer, no database.

2. **Changepoint reset.** Before fitting, check for a level shift in the window. Use a simple
   CUSUM or a two-sample comparison of the first and last thirds. If a shift is detected,
   truncate the window to post-shift samples only. This handles maintenance resets and tool
   changes — without it the forecaster produces nonsense after every repair.

3. **Smooth.** EWMA with α ≈ 0.2 to suppress per-cycle noise without lagging a real trend.

4. **Robust slope.** Theil–Sen estimator (`scipy.stats.theilslopes`) on the smoothed series
   against elapsed hours. Returns the slope plus a low/high CI directly. Theil–Sen is chosen
   because single-cycle sensor spikes are common on a real line and would wreck ordinary
   least squares.

5. **Significance gate.** If the slope CI includes zero, the signal is not meaningfully
   trending. Emit `status="stable"` and no forecast. **This gate is what keeps the false
   alarm rate down — it is the single most important line of code in the module.**

6. **Project to limit.**
   ```
   hours_to_limit = (spec_limit - current_smoothed_value) / slope
   ```
   Propagate the slope CI through the same expression to get the CI on time-to-limit.
   Note that the CI is asymmetric — a shallower slope means a much longer time — so compute
   both endpoints explicitly rather than assuming symmetry around the point estimate.

7. **Sanity gates.** Suppress the forecast (emit `status="unstable"` with a reason) if:
   - `n_samples < 60`
   - the residual spread after detrending exceeds a threshold (the fit does not describe the
     data)
   - `hours_to_limit > 72` (beyond useful planning horizon — report "drifting slowly", not
     a false-precision number)
   - a step change was detected in the last 30 samples

8. **Curvature check.** Fit slope separately on the first and second halves of the window.
   If the second-half slope materially exceeds the first, flag
   `reason="accelerating — linear estimate is optimistic"` and report the second-half-based
   estimate as a conservative lower bound. This is how the `accelerating_wear` episodes get
   handled honestly instead of silently mis-forecast.

### 6.3 What it must emit to the UI

Plain language, always with the uncertainty attached:

> **S12 weld current — 4.5 h to tolerance limit** (range 3.1–6.8 h)
> Drifting +0.4%/h since 09:00. Confidence: high (tier-A sensor, 280 samples).
> Recommend scheduling electrode service at the 14:00 shift change.

The recommendation should snap to the next scheduled break or shift change where possible —
that is what makes it operationally useful rather than merely alarming.

### 6.4 Validation of the forecaster — the headline metric

Because the simulator records `actual_spec_crossing_ts` in `degradation_truth.parquet`, the
forecast error is directly measurable. This is the strongest evidence in the submission.

Produce a **forecast error vs. lead time** curve:

- For every `linear_wear` and `accelerating_wear` episode, at every point where a forecast was
  emitted, record `error = predicted_crossing_ts - actual_crossing_ts` and the lead time
  remaining at that moment.
- Bucket by lead time (8h, 6h, 4h, 2h, 1h) and report median absolute error and the
  interquartile range per bucket.
- The expected and desirable shape: error shrinks as the crossing approaches. Say so.
- Report **CI coverage**: what fraction of the time did the true crossing fall inside the
  stated 95% interval? If it is near 95%, the uncertainty estimates are honest, and that
  claim is worth more than a low error number.

Also report:
- **False forecast rate** — forecasts emitted during `noise_only` episodes, per shift. Target
  under 0.5/shift.
- **Step-change refusal rate** — fraction of `step_change` episodes where the forecaster
  correctly declined to forecast. Target above 90%.

State all of these in the README as a table. Do not round them in your favour.

### 6.5 Do not forecast a step change

A blocked nozzle is not degradation. It is a discrete event. Extrapolating a trend through it
produces a confident, wrong number, and that is exactly the kind of false alarm that destroys
floor trust. The correct behaviour is `status="unstable"`, `reason="step change detected —
not a wear trend"`, and hand off to the detection layer instead. **Show this refusal happening
in the demo video** — a system that knows what it cannot predict is more persuasive than one
that predicts everything.

---

## 7. The other models

### 7.1 Detection (`models/detect.py`)
Spec-limit checks plus EWMA control charts on tier A/B signals. Zero training data. Emits
`Detection(station, signal, value, deviation_pct, severity, ts)`. Control charts because plant
engineers already trust them — this is the credibility floor.

### 7.2 Bottleneck (`models/bottleneck.py`)
Roser's active-period method: the momentary bottleneck is the station with the longest
uninterrupted active period. Roughly 20 lines over the per-tick status log. Add a short-horizon
buffer-trend projection ("S12 breaches takt in ~22 min at current trend"). Output a
station × time bottleneck-migration series — this becomes the manager heatmap.

### 7.3 Latent defect inference (`models/infer.py`)
Given a detection or forecast window at station X:
1. Query genealogy for all VINs that passed X during the window and have not yet reached
   their inspection station.
2. Assign each a risk score from deviation magnitude, dwell anomaly, and time-in-window.
3. Attach a confidence score (§8).
4. Output a VIN list with expected surfacing station and ETA.

This is the "63 vehicles are already carrying this" moment. Make the output a clean table —
it goes straight into the supervisor view and the video.

### 7.4 Cascade simulation (`models/cascade.py`)
Fork current line state, apply the projected degradation forward, run the simulator's own
mechanics at accelerated tick rate, and report: time to first starvation, projected stoppage
duration, units affected, rework cost. Reuse `simulator/line.py` — do not write a second
mechanics engine.

Also support a **counterfactual**: run with and without the proposed intervention and show
both. That comparison is what justifies the recommendation.

### 7.5 Learned model (`models/learned.py`) — OPTIONAL, build last
`HistGradientBoostingClassifier` on per-VIN genealogy features predicting downstream
inspection failure. Only attempt this if everything above is finished and validated. Frame it
in the README as the **Week 4+ stage** that layers onto day-one spec logic — this is what
resolves the Round 1 contradiction between "no training data needed" and "historical logs
train the model." If built, always show top feature contributions alongside any score.

---

## 8. Confidence scoring (`twin/confidence.py`)

Every detection, forecast, and inference carries a 0–1 confidence. Compute from:

- **Sensor tier** of the contributing station(s): A = 1.0, B = 0.6, C = 0.3
- **Sample sufficiency**: ramp from 0 to 1 over the required window size
- **Model agreement**: do spec check, control chart, and forecast agree? Agreement raises it.
- **Historical accuracy** for that station, if available

Surface it as High / Medium / Low with the numeric value on hover. **Never show a prediction
without it.** Being visibly honest about uncertainty is the mechanism that protects floor
trust, and it is an explicitly graded concern.

### 8.1 Retrofit ROI ranking

Rank tier-C stations by how much instrumenting them would reduce prediction uncertainty.
Simple, defensible approach: for each dark station, count defects historically back-attributed
to it (via genealogy correlation) weighted by rework cost, divided by estimated retrofit cost.
Output a ranked table: **"Instrument S07 and S23 in the next maintenance window; expected
confidence gain +18%, payback 4.2 months."**

This directly answers the "production can only pause during a few scheduled windows per year"
constraint. It is a small module with disproportionate credit — do not skip it.

---

## 9. Integration (`integration/sources.py`)

Minimal, honest, and mostly documentation:

```python
class DataSource(Protocol):
    def stream(self) -> Iterator[Event]: ...
    def health(self) -> dict: ...
```

Two implementations: `SimulatedSource` (real, used by everything) and `OPCUASource` (stub
that raises `NotImplementedError` with a docstring describing the real tag mapping, polling
interval, and edge-gateway placement). No writer interface exists anywhere in the codebase —
that absence is the point, and the README should say so explicitly.

Ship an architecture diagram in `/docs`: PLCs → read-only OPC-UA/MQTT tap → edge gateway →
twin → dashboards. Historian tap where available. No path back into line control.

---

## 10. Dashboard (`app/`)

Streamlit, three tabs, all reading the same twin state. Say that explicitly in the UI header —
"one model, three views" is a graded requirement.

**Supervisor (real-time).** Line map with color-coded stations. Active detections. **The
degradation forecast panel: countdown to spec crossing with its uncertainty band, per drifting
signal.** At-risk VIN table with expected surfacing station and ETA. One concrete action per
alert. Recommendation cards with an "Acknowledge / Mark executed" control — never
"Auto-apply."

**Plant manager (planning).** Weekly OEE. Bottleneck migration heatmap (station × hour) —
visually striking and proves the bottleneck moves. Rework cost trend. Maintenance window
planner fed by forecast horizons. Sensor retrofit ranking table.

**Leadership (investment case).** ROI model with sliders (units/year, defect rate reduction,
rework cost/unit, throughput gain, license cost) — every input adjustable, nothing hardcoded.
Payback period. Multi-site rollout scenario across three plants with different sensor maturity.
Phased roadmap.

Keep styling minimal. A clear, uncluttered dashboard beats a decorated one.

---

## 11. Build order

Strictly sequential. Do not start a step before the previous one runs.

1. `config/line.yaml` + `simulator/line.py` — line runs, events emitted, buffers behave.
   Sanity check: throughput and station utilisation are plausible.
2. `simulator/degradation.py` + `defects.py` — drift and latent defects appear in ground truth.
   Sanity check: a defect injected at S12 surfaces at S41 with a realistic delay.
3. `twin/genealogy.py` — can reconstruct any VIN's full station history.
4. **`models/forecast.py`** — the centerpiece. Get it right before moving on.
5. `validation/backtest.py` for the forecaster specifically — produce the forecast-error-vs-
   lead-time curve and the CI coverage number. **If these numbers are bad, fix the forecaster
   before building anything else.**
6. `models/detect.py`, `bottleneck.py`, `infer.py`.
7. `twin/confidence.py` + retrofit ranking.
8. `models/cascade.py`.
9. `app/` — three views.
10. `integration/sources.py`, architecture diagram.
11. `README.md`, `ASSUMPTIONS.md`, validation report.
12. `models/learned.py` only if time remains.

---

## 12. `ASSUMPTIONS.md` — required

Every invented number, listed with its basis and its sensitivity. Including at minimum:
units/hour, rework cost per unit, downtime cost per hour, defect base rate, drift rates,
retrofit cost per station, maintenance window frequency.

Any figure taken from a pitch deck rather than a source must be labelled as an assumption, not
presented as a fact. If a headline claim (e.g. cost reduction percentages) cannot be sourced,
restate it as a modelled output of these assumptions with the model shown. The brief explicitly
invites stated assumptions, so this file earns credit rather than costing it.

---

## 13. `README.md` — required contents

- One-paragraph problem framing (bottleneck and defect share a root cause).
- The four-term table from §1, verbatim. Lead with it.
- Architecture diagram.
- **Validation results table** — forecast error by lead-time bucket, CI coverage, false
  forecast rate, step-change refusal rate, defect inference precision/recall, mean lead time,
  false alarms per shift.
- Read-only stance, stated plainly.
- How to run: generate data, run backtest, launch dashboard. Three commands, copy-pasteable.
- Known limitations. Include the honest ones: it is simulated data; linear extrapolation
  under-performs on accelerating wear; tier-C inference is statistical, not causal.
- Link to the demo video.

---

## 14. Demo video — what the prototype must be able to show

Build so that this five-minute sequence is possible live, with real timestamps from the
`weld_drift_demo` scenario:

1. Line running normally.
2. **09:14** — S12 weld current drift detected. EWMA chart breaks trend.
3. **09:22** — forecast appears: crosses tolerance in 4.5 h (3.1–6.8 h), confidence high.
   Recommend electrode service at the 14:00 shift change. *This is the moment the pitch is
   about — hold on it.*
4. Genealogy panel: 63 VINs already downstream carrying elevated risk, surfacing at S41 from
   ~10:32.
5. Cascade sim: without intervention, buffer starvation at 11:15, 42-minute stoppage.
6. Recommendation issued → supervisor acknowledges → marked executed by human. Narrate: the
   twin never touched the PLC.
7. Contrast: end-of-line test would have caught the first failure at 10:32. We flagged it at
   09:14. **78 minutes of lead time, 63 vehicles protected.**
8. Cut to a `step_change` episode where the forecaster **declines** to forecast. Thirty
   seconds. It will be more persuasive than any of the successes.
9. Validation slide: the forecast-error-vs-lead-time curve and CI coverage.
10. Manager view heatmap and retrofit ranking, briefly.

---

## 15. Out of scope — do not build

Kafka or any broker. A database. Deep learning. Real OPC-UA connectivity. Authentication.
Multi-tenancy. Docker Compose. A REST API. Mobile layout. Dark mode. Unit test coverage beyond
the simulator invariants and the forecaster's gates. Any write path to line control equipment.

If a feature is not named in this document, it is out of scope.
