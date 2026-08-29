# ASSUMPTIONS.md

Every invented number in this prototype, with its basis and sensitivity. This file exists so
that no modelled output is mistaken for a sourced fact. Anything not listed here but appearing
in the README as a headline claim should be treated as a bug — flag it.

## Line topology

- **41 stations, not 40.** BUILD_SPEC.md names zones "body 1-15, paint 16-25, assembly 26-40"
  (40 stations) but also names `S41` as the leak/electrical end-of-line inspection station.
  Those two statements are inconsistent. Resolved by extending the assembly zone to 26-41
  (16 stations), giving 41 stations total. This does not affect any modelling logic — it only
  changes station counts in loops and configs.
- **Tier B station selection (12 stations):** `S03, S07, S09, S16, S21, S23, S26, S28, S33,
  S36, S38, S41`. Chosen to (a) spread tier-B coverage across all three zones, and (b) make the
  three named inspection stations (S28, S38, S41) tier-B by default, since a real plant would
  minimally instrument its inspection points. This is a design choice, not a source-backed
  number — sensitivity: if inspection stations were tier-C instead, defect surfacing would be
  detected only via barcode timestamps and inference, which is a strictly harder (but still
  workable) version of the demo.
- **Base cycle time (55-76s) and CV (0.06):** invented, loosely modeled on typical automotive
  line takt times (45-90s per station is common in body/paint/trim). Not sourced.
- **Buffer sizes (8-10 units):** invented, chosen to be large enough that transient variation
  doesn't constantly starve/block stations, small enough that a real bottleneck still creates
  visible blocking within the demo's simulated time horizon (hours, not days).
- **Variant cycle multipliers (sedan 1.0, suv 1.18, van 1.27):** invented. Meant to be large
  enough to visibly shift the bottleneck between zones on variant-mix changes, per BUILD_SPEC
  §5.1's requirement that the bottleneck genuinely moves.

## Signals (tier A/B)

- **Weld current (S12): nominal 240A, spec 228-252A (±5%), noise sd 1.8A** — taken directly
  from BUILD_SPEC.md's own example. Treated as sourced-by-spec, not invented.
- **Paint flow (S19) and press force (S31):** invented plausible values with ~±5-12% spec
  bands, matching the tightness of the S12 example.
- **Tier-B coarse signals (torque_nm, film_thickness_um, clamp_pressure_bar):** invented,
  one per zone, loosely representative of what a single coarse sensor would measure at a
  body/paint/assembly station respectively.

## Degradation (simulator/degradation.py)

- **Episode magnitudes** (drift rate, acceleration, step offset) are scaled to each signal's
  own spec span (e.g. linear_wear rate = 3-12% of span per hour) rather than fixed absolute
  numbers, so the same code produces sensible episodes regardless of a signal's units.
  Invented, not sourced.
- **Background episode rate (0.15 episodes/signal/day default):** invented, tuned so a 30-day
  `nominal` scenario produces a handful of drift/step/noise episodes per instrumented signal —
  enough for the forecaster's false-positive and step-change-refusal validation (§6.4) to have
  a non-trivial sample size without every signal being perpetually mid-episode.

## Latent defects (simulator/defects.py)

- **Defect-surfacing inspection station mapping:** BUILD_SPEC.md's own demo narrative (§5.4,
  §14) has a weld defect at S12 (body zone) surface at S41 (leak/electrical EOL), not at S28
  (body geometry) as a naive zone-based rule would predict. Resolved with explicit overrides
  for the three tier-A signals (S12→S41, S19→S38, S31→S28) matching the spec's own examples;
  everything else (tier-B signals, tier-C background defects) falls back to a zone default
  (body→S28, paint→S38, assembly→S41). This is a narrative/design choice, not a physical model
  of failure modes.
- **Defect probabilities (out_of_spec 0.35, marginal_band 0.08, background 0.001 defaults):**
  invented. Sensitivity: these directly set the defect base rate and inference precision/recall
  numbers reported later — treat any headline "N vehicles at risk" count as a function of these
  assumptions, not a measured plant statistic.
- **Detection probability at inspection (0.9 default):** invented — models imperfect end-of-line
  testing so some defects escape (reworked=False) even after reaching inspection, which is what
  makes "we caught it 78 minutes earlier than EOL testing would have" a meaningful claim rather
  than a tautology.
- **Observed S12→S41 surfacing delay ≈ 46 minutes** in this config's cycle times (29 stations,
  ~55-80s base cycle each). BUILD_SPEC.md's own illustrative example says "~90 minutes" — that
  number was never sourced either; ours is a direct consequence of the invented cycle times in
  `config/line.yaml`, not an attempt to hit the spec's figure.

## Forecaster (models/forecast.py) — empirically tuned parameters

Two gate parameters were tuned against synthetic pure-noise and step-change trials
(scripts not shipped; results reproduced by validation/backtest.py once built) to hit the
BUILD_SPEC.md §6.4 targets, rather than picked from a source:

- **Theil-Sen CI level: 0.99, not the textbook 0.95.** The slope significance gate runs on
  every tick against a heavily overlapping (autocorrelated) rolling window, not as one
  independent trial per window — a 95% CI let roughly 2.5 false "trending" episodes/shift
  through on pure noise. 0.99 alone was insufficient; combined with the persistence gate
  below it brings the measured rate to ~0.38/shift (target < 0.5/shift).
- **Persistence gate: 10 consecutive significant, same-direction evaluations required
  before a forecast surfaces** (`PERSISTENCE_TICKS` in forecast.py). This is a standard
  monitoring-system debounce, not part of BUILD_SPEC.md's literal algorithm — added because
  alpha alone couldn't reach the false-forecast target. Costs a few minutes of lead time on
  a true drift (negligible against a multi-hour horizon); Test 1 in the smoke suite confirms
  real drift is still detected and converges correctly as the crossing approaches.
- **Changepoint detection threshold: shift > 1.3× pooled within-group std** (not the 2×
  first drafted). Tuned so the step-change refusal rate clears the >90% target across step
  magnitudes matching `degradation.py`'s own random step_change episodes (15-40% of a
  signal's spec span). Measured: 92% refusal rate.
- **Theil-Sen is fit on raw readings, not the EWMA-smoothed series**, despite BUILD_SPEC.md
  §6.2 step 4 literally saying "on the smoothed series." Fitting on EWMA output introduces
  serial correlation that artificially narrows Theil-Sen's CI, which was the dominant cause
  of the false-positive problem above — feeding it smoothed data undermines exactly the
  guarantee §6.2 step 5 calls "the single most important line of code in the module." The
  EWMA is still used conceptually (as the "current level" for projection, via the fitted
  trend line's endpoint) but does not feed the significance test.

- **CI includes a level-uncertainty term, not just slope uncertainty.** BUILD_SPEC.md §6.2
  step 6 describes propagating only the slope CI through to hours-to-limit. Doing only that
  measured 64% coverage against ground truth (target ~95%) — the interval collapsed to near
  zero right as the crossing approached, because slope uncertainty shrinks with more data even
  though uncertainty about the current level doesn't vanish. Added the standard regression
  prediction-interval term (leverage-scaled residual spread at the query point, using the same
  z as the slope CI) converted to an hours-margin via the slope. This raised measured coverage
  to 91-97% across two independent backtest runs (3-day and 7-day synthetic scenarios) —
  reported as measured, not tuned to hit exactly 95%.

## Bottleneck (models/bottleneck.py)

- **Buffer-trend projection uses blocked-time fraction as a proxy for buffer occupancy.**
  `simulator/line.py` logs station status intervals (working/blocked/starved/down), not
  literal buffer fill levels over time. Since a station blocks more often as its downstream
  buffer fills, the trend in blocked-time fraction over a trailing window is used as a
  practical stand-in for "buffer filling toward capacity." This is a proxy, not a measurement
  of the actual buffer state — flagged in case a future pass wants to log buffer length
  directly instead.

## Confidence & retrofit ROI (twin/confidence.py)

- **Confidence weights (tier 0.40, sample sufficiency 0.25, model agreement 0.20, historical
  accuracy 0.15):** invented. Tier is weighted heaviest since it's the most directly
  observable factor (we always know a station's sensor tier; we don't always have model
  agreement or history yet on day one).
- **Historical accuracy defaults to a neutral 0.5** when no track record exists for a station,
  so day-one confidence never requires historical data, per the hard constraint in §2.
- **Rework cost per unit: $150; retrofit cost per station: $10,000-$25,000 (varies by
  station).** Invented placeholders for the retrofit ROI ranking formula. These are exactly
  the kind of number `models/learned.py`/the ROI dashboard sliders let a user override — any
  headline payback-period claim is a function of these, not a sourced plant statistic.

## Simulator RNG architecture — a bug that silently invalidated cascade counterfactuals

- **One shared RNG stream per LineSimulator meant forked runs diverged everywhere, not just
  downstream of the actual perturbation.** Building the cascade projection feature (below)
  surfaced this: comparing a "do nothing" fork against a "baseline" fork of the same sim showed
  nearly all 41 stations as "affected" — including ones physically upstream of the failing
  station, which should be impossible. Root cause: `_cycle_time`, `_sample_signals`, and the
  defect-injector call all drew from one shared `self.rng`. Skipping one station's draw for one
  tick (going down) shifts every later draw for every OTHER station's position in the same
  sequence, so two forked runs diverge almost immediately for reasons unrelated to real
  downstream causality — this also means the cascade counterfactual numbers reported earlier
  in this project (units_protected=20, rework_cost_avoided=$13,950) were partly contaminated by
  this artifact, not a clean causal estimate.
  - **Fixed with one independent RNG stream per station** (`self.station_rngs`), so a
    perturbation to one station can only affect others through the real buffer/blocking
    mechanics, not through shared randomness. Verified: after the fix, a cascade fork correctly
    excludes stations upstream of a failure from the "affected" set.
  - **First fix attempt broke byte-reproducibility.** Seeded each station's RNG with
    `random.Random((seed, station_id))` — a tuple containing a string. `random.Random` calls
    `hash()` on non-int/bytes seeds, and Python randomizes string hashing per-process
    (`PYTHONHASHSEED`) by default, so the same scenario produced different data on every
    separate run (deterministic *within* one process, not *across* processes — exactly the kind
    of bug that looks fine in a single dev session and breaks in CI or a fresh terminal).
    Fixed with `hashlib.sha256(f"{seed}:{station_id}")`, which is stable regardless of hash
    randomization. Re-verified full dataframe equality across two separate process runs before
    trusting it again.
  - **All three scenario datasets were regenerated** after this fix (the exact simulated values
    changed, though still fully deterministic); validation was rerun and the headline numbers
    stayed in the same range (e.g. CI coverage 74.7%→79.3%), confirming no regression.

## Cascade simulation (models/cascade.py)

- **A cascade projection can legitimately mark most of the downstream line "affected", and
  that's a finding, not a bug.** With small buffers (8-10 units) and ~60-90s cycle times, a
  45-minute stoppage drains every downstream station's buffer well within a few hours — on
  `weld_drift_demo`, an S12 failure genuinely disrupts all 32 stations from S10 through S41
  (confirmed via baseline comparison, not just "ever blipped blocked/starved" — see the RNG fix
  above for why that distinction required a real methodology, not just a threshold). Resist the
  urge to visually shrink this down to look tidier — a JIT line with small buffers really does
  cascade a single-station failure across most of what's downstream of it, and that's exactly
  the kind of consequence the forecaster's lead time exists to prevent.

- **"Time to first starvation" is measured only among downstream stations**, since the
  BUILD_SPEC.md example ("buffer B3 starves") describes a downstream buffer running dry once
  its feeding station goes down — an upstream station going down causes *blocking* further
  upstream, not starvation, so it's a different (also real) consequence not double-counted here.
- **An "intervention" both takes the station offline AND resets the underlying degradation
  episode** (`reset_signal` in `run_single_cascade`). A first version only modeled the
  mechanical down-time, which meant the drifting signal kept getting worse identically in both
  the do-nothing and with-intervention branches — the comparison never showed any rework-cost
  benefit from intervening, which is wrong: a real maintenance visit fixes the electrode, not
  just the schedule. Confirmed by rerunning the smoke test before/after: rework cost avoided
  went from $0 (bug) to $13,950 (fixed) on the same scenario.

## Forecaster — two more real bugs found during full-scenario validation

Isolated synthetic tests (single signal, hand-fed readings) missed two structural bugs that
only showed up once `validation/report.py` ran the forecaster against the full generated
`noisy_line` scenario (real tier mix, real sampling cadence, real mixed episode types):

- **`WINDOW_MAX_HOURS=4.0` made tier-B forecasting structurally impossible.** Tier-B signals
  are sampled 1-in-5 cycles (~5-6 min/sample); a 4-hour window can only ever hold ~35-45
  samples, permanently below `MIN_SAMPLES=60` no matter how long a real drift continues —
  older readings age out of the window as fast as new ones arrive. Every linear_wear episode
  on a tier-B station in `noisy_line` produced zero forecasts before crossing, for this reason.
  Raised to 8 hours (comfortably covers tier B's ~5.8h-to-60-samples cadence; tier A/C are
  unaffected since `WINDOW_MAX_SAMPLES=300` binds first for them well under 8h).
- **The curvature correction (§6.2 step 8) understated real acceleration.** Substituting the
  second-half's own *locally-linear* slope, as first implemented, still under-predicts a
  genuinely accelerating process — the curve keeps steepening past even the recent local rate.
  Measured on `noisy_line`'s real accelerating_wear episodes, this left forecasts 3-8h
  *optimistic* at 5-7h lead time (predicted crossing later than actual — the dangerous
  direction, telling a plant engineer they have more time than they do) on 82% of forecasts.
  Replaced with an explicit curvature estimate (rate of change of slope between the two half-
  windows) extrapolated as a quadratic from "now" — the same closed form
  `simulator/degradation.py` uses for ground truth. This cut mean error from 2.98h to 0.40h and
  eliminated the directional bias (50/50 over/under afterward, from 82% optimistic).

**Final measured numbers (on `noisy_line`, the scenario with the fullest episode-type mix —
see `docs/validation_metrics.md` for the up-to-date run):** 95% CI coverage ~67-75% (short of
the ~95% target — accelerating_wear's curvature-driven uncertainty is not fully propagated into
the interval; the point estimates are honest and no longer directionally biased, but the stated
interval is still narrower than it should be for that episode type specifically); false forecast
rate ~0.01/shift (well under the <0.5 target); step-change refusal ~84% (short of the >90%
target on this larger, noisier scenario — see below). These are reported as measured, not
adjusted to look better. A future pass could: propagate curvature-estimate uncertainty into the
CI directly (rather than reusing the linear-fit margin as a proxy), and/or report CI coverage
separately by episode type rather than blended, since linear_wear and accelerating_wear have
different, and differently-honest, uncertainty characteristics.

## Defect inference validation (validation/report.py)

- **Evaluated at the spec-crossing moment, not end-of-run.** An earlier version snapshotted
  at the very end of the generated scenario, which measures only the handful of vehicles still
  mid-transit when the simulation happens to stop — a near-arbitrary trailing cohort, not a
  realistic "flag them now" moment. Snapshotting at the actual spec-crossing timestamp (matching
  the demo narrative) is representative of how a supervisor would actually use this.
- **Measured precision ~12%, recall ~80%** on `weld_drift_demo`. This is an honest property of
  a risk-based flag, not a defect: the inference casts a wide net over everyone who passed the
  drifting station in the window (32 flagged), while only a handful (5) actually picked up a
  real defect, because defect probability is itself stochastic even in the marginal/out-of-spec
  band (see `simulator/defects.py`'s probabilities). High recall (catches 80% of real defects)
  at the cost of low precision is the deliberately safer failure mode for a quality-inspection
  aid — over-flagging for inspection is cheap; missing a real defect is not. Reported as
  measured per BUILD_SPEC.md §6.4's "do not round in your favour" instruction.

## Learned model (models/learned.py, optional Stage 2)

- **Test AUC ~0.57 on the `nominal` scenario** (a 20k-VIN subsample) — modestly above the 0.5
  chance baseline, not strong. This is consistent with, not contradicted by, the earlier
  finding that defect inference has ~12% precision: `simulator/defects.py`'s injection
  probability is substantially stochastic even within the out-of-spec/marginal bands by
  design (matching the real-world claim that "each car individually can pass inspection while
  the defect compounds silently" — not every deviation deterministically causes a defect). A
  model trained on this ground truth should not be expected to score much better than this
  without also modelling that irreducible randomness explicitly. Reported as measured, per
  BUILD_SPEC.md §7.5's framing of this module as an optional, lowest-priority enhancement.

## Detection (models/detect.py) — runs-rule fix

- **EWMA control chart requires 2 consecutive same-direction breaches before a detection is
  `confirmed`, not 1.** A fixed 3-sigma chart has a small but real single-point false-alarm
  rate; run across 14 instrumented signals over thousands of ticks in `weld_drift_demo`, that
  produced two isolated single-tick false alarms (S31 press_force_kn at 07:51, S19
  paint_flow_rate at ~10:01) on stations with zero scheduled anomaly. Verified this was a
  genuine statistical fluctuation, not a warm-up formula bug (the control limit had already
  converged to steady-state by n=53).
  - **First attempt fully suppressed unconfirmed single-tick breaches** (no `Detection` object
    emitted at all below the streak threshold). Revised after feedback: hiding a real spike
    entirely is its own kind of dishonesty, and a real plant does see transient blips that
    aren't sustained wear. `Detection` now carries a `confirmed: bool` field — the chart emits
    on the *first* breach, `confirmed=False`, and only flips to `True` once
    `CONSECUTIVE_BREACHES_REQUIRED` consecutive same-direction breaches accumulate. Unconfirmed
    detections are still shown in the dashboard's Active Detections list (marked distinctly),
    but only confirmed ones are eligible to drive a "investigate now" recommendation card — the
    runs-rule gates what's *actionable*, not what's *visible*. Spec-limit breaches
    (`spec_limit_check`) are always `confirmed=True`: a hard physical spec violation on one
    reading is a fact, not a statistical inference needing persistence.
  - Cost of the persistence requirement on the real anomaly: ~18 extra minutes before S12's
    drift is first *confirmed* (09:38→09:56) — though the first two ticks at 09:38 do show up
    immediately as unconfirmed, so nothing is hidden even during that window.

## To be added as later phases are built

Units/hour, rework cost per unit, downtime cost per hour, retrofit cost per station,
maintenance window frequency — added when cascade simulation and the confidence/ROI modules
are built.
