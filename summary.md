# Summary — Classical AEC Baselines on Microsoft AEC-Challenge Synthetic Data

Date: 2026-07-16 (WebRTC AEC added 2026-07-17)

## Setup

- **Dataset:** [microsoft/AEC-Challenge](https://github.com/microsoft/AEC-Challenge)
  synthetic set (16 kHz, 10 s clips). 12 scenarios sampled from `meta.csv`
  (seed 42): 8 clean-linear (`is_farend_nonlinear=0`, no noise, pool of 490)
  and 4 nonlinear loudspeaker distortion (`is_farend_nonlinear=1`, no noise,
  pool of 1,960). All scenarios contain **double-talk** (near-end speech
  active ~30–50% of each clip, SER −10…+8 dB).
- **Algorithms:** [ewan-xu/pyaec](https://github.com/ewan-xu/pyaec) classical
  adaptive filters, all sized for ~256–512 ms echo-path coverage at 16 kHz.
  Time-domain RLS/APA/Kalman were excluded — O(N²) per sample is infeasible at
  N=4096 taps. Plus **WebRTC AEC** (full, `aec_type=2`, NS/AGC/VAD off) via
  [python-webrtc-audio-processing](https://github.com/xiongyihui/python-webrtc-audio-processing),
  built from source with macOS-arm64 patches (excluded `_win` sources; see
  `python-webrtc-audio-processing/setup.py`), run over 10 ms int16 frames on
  the same delay-aligned inputs.
- **Front-end:** cross-correlation bulk-delay estimator aligning far-end to
  echo (min per-2s-window lag, 64-sample margin) before filtering.
- **Metrics** (all from ground-truth echo, valid during double-talk):
  - **True ERLE** `10·log10(Σy² / Σr²)` with residual `r = e − (d − y)`;
    "steady" = second half of the clip. Combined measure — penalizes both
    residual echo and near-end damage.
  - **ST-FE ERLE**: classic ERLE `10·log10(Σd² / Σe²)` over far-end
    single-talk frames (echo active, near-end silent) — pure echo suppression.
  - **Near-end SDR**: `10·log10(Σs² / Σ(e−s)²)` over near-end-active frames —
    duplex quality (how intact near-end speech survives).
  - RTF = runtime / audio duration (M2-class CPU, single core).

## Results — means over 12 scenarios

Ordered by steady-state true ERLE (0 dB = no net improvement):

| Algorithm | True ERLE steady | ST-FE ERLE | Near-end SDR | RTF |
|---|---|---|---|---|
| FDAF (M=4096, mu=0.1) | **6.4** | 4.9 | 8.0 | 0.004 |
| FDKF (M=4096) | 4.6 | 3.1 | 6.2 | 0.005 |
| PFDKF (32×256) | 4.5 | 1.7 | 6.0 | 0.013 |
| WebRTC AEC (full) | 0.1 | **16.3** | −1.8 | 0.001 |
| NLMS (4096 taps, mu=0.05) | −2.3 | 0.8 | 0.1 | 0.12 |
| NLMS (4096 taps, mu=0.2) | −5.6 | −1.5 | −3.6 | 0.12 |
| PFDAF (32×256, mu=0.1) | −5.9 | −4.7 | 0.6 | 0.004 |

By scenario (steady true ERLE): FDAF 7.4 clean / 4.5 nonlinear;
PFDKF 4.2 / 5.0; WebRTC 0.2 / −0.0 (ST-FE ERLE 18.0 / 12.9).

Per-file details: `results/benchmark_per_file.csv`,
`results/webrtc_per_file.csv`, aggregated in `results/combined_summary.csv`.
Plots: `plots/erle_by_algorithm.png`, `plots/suppression_vs_quality.png`,
`plots/example_waveforms.png`.

## Findings

1. **This dataset is deliberately hostile to classical AEC.** The echo path
   includes a bulk delay of 237–1,873 samples (15–117 ms) that **jumps within
   each clip** (e.g. 1235→1809→1236 samples across 2 s windows), simulating
   real device buffering/clock behavior. Without delay alignment, the
   frequency-domain Kalman filters do not adapt at all (0.0 dB ERLE even on
   echo-only signals); with alignment they reach 4–6 dB.
2. **Double-talk divergence dominates the gradient filters.** NLMS reaches
   ~20 dB ERLE on echo-only signals but goes *negative* under double-talk —
   it corrupts near-end speech while mis-adapting. A double-talk detector (or
   step-size control) is mandatory for LMS-family filters; pyaec does not
   include one. The Kalman filters (FDKF/PFDKF) are inherently double-talk
   robust and never went negative.
3. **FDAF's win is partly an artifact of its conservatism** — with a single
   4096-sample block it adapts slowly, which accidentally protects it during
   double-talk. PFDKF is the most consistent across conditions (4.2 clean /
   5.0 nonlinear) and is the strongest classical baseline in the literature.
4. **Nonlinear distortion costs ~2–3 dB** for the linear filters, as expected —
   a linear FIR cannot model loudspeaker saturation (pyaec's nonlinear
   filters — Volterra/FLAF — are O(N²)-ish at this tap count and were out of
   scope).
5. **WebRTC AEC is a different animal: linear filter + nonlinear suppressor
   (NLP).** It delivers by far the strongest echo suppression (16.3 dB mean
   ST-FE ERLE, up to 25.8 dB per file, at RTF 0.001 — 4× faster than any
   pyaec filter) but its NLP gates/attenuates near-end speech whenever
   far-end is active, giving −1.8 dB near-end SDR (half-duplex behavior).
   The combined true-ERLE metric nets out near zero — the suppression gains
   and duplex damage cancel. `plots/suppression_vs_quality.png` shows the
   trade-off: pyaec's linear filters occupy the full-duplex/weak-suppression
   corner, WebRTC the strong-suppression/half-duplex corner. This duplex
   trade-off is exactly what the AEC-Challenge's subjective (MOS) evaluation
   was designed to capture. Practical notes: WebRTC needed the same delay
   alignment (or an accurate `set_system_delay`) to converge, and handled
   the dataset's nonlinear scenarios far better than the linear filters
   (12.9 dB ST-FE ERLE) since its suppressor doesn't rely on linear
   modeling alone.
6. **Context:** 4–7 dB true ERLE under continuous double-talk with a
   time-varying echo path is the realistic classical ceiling here, and is why
   this dataset exists — the AEC-Challenge winners are neural models (e.g.
   DTLN-aec, Kalman+DNN hybrids) that score far higher subjectively. Classical
   filters remain useful as the linear front-end stage of such hybrids.

## Reproduction

```bash
source .venv/bin/activate
python src/benchmark.py    # writes results/*.csv, logs/benchmark.log
python src/make_plots.py   # writes plots/*.png
```

Scenario audio (48 files, ~15 MB) was fetched per-file from
`https://media.githubusercontent.com/media/microsoft/AEC-Challenge/main/datasets/synthetic/...`
to avoid the full multi-GB Git-LFS download; fileids are in
`data/selected_scenarios.csv`.
