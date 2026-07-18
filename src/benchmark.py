"""Benchmark pyaec classical adaptive filters on the Microsoft AEC-Challenge
synthetic dataset.

For each scenario the synthetic dataset provides:
  farend_speech  x : loudspeaker (reference) signal
  echo           y : x after (optional nonlinearity and) room impulse response
  nearend_speech s : clean near-end talker
  nearend_mic    d : microphone signal = scale*s + y  (clean scenarios)

An adaptive AEC produces e = d - w*x.  Since the linear filter only touches
the x-path, the near-end component passes through unchanged and the true
residual echo is r = e - (d - y).  We report the true ERLE

    ERLE = 10*log10( sum(y^2) / sum(r^2) )

over the full clip and over the second half (steady state, after convergence).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# numpy 2.x removed these aliases; pyaec (2020) still uses them
np.complex = complex
np.float = float

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pyaec"))

from time_domain_adaptive_filters.nlms import nlms
from frequency_domain_adaptive_filters.fdaf import fdaf
from frequency_domain_adaptive_filters.fdkf import fdkf
from frequency_domain_adaptive_filters.pfdaf import pfdaf
from frequency_domain_adaptive_filters.pfdkf import pfdkf

DATA = ROOT / "data"
RESULTS = ROOT / "results"

SR = 16000

# ~256-512 ms echo path coverage at 16 kHz
ALGOS = {
    "NLMS (4096, mu=0.2)": lambda x, d: nlms(x, d, N=4096, mu=0.2),
    "NLMS (4096, mu=0.05)": lambda x, d: nlms(x, d, N=4096, mu=0.05),
    "FDAF (M=4096)": lambda x, d: fdaf(x, d, M=4096, mu=0.1),
    "FDKF (M=4096)": lambda x, d: fdkf(x, d, M=4096),
    "PFDAF (32x256)": lambda x, d: pfdaf(x, d, N=32, M=256, mu=0.1, partial_constrain=True),
    "PFDKF (32x256)": lambda x, d: pfdkf(x, d, N=32, M=256, partial_constrain=True),
}


def est_delay(x, y, w=2 * SR):
    """Estimate the bulk far-end -> echo delay in samples.

    The dataset applies a time-varying delay, so we take the minimum of
    per-window cross-correlation lags (with a small safety margin) and let
    the adaptive filter's taps cover the residual variation.
    """
    from numpy.fft import rfft, irfft
    lags = []
    for st in range(0, min(len(x), len(y)) - w, w):
        xe = x[st:st + w]
        if np.sqrt(np.mean(xe**2)) < 0.005:
            continue
        cc = irfft(rfft(y[st:st + w], 2 * w) * rfft(xe, 2 * w).conj())
        lags.append(int(np.argmax(np.abs(cc[:w]))))
    return max(0, min(lags) - 64) if lags else 0


def erle_db(echo, residual):
    """Echo return loss enhancement in dB (higher is better)."""
    return 10 * np.log10(np.sum(echo**2) / (np.sum(residual**2) + 1e-12) + 1e-12)


def segment_metrics(y, s, d, e, frame=320):
    """Segment-based AEC metrics from ground truth.

    Returns (erle_stfe_db, nearend_sdr_db):
      erle_stfe_db  - classic ERLE 10*log10(sum d^2 / sum e^2) over far-end
                      single-talk frames (echo active, near-end silent)
      nearend_sdr_db - 10*log10(sum s^2 / sum (e-s)^2) over near-end active
                      frames; penalizes both residual echo and near-end
                      suppression (relevant for NLP-based AECs like WebRTC)
    """
    m = min(len(y), len(s), len(d), len(e))
    nfr = m // frame
    yf = y[:nfr * frame].reshape(nfr, frame)
    sf_ = s[:nfr * frame].reshape(nfr, frame)
    df = d[:nfr * frame].reshape(nfr, frame)
    ef = e[:nfr * frame].reshape(nfr, frame)
    yr = np.sqrt(np.mean(yf**2, axis=1))
    sr_ = np.sqrt(np.mean(sf_**2, axis=1))
    thr = 0.005
    stfe = (yr > thr) & (sr_ < thr)
    near = sr_ > thr
    erle_stfe = (10 * np.log10(np.sum(df[stfe]**2) / (np.sum(ef[stfe]**2) + 1e-12) + 1e-12)
                 if stfe.any() else np.nan)
    sdr = (10 * np.log10(np.sum(sf_[near]**2) / (np.sum((ef[near] - sf_[near])**2) + 1e-12) + 1e-12)
           if near.any() else np.nan)
    return erle_stfe, sdr


def main():
    scenarios = pd.read_csv(DATA / "selected_scenarios.csv")
    rows = []
    for _, sc in scenarios.iterrows():
        fid = int(sc.fileid)
        x, sr = sf.read(DATA / f"farend_speech_fileid_{fid}.wav")
        y, _ = sf.read(DATA / f"echo_fileid_{fid}.wav")
        d, _ = sf.read(DATA / f"nearend_mic_fileid_{fid}.wav")
        n = min(len(x), len(y), len(d))
        x, y, d = x[:n], y[:n], d[:n]
        nearend = d - y  # scaled near-end speech as it appears in the mic

        # bulk delay alignment (standard AEC front-end, cf. WebRTC delay agnostic AEC)
        lag = est_delay(x, y)
        x = np.concatenate([np.zeros(lag), x[:n - lag]])

        kind = "nonlinear" if sc.is_farend_nonlinear else "clean-linear"
        for name, fn in ALGOS.items():
            t0 = time.perf_counter()
            e = np.asarray(fn(x, d))
            rt = time.perf_counter() - t0
            m = min(len(e), n)
            resid = e[:m] - nearend[:m]
            half = m // 2
            erle_stfe, near_sdr = segment_metrics(y[:m], nearend[:m], d[:m], e[:m])
            rows.append({
                "algo": name,
                "fileid": fid,
                "scenario": kind,
                "lag_samples": lag,
                "ser_db": sc.ser,
                "erle_full_db": erle_db(y[:m], resid),
                "erle_steady_db": erle_db(y[half:m], resid[half:]),
                "erle_stfe_db": erle_stfe,
                "nearend_sdr_db": near_sdr,
                "runtime_s": rt,
                "rtf": rt / (m / sr),
            })
            r = rows[-1]
            print(f"fileid={fid} {kind:<12} {name:<18} "
                  f"ERLE full={r['erle_full_db']:5.1f} dB  "
                  f"steady={r['erle_steady_db']:5.1f} dB  RTF={r['rtf']:.2f}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "benchmark_per_file.csv", index=False)

    summary = (df.groupby(["algo", "scenario"])
                 .agg(erle_full_db=("erle_full_db", "mean"),
                      erle_steady_db=("erle_steady_db", "mean"),
                      erle_stfe_db=("erle_stfe_db", "mean"),
                      nearend_sdr_db=("nearend_sdr_db", "mean"),
                      rtf=("rtf", "mean"))
                 .round(2)
                 .reset_index()
                 .sort_values(["scenario", "erle_steady_db"], ascending=[True, False]))
    summary.to_csv(RESULTS / "benchmark_summary.csv", index=False)
    print("\n=== Mean ERLE by algorithm and scenario ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
