"""Benchmark WebRTC AEC (python-webrtc-audio-processing) on the same
AEC-Challenge scenarios and metrics as src/benchmark.py.

WebRTC's AEC = linear adaptive filter + nonlinear echo suppressor (NLP).
The NLP also attenuates near-end speech during far-end activity, which the
combined "true ERLE" metric penalizes — see the segment metrics for the
echo-suppression vs near-end-quality split.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from benchmark import DATA, RESULTS, SR, est_delay, erle_db, segment_metrics

from webrtc_audio_processing import AudioProcessingModule as AP

FRAME = SR // 100  # 10 ms


def run_webrtc(x, d, delay_ms=0):
    """Run WebRTC full AEC (aec_type=2), NS/AGC/VAD off, over 10 ms frames."""
    ap = AP(aec_type=2, enable_ns=False, agc_type=0, enable_vad=False)
    ap.set_stream_format(SR, 1)
    ap.set_reverse_stream_format(SR, 1)
    n = min(len(x), len(d)) // FRAME * FRAME
    to16 = lambda sig: np.clip(sig * 32768, -32768, 32767).astype('<i2')
    x16, d16 = to16(x[:n]), to16(d[:n])
    out = np.zeros(n, dtype=np.int16)
    for i in range(0, n, FRAME):
        ap.process_reverse_stream(x16[i:i + FRAME].tobytes())
        ap.set_system_delay(delay_ms)
        out[i:i + FRAME] = np.frombuffer(
            ap.process_stream(d16[i:i + FRAME].tobytes()), dtype='<i2')
    return out.astype(np.float64) / 32768


def main():
    scenarios = pd.read_csv(DATA / "selected_scenarios.csv")
    rows = []
    for _, sc in scenarios.iterrows():
        fid = int(sc.fileid)
        x, _ = sf.read(DATA / f"farend_speech_fileid_{fid}.wav")
        y, _ = sf.read(DATA / f"echo_fileid_{fid}.wav")
        d, _ = sf.read(DATA / f"nearend_mic_fileid_{fid}.wav")
        n = min(len(x), len(y), len(d))
        x, y, d = x[:n], y[:n], d[:n]
        nearend = d - y

        lag = est_delay(x, y)
        xs = np.concatenate([np.zeros(lag), x[:n - lag]])

        kind = "nonlinear" if sc.is_farend_nonlinear else "clean-linear"
        t0 = time.perf_counter()
        e = run_webrtc(xs, d, delay_ms=0)
        rt = time.perf_counter() - t0
        m = len(e)
        resid = e - nearend[:m]
        half = m // 2
        erle_stfe, near_sdr = segment_metrics(y[:m], nearend[:m], d[:m], e)
        rows.append({
            "algo": "WebRTC AEC (full)",
            "fileid": fid,
            "scenario": kind,
            "lag_samples": lag,
            "ser_db": sc.ser,
            "erle_full_db": erle_db(y[:m], resid),
            "erle_steady_db": erle_db(y[half:m], resid[half:]),
            "erle_stfe_db": erle_stfe,
            "nearend_sdr_db": near_sdr,
            "runtime_s": rt,
            "rtf": rt / (m / SR),
        })
        r = rows[-1]
        print(f"fileid={fid} {kind:<12} ERLE steady={r['erle_steady_db']:5.1f} dB  "
              f"ST-FE ERLE={r['erle_stfe_db']:5.1f} dB  "
              f"near-SDR={r['nearend_sdr_db']:5.1f} dB  RTF={r['rtf']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "webrtc_per_file.csv", index=False)
    print("\nmeans:")
    print(df.groupby("scenario")[["erle_full_db", "erle_steady_db",
                                  "erle_stfe_db", "nearend_sdr_db", "rtf"]]
            .mean().round(2).to_string())


if __name__ == "__main__":
    main()
