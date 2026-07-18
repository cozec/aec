"""Generate benchmark plots: ERLE summary bars and an example waveform figure."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"

# validated categorical palette (light mode): slot1 blue, slot2 green
BLUE, GREEN = "#2a78d6", "#008300"
INK, MUTED, GRID = "#33322e", "#6f6d66", "#e8e6e1"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "font.size": 11,
})


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def combined():
    """Per-file results of pyaec filters + WebRTC AEC, aggregated."""
    frames = [pd.read_csv(ROOT / "results" / "benchmark_per_file.csv")]
    wp = ROOT / "results" / "webrtc_per_file.csv"
    if wp.exists():
        frames.append(pd.read_csv(wp))
    df = pd.concat(frames, ignore_index=True)
    agg = (df.groupby(["algo", "scenario"])
             [["erle_full_db", "erle_steady_db", "erle_stfe_db",
               "nearend_sdr_db", "rtf"]].mean().round(2).reset_index())
    agg.to_csv(ROOT / "results" / "combined_summary.csv", index=False)
    return agg


def erle_bars():
    df = combined()
    order = (df[df.scenario == "clean-linear"]
             .sort_values("erle_steady_db").algo.tolist())
    y = np.arange(len(order))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for off, scen, color in [(h / 2 + 0.01, "clean-linear", BLUE),
                             (-h / 2 - 0.01, "nonlinear", GREEN)]:
        sub = df[df.scenario == scen].set_index("algo").loc[order]
        vals = sub.erle_steady_db.values
        ax.barh(y + off, vals, height=h, color=color, label=scen, zorder=3)
        for yi, v in zip(y + off, vals):
            ax.annotate(f"{v:.1f}", xy=(v, yi),
                        xytext=(4 if v >= 0 else -4, 0),
                        textcoords="offset points",
                        va="center", ha="left" if v >= 0 else "right",
                        fontsize=9, color=MUTED)
    ax.axvline(0, color=MUTED, lw=1, zorder=2)
    ax.set_yticks(y, order)
    ax.set_xlabel("Steady-state true ERLE (dB) — higher is better")
    ax.set_title("Classical AEC baselines (pyaec + WebRTC) on Microsoft AEC-Challenge synthetic data\n"
                 "mean over 8 clean-linear + 4 nonlinear double-talk scenarios, 16 kHz",
                 loc="left", fontsize=11)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "erle_by_algorithm.png", dpi=150)
    print("wrote plots/erle_by_algorithm.png")


def tradeoff_scatter():
    """Echo suppression (ST-FE ERLE) vs near-end quality (SDR) per algorithm."""
    df = combined()
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for scen, color in [("clean-linear", BLUE), ("nonlinear", GREEN)]:
        sub = df[df.scenario == scen]
        ax.scatter(sub.erle_stfe_db, sub.nearend_sdr_db, s=70, color=color,
                   label=scen, zorder=3, edgecolors=SURFACE, linewidths=2)
        for _, r in sub.iterrows():
            short = (r.algo.split(" (")[0]
                     .replace("NLMS", f"NLMS mu={'0.2' if '0.2' in r.algo else '0.05'}")
                     if r.algo.startswith("NLMS") else r.algo.split(" (")[0])
            if r.algo.startswith("WebRTC"):
                short = "WebRTC AEC"
            # nudge the green FDAF label down to avoid the blue PFDKF label
            off = (6, -12) if (scen == "nonlinear" and short == "FDAF") else (6, 5)
            ax.annotate(short, (r.erle_stfe_db, r.nearend_sdr_db),
                        xytext=off, textcoords="offset points",
                        fontsize=9, color=MUTED)
    ax.axhline(0, color=MUTED, lw=1, zorder=2)
    ax.axvline(0, color=MUTED, lw=1, zorder=2)
    ax.set_xlabel("ERLE on far-end single-talk segments (dB) — echo suppression")
    ax.set_ylabel("Near-end SDR during near-end activity (dB)")
    ax.set_title("Echo suppression vs near-end speech quality\n"
                 "top-right is better; WebRTC's NLP trades duplex quality for suppression",
                 loc="left", fontsize=11)
    ax.grid(color=GRID, zorder=0)
    ax.legend(frameon=False, loc="upper left", title=None)
    style(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "suppression_vs_quality.png", dpi=150)
    print("wrote plots/suppression_vs_quality.png")


def example_waveforms(fid=8830):
    sys.path.insert(0, str(ROOT / "pyaec"))
    np.complex = complex
    np.float = float
    from frequency_domain_adaptive_filters.fdaf import fdaf
    from benchmark import est_delay

    x, sr = sf.read(ROOT / f"data/farend_speech_fileid_{fid}.wav")
    y, _ = sf.read(ROOT / f"data/echo_fileid_{fid}.wav")
    d, _ = sf.read(ROOT / f"data/nearend_mic_fileid_{fid}.wav")
    n = min(len(x), len(y), len(d))
    x, y, d = x[:n], y[:n], d[:n]
    lag = est_delay(x, y)
    x = np.concatenate([np.zeros(lag), x[:n - lag]])
    e = np.asarray(fdaf(x, d, M=4096, mu=0.1))[:n]
    s = d - y  # near-end component in the mic (target)

    t = np.arange(n) / sr
    panels = [(d, f"Microphone (near-end + echo), fileid {fid}"),
              (e[:n], "FDAF output (echo cancelled)"),
              (s, "Near-end speech target (mic minus echo)")]
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 5.4), sharex=True, sharey=True)
    for ax, (sig, title) in zip(axes, panels):
        m = min(len(sig), n)
        ax.plot(t[:m], sig[:m], color=BLUE, lw=0.4)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", color=GRID, zorder=0)
        style(ax)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(PLOTS / "example_waveforms.png", dpi=150)
    print("wrote plots/example_waveforms.png")


def example_webrtc(fid=8830):
    """Mic / WebRTC output / near-end target, same layout as example_waveforms."""
    from benchmark import est_delay
    from webrtc_benchmark import run_webrtc

    x, sr = sf.read(ROOT / f"data/farend_speech_fileid_{fid}.wav")
    y, _ = sf.read(ROOT / f"data/echo_fileid_{fid}.wav")
    d, _ = sf.read(ROOT / f"data/nearend_mic_fileid_{fid}.wav")
    n = min(len(x), len(y), len(d))
    x, y, d = x[:n], y[:n], d[:n]
    lag = est_delay(x, y)
    xs = np.concatenate([np.zeros(lag), x[:n - lag]])
    e = run_webrtc(xs, d, delay_ms=0)
    s = d - y

    t = np.arange(n) / sr
    panels = [(d, f"Microphone (near-end + echo), fileid {fid}"),
              (e, "WebRTC AEC output (echo suppressed; note near-end gating)"),
              (s, "Near-end speech target (mic minus echo)")]
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 5.4), sharex=True, sharey=True)
    for ax, (sig, title) in zip(axes, panels):
        m = min(len(sig), n)
        ax.plot(t[:m], sig[:m], color=BLUE, lw=0.4)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", color=GRID, zorder=0)
        style(ax)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(PLOTS / "example_webrtc.png", dpi=150)
    print("wrote plots/example_webrtc.png")


if __name__ == "__main__":
    erle_bars()
    tradeoff_scatter()
    example_waveforms()
    example_webrtc()
