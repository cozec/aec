# TODOS

## AECMOS / subjective-proxy evaluation column
- **What:** Run Microsoft's AECMOS (neural MOS estimator) over all algorithm outputs; add a perceptual-quality column to the results table.
- **Why:** The AEC-Challenge's real leaderboard metric is subjective MOS; all current repo metrics are signal-level. An AECMOS column connects the table to how the field actually scores AEC.
- **Pros:** Strongest interview credibility; cheap — inference over wavs the pipeline already saves.
- **Cons:** Another external dependency; AECMOS has its own reliability caveats.
- **Context:** Enhanced outputs `e` are saved per algorithm per clip by the neural-Kalman plan's pipeline. AECMOS consumes (mic, far-end, enhanced) triples. See design doc `~/.gstack/projects/cozec-aec/adam-main-design-20260718-181947.md`.
- **Depends on / blocked by:** Neural-Kalman plan steps 1–4 producing outputs. Nothing else.

## Nonlinear delay-jump variant
- **What:** Rerun the delay-jump sweep (`src/delay_jump.py`) on the 4 nonlinear (loudspeaker-distortion) eval scenarios.
- **Why:** Devices that produce delay jumps (cheap speakerphones) are the same ones with nonlinear speakers — the combined condition is the realistic worst case, and no published recovery curves exist for it.
- **Pros:** Extends the repo's novel contribution to the harder condition; reuses `delay_jump.py` unchanged.
- **Cons:** Linear-filter rows will mostly flatline (2–3 dB baselines), so the plot mainly differentiates WebRTC vs NKF; smaller n.
- **Context:** Needs the design doc's 13A screening + fetch-more loop pointed at nonlinear non-eval clips.
- **Depends on / blocked by:** Delay-jump benchmark (plan step 2) complete.
