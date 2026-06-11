"""Logit-space blending of the model, EloMOV, and bookmaker probabilities.

The production home-win probability is a convex combination **in log-odds
space** of three signals that are all available in a prediction-time feature
row:

- ``p_model`` — the loaded artefact's output (XGBoost in production).
- ``p_elo_mov`` — the ``elo_mov_home_win_prob`` feature (MOV-weighted Elo).
- ``p_market`` — the ``odds_home_win_prob`` feature (de-vigged bookmaker
  line), absent when ``missing_odds`` is set.

Why these weights: walk-forward evaluation on the 2023–2026 baseline
(692 predictions, full closing-line coverage) showed fixed logit-space
weights beat every per-round-refit meta-learner, and the optimum is flat
across market weights 0.3–0.6 (accuracy 0.643–0.646, log-loss
0.634–0.641). The chosen point (market 0.40, EloMOV 0.36, model 0.24)
scored accuracy 0.6445 / log-loss 0.6380 / brier 0.2232 versus the previous
production behaviour (linear 0.3 market shrink) at 0.6431 / 0.6534 / 0.2298.
Logit-space mixing is the correct geometry for combining probabilities —
linear mixing systematically under-weights confident signals near 0/1.

When the market probability is unavailable the remaining weights are
renormalised (model 0.4 / EloMOV 0.6), so unpriced fixtures degrade to a
model+rating blend rather than the raw model output.
"""

from __future__ import annotations

import math

MODEL_BLEND_WEIGHT = 0.24
ELO_MOV_BLEND_WEIGHT = 0.36
MARKET_BLEND_WEIGHT = 0.40

# Guard band for logit(); probabilities are clipped into (eps, 1-eps) so a
# degenerate base (exactly 0.0 or 1.0) can't produce an infinite log-odds.
_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def blend_home_win_prob(
    model_prob: float,
    elo_mov_prob: float | None,
    market_prob: float | None,
) -> float:
    """Combine the three home-win probability signals in logit space.

    ``elo_mov_prob`` / ``market_prob`` may be ``None`` when the corresponding
    signal is unavailable; the remaining weights are renormalised so the
    output is always a convex logit-space combination of what's present.
    """
    parts: list[tuple[float, float]] = [(MODEL_BLEND_WEIGHT, model_prob)]
    if elo_mov_prob is not None:
        parts.append((ELO_MOV_BLEND_WEIGHT, elo_mov_prob))
    if market_prob is not None:
        parts.append((MARKET_BLEND_WEIGHT, market_prob))
    total = sum(w for w, _ in parts)
    z = sum(w * _logit(p) for w, p in parts) / total
    return _sigmoid(z)
