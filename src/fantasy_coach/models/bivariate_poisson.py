"""Bivariate Poisson + Dixon-Coles correlated score-distribution model (#209).

Models home/away NRL scores as a bivariate Poisson (Karlis & Ntzoufras 2003):

    H = Z1 + Z3,  A = Z2 + Z3
    Z1 ~ Poisson(λ1),  Z2 ~ Poisson(λ2),  Z3 ~ Poisson(λ3)

The marginal rates are E[H] = λ1 + λ3 = λh and E[A] = λ2 + λ3 = λa;
Cov(H, A) = λ3 > 0.  The feature GLM parameterises λh, λa exactly as in
the Skellam model so that the Skellam coefficients provide a valid warm start.

Dixon-Coles correction (1997): the four near-zero cells {(0,0), (0,1),
(1,0), (1,1)} are multiplied by a correction factor φ(h, a, λh, λa, τ)
that counteracts the independence assumption at very low scores — the context
where independent-Poisson models systematically misfit.  τ is fit as part of
the MLE; for NRL where sub-6-point scores are rare it typically shrinks toward
zero.

Joint PMF:

    P(H=h, A=a | λ1, λ2, λ3) =
        exp(−(λ1+λ2+λ3)) · (λ1^h / h!) · (λ2^a / a!)
        · Σ_{k=0}^{min(h,a)}  C(h,k)·C(a,k)·k! · (λ3/(λ1·λ2))^k

References
----------
Dixon, M. J. & Coles, S. G. (1997). Modelling Association Football Scores.
    Applied Statistics, 46(2), 265–280.
Karlis, D. & Ntzoufras, I. (2003). Analysis of sports data by using bivariate
    Poisson models. J. Royal Statistical Society: Series D, 52(3), 381–393.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.special import logsumexp as _logsumexp

from fantasy_coach.feature_engineering import FEATURE_NAMES
from fantasy_coach.features import MatchRow
from fantasy_coach.models.elo import Elo

# NRL total-score truncation.  Teams very rarely exceed 80 points.
_H_MAX = 80
_A_MAX = 80

# Minimum positive rate to avoid log(0).
_MIN_RATE = 1e-6


# ---------------------------------------------------------------------------
# Core PMF
# ---------------------------------------------------------------------------


def _log_bp_pmf(h: int, a: int, l1: float, l2: float, l3: float) -> float:
    """Log P(H=h, A=a) under the bivariate Poisson parameterised by (l1, l2, l3).

    l1 = λh − λ3  (home exclusive rate)
    l2 = λa − λ3  (away exclusive rate)
    l3 = covariance parameter (λ3)
    """
    k = np.arange(min(h, a) + 1, dtype=float)
    # log C(h,k)·C(a,k)·k! · (l3/(l1·l2))^k
    log_terms = (
        gammaln(h + 1) - gammaln(h - k + 1) - gammaln(k + 1)
        + gammaln(a + 1) - gammaln(a - k + 1) - gammaln(k + 1)
        + gammaln(k + 1)
        + k * (np.log(l3) - np.log(l1) - np.log(l2))
    )
    return (
        -(l1 + l2 + l3)
        + h * np.log(l1)
        + a * np.log(l2)
        - gammaln(h + 1)
        - gammaln(a + 1)
        + float(_logsumexp(log_terms))
    )


def _dc_factor(h: int, a: int, lh: float, la: float, tau: float) -> float:
    """Dixon-Coles correction factor for cells h ≤ 1 and a ≤ 1."""
    if tau == 0.0 or (h > 1 or a > 1):
        return 1.0
    if h == 0 and a == 0:
        return max(1.0 - tau * lh * la, _MIN_RATE)
    if h == 0 and a == 1:
        return max(1.0 + tau * lh, _MIN_RATE)
    if h == 1 and a == 0:
        return max(1.0 + tau * la, _MIN_RATE)
    # h == 1, a == 1
    return max(1.0 - tau, _MIN_RATE)


# ---------------------------------------------------------------------------
# Score grid computation
# ---------------------------------------------------------------------------


def _score_grid(lh: float, la: float, l3: float, tau: float) -> np.ndarray:
    """Return the (H_MAX+1, A_MAX+1) joint PMF grid P(H=h, A=a).

    Uses vectorised numpy operations for the inner correlation sum to avoid
    nested Python loops.  Cells are renormalised after the Dixon-Coles
    correction.
    """
    l3c = min(l3, min(lh, la) * 0.9)  # ensure λ1, λ2 > 0
    l1 = max(lh - l3c, _MIN_RATE)
    l2 = max(la - l3c, _MIN_RATE)

    H = np.arange(_H_MAX + 1)
    A = np.arange(_A_MAX + 1)
    HH, AA = np.meshgrid(H, A, indexing="ij")  # shape (H+1, A+1)

    # Vectorise the inner correlation sum over k = 0..min(h, a).
    K_max = int(np.minimum(HH, AA).max())  # = _H_MAX (since H_MAX == A_MAX)
    K = np.arange(K_max + 1)

    # Broadcast shapes: K → (K+1, 1, 1), HH/AA → (1, H+1, A+1)
    K3 = K[:, None, None]
    HH3 = HH[None, :, :]
    AA3 = AA[None, :, :]

    valid = np.minimum(HH3, AA3) >= K3
    safe_h_k = np.maximum(HH3 - K3, 0)
    safe_a_k = np.maximum(AA3 - K3, 0)

    log_r = float(np.log(l3c) - np.log(l1) - np.log(l2))
    log_terms = np.where(
        valid,
        (gammaln(HH3 + 1) - gammaln(safe_h_k + 1) - gammaln(K3 + 1)
         + gammaln(AA3 + 1) - gammaln(safe_a_k + 1) - gammaln(K3 + 1)
         + gammaln(K3 + 1)
         + K3 * log_r),
        -np.inf,
    )
    log_corr_sum = _logsumexp(log_terms, axis=0)  # shape (H+1, A+1)

    log_grid = (
        -(l1 + l2 + l3c)
        + HH * np.log(l1)
        + AA * np.log(l2)
        - gammaln(HH + 1)
        - gammaln(AA + 1)
        + log_corr_sum
    )
    grid = np.maximum(np.exp(log_grid), 0.0)

    # Dixon-Coles correction on the four near-zero cells
    for h in range(2):
        for a in range(2):
            grid[h, a] *= _dc_factor(h, a, lh, la, tau)

    total = grid.sum()
    if total > 0:
        grid /= total
    return grid


# ---------------------------------------------------------------------------
# Training frame (reuses Skellam frame)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BivariateTrainingFrame:
    """Feature matrix with home + away score targets (same layout as Skellam)."""

    X: np.ndarray          # shape (n, n_features)
    y_home: np.ndarray     # int scores
    y_away: np.ndarray
    match_ids: np.ndarray
    start_times: np.ndarray
    feature_names: tuple[str, ...] = field(default_factory=lambda: FEATURE_NAMES)


def build_bivariate_frame(
    matches: list[MatchRow],
    *,
    elo: Elo | None = None,
) -> BivariateTrainingFrame:
    """Build a training frame from completed matches (includes draws)."""
    from fantasy_coach.models.skellam import build_skellam_frame

    sf = build_skellam_frame(matches, elo=elo)
    return BivariateTrainingFrame(
        X=sf.X,
        y_home=sf.y_home.astype(int),
        y_away=sf.y_away.astype(int),
        match_ids=sf.match_ids,
        start_times=sf.start_times,
        feature_names=sf.feature_names,
    )


# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreLine:
    home_score: int
    away_score: int
    probability: float


@dataclass(frozen=True)
class BivariatePrediction:
    """Full correlated score-distribution prediction for one match."""

    lambda_home: float
    lambda_away: float
    lambda3: float
    score_grid: np.ndarray   # shape (81, 81)
    home_win_prob: float
    draw_prob: float
    predicted_margin: float  # E[H − A]
    predicted_total: float   # E[H + A]
    top_scorelines: list[ScoreLine]  # top-5 most probable outcomes


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BivariatePoissonModel:
    """Bivariate Poisson model satisfying the standard Model protocol.

    Internally stores feature weights (home/away GLM) plus global parameters
    λ3 and τ (Dixon-Coles correction).  Satisfies ``Model`` via
    ``predict_home_win_prob``.
    """

    def __init__(
        self,
        w_home: np.ndarray,
        b_home: float,
        w_away: np.ndarray,
        b_away: float,
        l3: float,
        tau: float,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
    ) -> None:
        self.w_home = w_home
        self.b_home = b_home
        self.w_away = w_away
        self.b_away = b_away
        self.l3 = l3
        self.tau = tau
        self.feature_names = feature_names

    def _rates(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lh = np.exp(X @ self.w_home + self.b_home)
        la = np.exp(-X @ self.w_away + self.b_away)
        return lh, la

    def predict_home_win_prob(self, X: np.ndarray) -> np.ndarray:
        """Return P(home wins) for each row in X.  Satisfies Model protocol."""
        lh_arr, la_arr = self._rates(X)
        probs = np.empty(len(X))
        for i in range(len(X)):
            g = _score_grid(float(lh_arr[i]), float(la_arr[i]), self.l3, self.tau)
            H = np.arange(_H_MAX + 1)
            A = np.arange(_A_MAX + 1)
            HH, AA = np.meshgrid(H, A, indexing="ij")
            probs[i] = float(g[HH > AA].sum())
        return probs

    def predict_distribution(self, x: np.ndarray) -> BivariatePrediction:
        """Return the full score-distribution prediction for a single match.

        x: (1, n_features) feature row.
        """
        lh = float(np.exp(x @ self.w_home + self.b_home)[0])
        la = float(np.exp(-x @ self.w_away + self.b_away)[0])

        g = _score_grid(lh, la, self.l3, self.tau)

        H = np.arange(_H_MAX + 1)
        A = np.arange(_A_MAX + 1)
        HH, AA = np.meshgrid(H, A, indexing="ij")

        home_win_prob = float(g[HH > AA].sum())
        draw_prob = float(g[HH == AA].sum())
        predicted_margin = float((g * (HH - AA)).sum())
        predicted_total = float((g * (HH + AA)).sum())

        # Top-5 scorelines
        flat_idx = np.argsort(g.ravel())[::-1][:5]
        top_scorelines = [
            ScoreLine(
                home_score=int(idx // (_A_MAX + 1)),
                away_score=int(idx % (_A_MAX + 1)),
                probability=float(g.ravel()[idx]),
            )
            for idx in flat_idx
        ]

        return BivariatePrediction(
            lambda_home=lh,
            lambda_away=la,
            lambda3=self.l3,
            score_grid=g,
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            predicted_margin=predicted_margin,
            predicted_total=predicted_total,
            top_scorelines=top_scorelines,
        )


# ---------------------------------------------------------------------------
# MLE objective
# ---------------------------------------------------------------------------


def _nll(
    params: np.ndarray,
    X: np.ndarray,
    y_home: np.ndarray,
    y_away: np.ndarray,
) -> float:
    """Negative log-likelihood (per sample) of the bivariate Poisson.

    Vectorised over the sample axis: computes the inner correlation sum for
    all samples in a single numpy broadcast over k=0..K_max.
    """
    n_feat = X.shape[1]
    w_h = params[:n_feat]
    b_h = params[n_feat]
    w_a = params[n_feat + 1 : 2 * n_feat + 1]
    b_a = params[2 * n_feat + 1]
    log_l3 = params[2 * n_feat + 2]
    tau = float(params[2 * n_feat + 3])

    l3 = float(np.exp(np.clip(log_l3, -6.0, 3.0)))
    lh_arr = np.exp(X @ w_h + b_h)        # shape (N,)
    la_arr = np.exp(-X @ w_a + b_a)       # shape (N,)

    n = len(y_home)
    h_arr = y_home.astype(int)
    a_arr = y_away.astype(int)

    # Per-sample λ3 clamp and derived λ1, λ2
    l3c_arr = np.minimum(l3, np.minimum(lh_arr, la_arr) * 0.9)
    l1_arr = np.maximum(lh_arr - l3c_arr, _MIN_RATE)
    l2_arr = np.maximum(la_arr - l3c_arr, _MIN_RATE)

    K_max = int(np.minimum(h_arr, a_arr).max())
    K = np.arange(K_max + 1)  # shape (K+1,)

    # Broadcast: K → (K+1, N), h/a → (1, N)
    K2 = K[:, None]
    H2 = h_arr[None, :]
    A2 = a_arr[None, :]
    L1_2 = l1_arr[None, :]
    L2_2 = l2_arr[None, :]
    L3c_2 = l3c_arr[None, :]

    valid = np.minimum(H2, A2) >= K2
    safe_h_k = np.maximum(H2 - K2, 0)
    safe_a_k = np.maximum(A2 - K2, 0)

    log_terms = np.where(
        valid,
        (gammaln(H2 + 1) - gammaln(safe_h_k + 1) - gammaln(K2 + 1)
         + gammaln(A2 + 1) - gammaln(safe_a_k + 1) - gammaln(K2 + 1)
         + gammaln(K2 + 1)
         + K2 * (np.log(L3c_2) - np.log(L1_2) - np.log(L2_2))),
        -np.inf,
    )
    log_corr = _logsumexp(log_terms, axis=0)  # shape (N,)

    log_p = (
        -(l1_arr + l2_arr + l3c_arr)
        + h_arr * np.log(l1_arr)
        + a_arr * np.log(l2_arr)
        - gammaln(h_arr + 1)
        - gammaln(a_arr + 1)
        + log_corr
    )

    # Dixon-Coles correction for the 4 near-zero cells
    if tau != 0.0:
        for i in range(n):
            h, a = int(h_arr[i]), int(a_arr[i])
            if h <= 1 and a <= 1:
                dc = _dc_factor(h, a, float(lh_arr[i]), float(la_arr[i]), tau)
                log_p[i] += np.log(dc)

    log_p = np.where(np.isfinite(log_p), log_p, -30.0)
    return float(-log_p.mean())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BivariateTrainResult:
    model: BivariatePoissonModel
    feature_names: tuple[str, ...]
    n_train: int
    final_nll: float


def train_bivariate_poisson(
    frame: BivariateTrainingFrame,
    *,
    warm_start_model: BivariatePoissonModel | None = None,
    max_iter: int = 200,
) -> BivariateTrainResult:
    """Fit the bivariate Poisson model by MLE (L-BFGS-B).

    Parameters
    ----------
    frame:
        Training data from ``build_bivariate_frame``.
    warm_start_model:
        If provided, initialises the linear weights from this model.
        Otherwise the weights are initialised to small random values.
    max_iter:
        Maximum L-BFGS-B iterations (default 200).
    """
    if frame.X.shape[0] < 10:
        raise ValueError(f"Need at least 10 rows to train; got {frame.X.shape[0]}")

    n_feat = frame.X.shape[1]

    # Normalise features for stable gradient steps
    mu_x = frame.X.mean(axis=0)
    std_x = frame.X.std(axis=0)
    std_x[std_x < 1e-8] = 1.0
    X_scaled = (frame.X - mu_x) / std_x

    if warm_start_model is not None:
        w_h0 = warm_start_model.w_home * std_x
        b_h0 = warm_start_model.b_home + float(warm_start_model.w_home @ mu_x)
        w_a0 = warm_start_model.w_away * std_x
        b_a0 = warm_start_model.b_away + float(warm_start_model.w_away @ mu_x)
        log_l3_0 = np.log(max(warm_start_model.l3, _MIN_RATE))
        tau_0 = warm_start_model.tau
    else:
        rng = np.random.default_rng(42)
        w_h0 = rng.normal(0, 0.01, n_feat)
        b_h0 = np.log(25.0)  # NRL average score ≈ 25
        w_a0 = rng.normal(0, 0.01, n_feat)
        b_a0 = np.log(20.0)
        log_l3_0 = np.log(1.0)
        tau_0 = 0.0

    x0 = np.concatenate([w_h0, [b_h0], w_a0, [b_a0], [log_l3_0], [tau_0]])

    # Bounds: tau bounded to avoid invalid DC-correction factors
    bounds = (
        [(None, None)] * (n_feat + 1)   # w_h, b_h
        + [(None, None)] * (n_feat + 1)  # w_a, b_a
        + [(-6.0, 3.0)]                  # log_l3
        + [(-0.1, 0.1)]                  # tau (small for NRL)
    )

    result = minimize(
        _nll,
        x0,
        args=(X_scaled, frame.y_home, frame.y_away),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-8},
    )

    if not result.success:
        warnings.warn(f"L-BFGS-B did not converge: {result.message}", stacklevel=2)

    params = result.x
    # Rescale weights back to the original (unscaled) feature space
    w_h = params[:n_feat] / std_x
    b_h = params[n_feat] - float((params[:n_feat] / std_x) @ mu_x)
    w_a = params[n_feat + 1 : 2 * n_feat + 1] / std_x
    b_a = params[2 * n_feat + 1] - float((params[n_feat + 1 : 2 * n_feat + 1] / std_x) @ mu_x)
    l3 = float(np.exp(np.clip(params[2 * n_feat + 2], -6.0, 3.0)))
    tau = float(params[2 * n_feat + 3])

    model = BivariatePoissonModel(
        w_home=w_h,
        b_home=b_h,
        w_away=w_a,
        b_away=b_a,
        l3=l3,
        tau=tau,
        feature_names=frame.feature_names,
    )

    return BivariateTrainResult(
        model=model,
        feature_names=frame.feature_names,
        n_train=frame.X.shape[0],
        final_nll=float(result.fun),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_bivariate_poisson(path: Path | str, result: BivariateTrainResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "bivariate_poisson",
            "w_home": result.model.w_home,
            "b_home": result.model.b_home,
            "w_away": result.model.w_away,
            "b_away": result.model.b_away,
            "l3": result.model.l3,
            "tau": result.model.tau,
            "feature_names": list(result.feature_names),
        },
        path,
    )


def load_bivariate_poisson(path: Path | str) -> BivariatePoissonModel:
    blob = joblib.load(Path(path))
    if blob.get("model_type") != "bivariate_poisson":
        raise ValueError(
            f"Expected model_type='bivariate_poisson', got {blob.get('model_type')!r}"
        )
    return _from_blob(blob)


def _from_blob(blob: dict) -> BivariatePoissonModel:
    return BivariatePoissonModel(
        w_home=np.asarray(blob["w_home"]),
        b_home=float(blob["b_home"]),
        w_away=np.asarray(blob["w_away"]),
        b_away=float(blob["b_away"]),
        l3=float(blob["l3"]),
        tau=float(blob["tau"]),
        feature_names=tuple(blob["feature_names"]),
    )
