"""Bayesian hierarchical team-strength model for NRL score prediction (#144).

Generative model (Poisson component form, Dixon-Coles 1997 lineage):

    log(λ_H) = μ + att[home] + dfn[away] + home_adv
    log(λ_A) = μ + att[away] + dfn[home]
    home_score ~ Poisson(λ_H)
    away_score ~ Poisson(λ_A)

Parameters:
    att[t]   — latent attack strength, ZeroSumNormal over all teams.
    dfn[t]   — latent defense strength (negative = concede more), ZeroSumNormal.
    home_adv — shared home-field log-rate boost.
    μ        — log-scale intercept (≈ log of typical NRL score).

Priors:
    σ_att, σ_dfn ~ HalfNormal(0.3)
    att ~ ZeroSumNormal(σ_att, shape=n_teams)
    dfn ~ ZeroSumNormal(σ_dfn, shape=n_teams)
    home_adv ~ Normal(0.1, 0.3)
    μ ~ Normal(log(20), 0.5)

Optional Dixon-Coles correction (``use_dc_correction=True``): a
τ-parameterised log-likelihood adjustment for near-zero score cells
{(0,0), (0,1), (1,0), (1,1)} — counteracts the independence assumption
for very-low-score matches. For NRL where sub-6-point totals are rare,
τ typically converges near zero.

Sampling: NUTS via PyMC5, with the numpyro backend used automatically
if numpyro is installed (≈5× faster per sample on CPU).

Persistence: a 500-sample trimmed posterior trace (interleaved chains,
thinned) is stored as a dict of numpy arrays alongside team metadata.
This is enough to compute any posterior summary and is cheap to joblib-dump
(~200 KB for 16 teams × 500 samples × 4 parameters).

References
----------
Dixon, M. J. & Coles, S. G. (1997). Modelling Association Football Scores.
    Applied Statistics, 46(2), 265-280.
Karbone, J. et al. (2018). Unpublished NRL extension of Dixon-Coles.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import skellam as _skellam_dist

from fantasy_coach.features import MatchRow

logger = logging.getLogger(__name__)

_MARGIN_LO = -60
_MARGIN_HI = 60
_MARGINS = np.arange(_MARGIN_LO, _MARGIN_HI + 1)  # shape (121,)
_N_TRIM_SAMPLES = 500


# ---------------------------------------------------------------------------
# Training frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianMatchData:
    """Match-level input for the hierarchical model."""

    home_idx: np.ndarray  # shape (n,) int — index into teams
    away_idx: np.ndarray  # shape (n,) int — index into teams
    home_score: np.ndarray  # shape (n,) int
    away_score: np.ndarray  # shape (n,) int
    teams: tuple[int, ...]  # team IDs ordered so teams[i] has index i
    team_names: dict[int, str] = field(default_factory=dict)


def build_bayesian_frame(matches: Iterable[MatchRow]) -> BayesianMatchData:
    """Convert a MatchRow iterable to BayesianMatchData for model training."""
    completed = sorted(
        (
            m
            for m in matches
            if m.match_state in {"FullTime", "FullTimeED"}
            and m.home.score is not None
            and m.away.score is not None
        ),
        key=lambda m: (m.start_time, m.match_id),
    )

    if not completed:
        return BayesianMatchData(
            home_idx=np.zeros(0, dtype=int),
            away_idx=np.zeros(0, dtype=int),
            home_score=np.zeros(0, dtype=int),
            away_score=np.zeros(0, dtype=int),
            teams=(),
        )

    team_names: dict[int, str] = {}
    all_ids: list[int] = []
    for m in completed:
        all_ids.append(m.home.team_id)
        all_ids.append(m.away.team_id)
        team_names[m.home.team_id] = m.home.name
        team_names[m.away.team_id] = m.away.name

    teams = tuple(sorted(set(all_ids)))
    idx = {t: i for i, t in enumerate(teams)}

    return BayesianMatchData(
        home_idx=np.array([idx[m.home.team_id] for m in completed], dtype=int),
        away_idx=np.array([idx[m.away.team_id] for m in completed], dtype=int),
        home_score=np.array([m.home.score for m in completed], dtype=int),
        away_score=np.array([m.away.score for m in completed], dtype=int),
        teams=teams,
        team_names=team_names,
    )


# ---------------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------------


def _skellam_home_win_prob(lh: float, la: float) -> float:
    """P(H > A) where H ~ Poisson(lh), A ~ Poisson(la) via Skellam PMF."""
    pmf = _skellam_dist.pmf(_MARGINS, max(lh, 1e-6), max(la, 1e-6))
    return float(pmf[_MARGINS > 0].sum())


def _hdi(samples: np.ndarray, credible_interval: float) -> tuple[float, float]:
    """Highest Density Interval from 1-D samples.

    Uses the minimum-width equal-mass window over sorted samples.
    Includes boundary values (convention for discrete distributions).
    """
    sorted_s = np.sort(samples)
    n = len(sorted_s)
    interval_width = max(1, int(np.ceil(credible_interval * n)))
    n_windows = n - interval_width
    if n_windows <= 0:
        return float(sorted_s[0]), float(sorted_s[-1])
    widths = sorted_s[interval_width:] - sorted_s[:n_windows]
    lo_idx = int(np.argmin(widths))
    return float(sorted_s[lo_idx]), float(sorted_s[lo_idx + interval_width])


def _nuts_sampler() -> str:
    """Return the fastest available NUTS backend name for PyMC."""
    try:
        import numpyro  # noqa: F401

        return "numpyro"
    except ImportError:
        return "pymc"


def _extract_posterior(
    idata: object,
    n_trim: int,
    var_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Flatten chain×draw and thin to at most ``n_trim`` samples."""
    posterior = idata.posterior  # type: ignore[attr-defined]
    result: dict[str, np.ndarray] = {}
    for var in var_names:
        try:
            vals = np.array(posterior[var])  # (chain, draw[, ...])
        except KeyError:
            continue
        flat = vals.reshape(-1, *vals.shape[2:])
        step = max(1, len(flat) // n_trim)
        result[var] = flat[::step][:n_trim]
    return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BayesianHierarchicalModel:
    """Fitted hierarchical Poisson model for win-probability prediction.

    Predictions are derived from the stored posterior trace:
    - ``predict_win_prob``: posterior mean P(home wins) via Skellam PMF.
    - ``predict_margin_hdi``: posterior predictive margin, summarised as HDI.

    Unknown teams fall back gracefully to 0.5 win probability.
    """

    def __init__(
        self,
        posterior_samples: dict[str, np.ndarray],
        teams: tuple[int, ...],
        team_names: dict[int, str] | None = None,
        use_dc_correction: bool = False,
    ) -> None:
        self.posterior_samples = posterior_samples
        self.teams = teams
        self._team_idx: dict[int, int] = {t: i for i, t in enumerate(teams)}
        self.team_names: dict[int, str] = team_names or {}
        self.use_dc_correction = use_dc_correction
        # Required by the Model protocol (feature_names is empty — this model
        # takes team IDs, not a feature-difference vector; adapter in PR B).
        self.feature_names: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_samples(self, home_idx: int, away_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior samples of (λ_H, λ_A) for a team pair."""
        mu = self.posterior_samples["mu"]  # (S,)
        att = self.posterior_samples["att"]  # (S, n_teams)
        dfn = self.posterior_samples["dfn"]  # (S, n_teams)
        home_adv = self.posterior_samples["home_adv"]  # (S,)

        log_lh = mu + att[:, home_idx] + dfn[:, away_idx] + home_adv
        log_la = mu + att[:, away_idx] + dfn[:, home_idx]
        return np.exp(log_lh), np.exp(log_la)

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict_win_prob(self, home_team_id: int, away_team_id: int) -> float:
        """Posterior mean P(home wins) for the given matchup.

        Returns 0.5 for unrecognised team IDs (graceful degradation).
        """
        if home_team_id not in self._team_idx or away_team_id not in self._team_idx:
            return 0.5
        h = self._team_idx[home_team_id]
        a = self._team_idx[away_team_id]
        lh, la = self._rate_samples(h, a)
        probs = np.array(
            [_skellam_home_win_prob(float(lh[s]), float(la[s])) for s in range(len(lh))]
        )
        return float(probs.mean())

    def predict_margin_hdi(self, home_team_id: int, away_team_id: int) -> dict[str, float]:
        """Posterior predictive margin summary: mean + 80%/95% HDI.

        Returns wide symmetric defaults for unrecognised teams.
        """
        if home_team_id not in self._team_idx or away_team_id not in self._team_idx:
            return {
                "mean": 0.0,
                "hdi_80_lo": -15.0,
                "hdi_80_hi": 15.0,
                "hdi_95_lo": -25.0,
                "hdi_95_hi": 25.0,
            }
        h = self._team_idx[home_team_id]
        a = self._team_idx[away_team_id]
        lh, la = self._rate_samples(h, a)

        rng = np.random.default_rng(seed=0)
        home_draws = rng.poisson(lh)
        away_draws = rng.poisson(la)
        margins = (home_draws - away_draws).astype(float)

        hdi_80 = _hdi(margins, 0.80)
        hdi_95 = _hdi(margins, 0.95)
        return {
            "mean": float(margins.mean()),
            "hdi_80_lo": hdi_80[0],
            "hdi_80_hi": hdi_80[1],
            "hdi_95_lo": hdi_95[0],
            "hdi_95_hi": hdi_95[1],
        }

    @property
    def team_strength_summary(self) -> dict[int, dict[str, float]]:
        """Posterior mean attack and defense for each team."""
        att_mean = self.posterior_samples["att"].mean(axis=0)
        dfn_mean = self.posterior_samples["dfn"].mean(axis=0)
        return {
            team_id: {"attack": float(att_mean[i]), "defense": float(dfn_mean[i])}
            for i, team_id in enumerate(self.teams)
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianTrainResult:
    model: BayesianHierarchicalModel
    n_train: int
    n_teams: int
    n_chains: int
    n_samples: int


def train_bayesian_hierarchical(
    data: BayesianMatchData,
    *,
    n_tune: int = 500,
    n_samples: int = 500,
    n_chains: int = 2,
    use_dc_correction: bool = False,
    random_seed: int = 42,
    n_trim: int = _N_TRIM_SAMPLES,
    progressbar: bool = False,
) -> BayesianTrainResult:
    """Fit the hierarchical model via NUTS.

    Requires ``pymc>=5.0`` from the ``training`` extras group.
    Uses the numpyro NUTS backend automatically when numpyro is installed.

    Parameters
    ----------
    data:
        Training frame from ``build_bayesian_frame``.
    n_tune:
        NUTS warmup (adaptation) steps per chain.
    n_samples:
        Posterior draws per chain.
    n_chains:
        Number of independent chains.
    use_dc_correction:
        Apply Dixon-Coles low-score likelihood correction (τ parameter).
    random_seed:
        RNG seed for reproducibility.
    n_trim:
        Maximum posterior samples retained (interleaved chains, thinned).
    progressbar:
        Show PyMC progress bar (disabled by default for CI/log cleanliness).
    """
    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError(
            "pymc is required for Bayesian model training. Install with: uv sync --extra training"
        ) from exc

    if len(data.teams) < 2:
        raise ValueError(f"Need at least 2 teams; got {len(data.teams)}")
    if len(data.home_idx) < 10:
        raise ValueError(f"Need at least 10 matches; got {len(data.home_idx)}")

    n_teams = len(data.teams)

    with pm.Model() as pymc_model:  # noqa: F841
        # Hierarchical variance priors
        sigma_att = pm.HalfNormal("sigma_att", sigma=0.3)
        sigma_dfn = pm.HalfNormal("sigma_dfn", sigma=0.3)

        # Sum-to-zero team parameters (ZeroSumNormal enforces Σatt=0, Σdfn=0)
        att = pm.ZeroSumNormal("att", sigma=sigma_att, shape=n_teams)
        dfn = pm.ZeroSumNormal("dfn", sigma=sigma_dfn, shape=n_teams)

        # Shared parameters
        home_adv = pm.Normal("home_adv", mu=0.1, sigma=0.3)
        mu = pm.Normal("mu", mu=float(np.log(20.0)), sigma=0.5)

        # Log-rate equations
        log_lh = mu + att[data.home_idx] + dfn[data.away_idx] + home_adv
        log_la = mu + att[data.away_idx] + dfn[data.home_idx]

        lh = pm.math.exp(log_lh)
        la = pm.math.exp(log_la)

        # Poisson likelihoods
        pm.Poisson("home_goals", mu=lh, observed=data.home_score)
        pm.Poisson("away_goals", mu=la, observed=data.away_score)

        # Dixon-Coles low-score correction (τ ~ Beta, tight prior for NRL)
        if use_dc_correction:
            _apply_dc_correction(data, lh, la)

        sampler = _nuts_sampler()
        logger.debug("Sampling with %s NUTS backend, %d chains", sampler, n_chains)
        idata = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            nuts_sampler=sampler,
            random_seed=random_seed,
            progressbar=progressbar,
        )

    var_names = ("att", "dfn", "home_adv", "mu", "sigma_att", "sigma_dfn")
    if use_dc_correction:
        var_names = (*var_names, "tau_dc")

    posterior_samples = _extract_posterior(idata, n_trim, var_names)

    model = BayesianHierarchicalModel(
        posterior_samples=posterior_samples,
        teams=data.teams,
        team_names=data.team_names,
        use_dc_correction=use_dc_correction,
    )
    return BayesianTrainResult(
        model=model,
        n_train=len(data.home_idx),
        n_teams=n_teams,
        n_chains=n_chains,
        n_samples=n_samples,
    )


def _apply_dc_correction(
    data: BayesianMatchData,
    lh: object,
    la: object,
) -> None:
    """Add the Dixon-Coles low-score log-likelihood adjustment as a Potential.

    φ(0,0) = 1 - λH·λA·τ
    φ(1,0) = 1 + λA·τ
    φ(0,1) = 1 + λH·τ
    φ(1,1) = 1 - τ
    """
    try:
        import pymc as pm
        import pytensor.tensor as pt
    except ImportError as exc:
        raise ImportError("pymc and pytensor are required") from exc

    tau = pm.Beta("tau_dc", alpha=2.0, beta=10.0)
    terms: list[object] = []

    mask_00 = (data.home_score == 0) & (data.away_score == 0)
    mask_10 = (data.home_score == 1) & (data.away_score == 0)
    mask_01 = (data.home_score == 0) & (data.away_score == 1)
    mask_11 = (data.home_score == 1) & (data.away_score == 1)

    if mask_00.any():
        terms.append(pt.sum(pt.log(1.0 - lh[mask_00] * la[mask_00] * tau)))
    if mask_10.any():
        terms.append(pt.sum(pt.log(1.0 + la[mask_10] * tau)))
    if mask_01.any():
        terms.append(pt.sum(pt.log(1.0 + lh[mask_01] * tau)))
    if mask_11.any():
        terms.append(pt.sum(pt.log(1.0 - tau) * float(mask_11.sum())))

    if terms:
        pm.Potential("dc_correction", pt.add(*terms))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_bayesian_hierarchical(path: Path | str, result: BayesianTrainResult) -> None:
    """Persist the trimmed posterior and metadata to a joblib file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "bayesian_hierarchical",
            "posterior_samples": result.model.posterior_samples,
            "teams": result.model.teams,
            "team_names": result.model.team_names,
            "use_dc_correction": result.model.use_dc_correction,
            "n_train": result.n_train,
            "n_teams": result.n_teams,
        },
        path,
    )


def load_bayesian_hierarchical(path: Path | str) -> BayesianHierarchicalModel:
    """Load a saved Bayesian hierarchical model (no PyMC import required)."""
    blob = joblib.load(Path(path))
    if blob.get("model_type") != "bayesian_hierarchical":
        raise ValueError(
            f"Expected model_type='bayesian_hierarchical', got {blob.get('model_type')!r}"
        )
    return BayesianHierarchicalModel(
        posterior_samples=blob["posterior_samples"],
        teams=tuple(blob["teams"]),
        team_names=blob.get("team_names", {}),
        use_dc_correction=blob.get("use_dc_correction", False),
    )
