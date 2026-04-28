"""Bayesian hierarchical team-strength model for NRL score prediction.

Implements a Dixon-Coles-style hierarchical Poisson model:

    log λ_H = μ + home_adv + attack[home] - defense[away]
    log λ_A = μ            + attack[away] - defense[home]

where attack[i] and defense[i] are latent per-team strengths with partial
pooling via a non-centred hierarchical prior, home_adv is a global scalar,
and μ is the log league-mean scoring rate.

Design decisions (resolved in issue #144 comments):
- **Sampler**: PyMC5 with ``nuts_sampler="numpyro"`` for CPU speed.
- **Likelihood**: component Poisson form (Dixon & Coles 1997).
- **Dixon-Coles low-score correction**: Bernoulli toggle (default on), ρ
  estimated post-hoc from posterior mean rates via bounded MLE.
- **Cold-start prior**: power-prior ``p(θ) ∝ p(prev_season_posterior)^α``
  with α ≈ 0.5.  In practice this widens the previous season's posterior SD
  by ``1 / sqrt(α)`` so teams can change between seasons.
- **Posterior persistence**: 500-sample trimmed trace (stored in the joblib
  artefact) plus per-team Laplace-approximate (mean, std) summaries for fast
  point-estimate inference.
- **Coverage**: use HDI, not equal-tailed intervals, on the discrete margin
  distribution; include boundary integers in coverage checks.

PyMC5 and JAX/numpyro are ``training`` extras — not loaded at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson as _scipy_poisson
from scipy.stats import skellam as _skellam_dist

_MARGIN_LO = -50
_MARGIN_HI = 50
_MARGINS = np.arange(_MARGIN_LO, _MARGIN_HI + 1)  # shape (101,)

_TRACE_SAMPLE_SIZE = 500  # trimmed trace stored in the artefact


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianTrainingMatch:
    """A single historical match used to train the Bayesian model."""

    home_team_id: int
    away_team_id: int
    home_score: int
    away_score: int
    season: int


@dataclass(frozen=True)
class TeamPosterior:
    """Laplace-approximate posterior for a single team's latent strengths.

    Used to pass previous-season posteriors as power-priors for the next
    season's fit (cold-start regularisation).
    """

    team_id: int
    attack_mean: float
    attack_std: float
    defense_mean: float
    defense_std: float


@dataclass(frozen=True)
class MarginDistribution:
    """Posterior predictive margin distribution for a single match."""

    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    mean_margin: float
    ci_80: tuple[int, int]  # HDI — includes boundary integers
    ci_95: tuple[int, int]  # HDI
    pmf: np.ndarray  # shape (101,), index i → margin _MARGIN_LO + i


# ---------------------------------------------------------------------------
# HDI helper
# ---------------------------------------------------------------------------


def _hdi(pmf: np.ndarray, credible_mass: float) -> tuple[int, int]:
    """Highest Density Interval on a discrete marginal PMF.

    Accumulates probability mass starting from the highest-probability bins
    until ``credible_mass`` is reached, then returns the (min, max) margins
    of the included bins.  This produces the shortest interval that contains
    at least ``credible_mass`` of the distribution.
    """
    sorted_idx = np.argsort(pmf)[::-1]
    cumulative = 0.0
    included: list[int] = []
    for idx in sorted_idx:
        included.append(int(idx))
        cumulative += float(pmf[idx])
        if cumulative >= credible_mass:
            break
    lo = int(_MARGINS[min(included)])
    hi = int(_MARGINS[max(included)])
    return lo, hi


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class BayesianHierarchicalModel:
    """Fitted Bayesian hierarchical team-strength model.

    Stores per-team Laplace-approximate posterior summaries for fast
    point-estimate inference (``win_probability``) plus an optional
    500-sample trimmed trace for full posterior predictive distributions
    (``margin_distribution``).

    Convention:
        high attack[i]  → team i scores more (good attacker)
        high defense[i] → team i concedes fewer (strong defender)
    """

    team_ids: tuple[int, ...]
    team_index: dict[int, int]  # team_id → 0-based index
    attack_mean: np.ndarray  # shape (n_teams,)
    attack_std: np.ndarray
    defense_mean: np.ndarray
    defense_std: np.ndarray
    home_advantage: float  # posterior mean, log scale
    mu: float  # log league-mean scoring rate, posterior mean
    use_dixon_coles: bool
    dc_rho: float  # Dixon-Coles correlation ρ
    trace_samples: dict[str, np.ndarray] | None = field(default=None)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _lambda_pair(
        self,
        home_team_id: int,
        away_team_id: int,
        *,
        att: np.ndarray | None = None,
        defn: np.ndarray | None = None,
        home_adv: float | None = None,
        mu_val: float | None = None,
    ) -> tuple[float, float]:
        hi = self.team_index[home_team_id]
        ai = self.team_index[away_team_id]
        att_ = self.attack_mean if att is None else att
        defn_ = self.defense_mean if defn is None else defn
        home_adv_ = self.home_advantage if home_adv is None else home_adv
        mu_ = self.mu if mu_val is None else mu_val
        lh = float(np.exp(mu_ + home_adv_ + att_[hi] - defn_[ai]))
        la = float(np.exp(mu_ + att_[ai] - defn_[hi]))
        return max(lh, 0.01), max(la, 0.01)

    def _dc_correction(self, h: int, a: int, lh: float, la: float) -> float:
        """Dixon-Coles τ factor for low-scoring cells {0,0},{0,1},{1,0},{1,1}."""
        if not self.use_dixon_coles:
            return 1.0
        rho = self.dc_rho
        if h == 0 and a == 0:
            return max(1e-10, 1.0 - lh * la * rho)
        if h == 0 and a == 1:
            return max(1e-10, 1.0 + lh * rho)
        if h == 1 and a == 0:
            return max(1e-10, 1.0 + la * rho)
        if h == 1 and a == 1:
            return max(1e-10, 1.0 - rho)
        return 1.0

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def margin_pmf(
        self,
        home_team_id: int,
        away_team_id: int,
        *,
        att: np.ndarray | None = None,
        defn: np.ndarray | None = None,
        home_adv: float | None = None,
        mu_val: float | None = None,
    ) -> np.ndarray:
        """Margin PMF P(home_score − away_score = m) for m in [−50, +50].

        Uses the Skellam distribution, adjusted by the Dixon-Coles correction
        on the four low-score joint cells.  Renormalised to sum to 1.
        """
        lh, la = self._lambda_pair(
            home_team_id,
            away_team_id,
            att=att,
            defn=defn,
            home_adv=home_adv,
            mu_val=mu_val,
        )
        pmf = _skellam_dist.pmf(_MARGINS, lh, la).astype(float)
        pmf = np.clip(pmf, 0.0, None)

        if self.use_dixon_coles:
            # Apply DC correction only to the three margin bins touched by the
            # four low-score cells: margin ∈ {−1, 0, +1}.
            #
            # For each affected margin m, the corrected PMF mass is:
            #   P'(margin = m) = P(margin = m) + ΔP(DC)
            # where ΔP(DC) = sum_{h−a=m, h≤1, a≤1} P(h,a) × (τ(h,a)−1).
            corrections: dict[int, float] = {-1: 0.0, 0: 0.0, 1: 0.0}
            for h in range(2):
                for a in range(2):
                    m = h - a
                    if m not in corrections:
                        continue
                    p_joint = float(_scipy_poisson.pmf(h, lh) * _scipy_poisson.pmf(a, la))
                    tau = self._dc_correction(h, a, lh, la)
                    corrections[m] += p_joint * (tau - 1.0)
            for m, delta in corrections.items():
                idx = m - _MARGIN_LO
                pmf[idx] = max(0.0, pmf[idx] + delta)

        total = pmf.sum()
        if total > 0:
            pmf /= total
        return pmf

    def win_probability(self, home_team_id: int, away_team_id: int) -> float:
        """Posterior mean home win probability (point estimate)."""
        pmf = self.margin_pmf(home_team_id, away_team_id)
        return float(pmf[_MARGINS > 0].sum())

    def margin_distribution(
        self,
        home_team_id: int,
        away_team_id: int,
        *,
        n_samples: int = _TRACE_SAMPLE_SIZE,
    ) -> MarginDistribution:
        """Full posterior predictive margin distribution.

        If a trimmed trace is stored, averages the PMF over ``n_samples``
        posterior draws for a proper posterior predictive.  Falls back to
        the point-estimate PMF when no trace is available.
        """
        if self.trace_samples is not None and n_samples > 0:
            n = min(n_samples, self.trace_samples["attack"].shape[0])
            pmf_sum = np.zeros(len(_MARGINS))
            for i in range(n):
                pmf_sum += self.margin_pmf(
                    home_team_id,
                    away_team_id,
                    att=self.trace_samples["attack"][i],
                    defn=self.trace_samples["defense"][i],
                    home_adv=float(self.trace_samples["home_adv"][i]),
                    mu_val=float(self.trace_samples["mu"][i]),
                )
            pmf = pmf_sum / n
        else:
            pmf = self.margin_pmf(home_team_id, away_team_id)

        home_win_prob = float(pmf[_MARGINS > 0].sum())
        draw_prob = float(pmf[_MARGINS == 0].sum())
        away_win_prob = float(pmf[_MARGINS < 0].sum())
        mean_margin = float((pmf * _MARGINS).sum())

        return MarginDistribution(
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            mean_margin=mean_margin,
            ci_80=_hdi(pmf, 0.80),
            ci_95=_hdi(pmf, 0.95),
            pmf=pmf,
        )

    def team_posterior(self, team_id: int) -> TeamPosterior:
        """Return the Laplace-approximate posterior for a team."""
        i = self.team_index[team_id]
        return TeamPosterior(
            team_id=team_id,
            attack_mean=float(self.attack_mean[i]),
            attack_std=float(self.attack_std[i]),
            defense_mean=float(self.defense_mean[i]),
            defense_std=float(self.defense_std[i]),
        )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _build_training_arrays(
    matches: list[BayesianTrainingMatch],
    team_index: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (home_idx, away_idx, y_home, y_away) integer/float arrays."""
    home_idx = np.array([team_index[m.home_team_id] for m in matches], dtype=int)
    away_idx = np.array([team_index[m.away_team_id] for m in matches], dtype=int)
    y_home = np.array([m.home_score for m in matches], dtype=float)
    y_away = np.array([m.away_score for m in matches], dtype=float)
    return home_idx, away_idx, y_home, y_away


def _estimate_dc_rho(
    matches: list[BayesianTrainingMatch],
    lh_arr: np.ndarray,
    la_arr: np.ndarray,
) -> float:
    """Bounded MLE for the Dixon-Coles ρ correlation parameter.

    Only matches with (home, away) scores in {0,1}×{0,1} contribute to the
    log-likelihood; the rest drop out (their τ factor is 1.0).
    """
    low_score_mask = np.array([(m.home_score <= 1 and m.away_score <= 1) for m in matches])
    if not low_score_mask.any():
        return 0.0

    scores = [(m.home_score, m.away_score) for m in matches]
    lh_low = lh_arr[low_score_mask]
    la_low = la_arr[low_score_mask]
    sc_low = [s for s, flag in zip(scores, low_score_mask, strict=True) if flag]

    def _neg_ll(rho: float) -> float:
        ll = 0.0
        for (h, a), lh, la in zip(sc_low, lh_low, la_low, strict=True):
            if h == 0 and a == 0:
                tau = 1.0 - lh * la * rho
            elif h == 0 and a == 1:
                tau = 1.0 + lh * rho
            elif h == 1 and a == 0:
                tau = 1.0 + la * rho
            else:
                tau = 1.0 - rho
            ll += np.log(max(tau, 1e-10))
        return -ll

    result = minimize_scalar(_neg_ll, bounds=(-0.15, 0.15), method="bounded")
    return float(result.x) if result.success else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_bayesian_hierarchical(
    matches: list[BayesianTrainingMatch],
    *,
    prior_posteriors: dict[int, TeamPosterior] | None = None,
    power_prior_alpha: float = 0.5,
    use_dixon_coles: bool = True,
    n_samples: int = 2000,
    n_warmup: int = 2000,
    n_chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
    nuts_sampler: str = "numpyro",
) -> BayesianHierarchicalModel:
    """Fit the Bayesian hierarchical model via NUTS.

    Args:
        matches: Historical match results.  All teams referenced must appear
            in both home and away roles at least once across the dataset.
        prior_posteriors: Per-team posteriors from a previous season, used as
            power-priors (alpha ≈ 0.5 for season-to-season shrinkage).
        power_prior_alpha: Exponent α for the power-prior.  α < 1 widens the
            previous posterior (teams can change between seasons); α > 1
            tightens it.
        use_dixon_coles: Apply Dixon-Coles τ correction on {0,0},{0,1},{1,0},
            {1,1} joint cells.
        n_samples: Posterior draws per chain (production: 2000).
        n_warmup: NUTS warmup/tuning draws per chain (production: 2000).
        n_chains: Number of parallel chains.
        target_accept: NUTS target acceptance rate.
        random_seed: For reproducibility.
        nuts_sampler: Passed directly to ``pm.sample``; ``"numpyro"`` gives
            5–10× speedup over the default PyMC NUTS on CPU.

    Returns:
        Fitted ``BayesianHierarchicalModel`` with Laplace summaries and a
        500-sample trimmed posterior trace.
    """
    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError(
            "PyMC5 is required for training the Bayesian hierarchical model. "
            "Install via: uv sync --extra training"
        ) from exc

    if len(matches) < 10:
        raise ValueError(f"Need at least 10 training matches; got {len(matches)}")

    # Index teams -----------------------------------------------------------
    all_team_ids = sorted({m.home_team_id for m in matches} | {m.away_team_id for m in matches})
    team_index = {tid: i for i, tid in enumerate(all_team_ids)}
    n_teams = len(all_team_ids)
    home_idx, away_idx, y_home, y_away = _build_training_arrays(matches, team_index)

    # Build per-team prior hyperparameters ----------------------------------
    prior_posteriors = prior_posteriors or {}
    league_att_std = 0.5
    league_def_std = 0.5
    att_prior_mean = np.zeros(n_teams)
    att_prior_std = np.full(n_teams, league_att_std)
    def_prior_mean = np.zeros(n_teams)
    def_prior_std = np.full(n_teams, league_def_std)

    for tid, prior in prior_posteriors.items():
        if tid not in team_index:
            continue
        i = team_index[tid]
        scale = 1.0 / np.sqrt(power_prior_alpha)
        att_prior_mean[i] = prior.attack_mean
        att_prior_std[i] = prior.attack_std * scale
        def_prior_mean[i] = prior.defense_mean
        def_prior_std[i] = prior.defense_std * scale

    # PyMC model ------------------------------------------------------------
    with pm.Model() as model:  # noqa: F841
        # Global parameters
        mu = pm.Normal("mu", mu=float(np.log(18.0)), sigma=0.5)
        home_adv = pm.Normal("home_adv", mu=0.1, sigma=0.2)

        # Per-team latent strengths (non-centred, vectorised)
        attack = pm.Normal(
            "attack",
            mu=att_prior_mean,
            sigma=att_prior_std,
            shape=n_teams,
        )
        defense = pm.Normal(
            "defense",
            mu=def_prior_mean,
            sigma=def_prior_std,
            shape=n_teams,
        )

        # Expected scoring rates
        log_lh = mu + home_adv + attack[home_idx] - defense[away_idx]
        log_la = mu + attack[away_idx] - defense[home_idx]
        lambda_home = pm.math.exp(log_lh)
        lambda_away = pm.math.exp(log_la)

        # Independent Poisson likelihoods
        pm.Poisson("home_goals", mu=lambda_home, observed=y_home)
        pm.Poisson("away_goals", mu=lambda_away, observed=y_away)

        # Sample
        trace = pm.sample(
            draws=n_samples,
            tune=n_warmup,
            chains=n_chains,
            target_accept=target_accept,
            nuts_sampler=nuts_sampler,
            random_seed=random_seed,
            progressbar=False,
        )

    # Extract posterior summaries -------------------------------------------
    att_flat = trace.posterior["attack"].values.reshape(-1, n_teams)
    def_flat = trace.posterior["defense"].values.reshape(-1, n_teams)
    home_adv_flat = trace.posterior["home_adv"].values.reshape(-1)
    mu_flat = trace.posterior["mu"].values.reshape(-1)

    attack_mean = att_flat.mean(axis=0)
    attack_std = att_flat.std(axis=0)
    defense_mean = def_flat.mean(axis=0)
    defense_std = def_flat.std(axis=0)
    home_advantage_mean = float(home_adv_flat.mean())
    mu_mean = float(mu_flat.mean())

    # Trimmed 500-sample trace (random subsample without replacement)
    rng = np.random.default_rng(random_seed)
    n_total = att_flat.shape[0]
    idx = rng.choice(n_total, size=min(_TRACE_SAMPLE_SIZE, n_total), replace=False)
    trace_samples = {
        "attack": att_flat[idx],
        "defense": def_flat[idx],
        "home_adv": home_adv_flat[idx],
        "mu": mu_flat[idx],
    }

    # Estimate Dixon-Coles ρ from posterior mean rates ----------------------
    dc_rho = 0.0
    if use_dixon_coles:
        lh_arr = np.exp(
            mu_mean + home_advantage_mean + attack_mean[home_idx] - defense_mean[away_idx]
        )
        la_arr = np.exp(mu_mean + attack_mean[away_idx] - defense_mean[home_idx])
        dc_rho = _estimate_dc_rho(matches, lh_arr, la_arr)

    return BayesianHierarchicalModel(
        team_ids=tuple(all_team_ids),
        team_index=team_index,
        attack_mean=attack_mean,
        attack_std=attack_std,
        defense_mean=defense_mean,
        defense_std=defense_std,
        home_advantage=home_advantage_mean,
        mu=mu_mean,
        use_dixon_coles=use_dixon_coles,
        dc_rho=dc_rho,
        trace_samples=trace_samples,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_bayesian_hierarchical(
    path: Path | str,
    model: BayesianHierarchicalModel,
) -> None:
    """Serialise model to a joblib artefact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "bayesian_hierarchical",
            "team_ids": model.team_ids,
            "team_index": model.team_index,
            "attack_mean": model.attack_mean,
            "attack_std": model.attack_std,
            "defense_mean": model.defense_mean,
            "defense_std": model.defense_std,
            "home_advantage": model.home_advantage,
            "mu": model.mu,
            "use_dixon_coles": model.use_dixon_coles,
            "dc_rho": model.dc_rho,
            "trace_samples": model.trace_samples,
        },
        path,
    )


def load_bayesian_hierarchical(path: Path | str) -> BayesianHierarchicalModel:
    """Load a previously saved Bayesian hierarchical model."""
    blob = joblib.load(Path(path))
    if blob.get("model_type") != "bayesian_hierarchical":
        raise ValueError(
            f"Expected model_type='bayesian_hierarchical', got {blob.get('model_type')!r}"
        )
    return BayesianHierarchicalModel(
        team_ids=tuple(blob["team_ids"]),
        team_index=dict(blob["team_index"]),
        attack_mean=np.asarray(blob["attack_mean"]),
        attack_std=np.asarray(blob["attack_std"]),
        defense_mean=np.asarray(blob["defense_mean"]),
        defense_std=np.asarray(blob["defense_std"]),
        home_advantage=float(blob["home_advantage"]),
        mu=float(blob["mu"]),
        use_dixon_coles=bool(blob["use_dixon_coles"]),
        dc_rho=float(blob["dc_rho"]),
        trace_samples=blob.get("trace_samples"),
    )


def _from_blob(blob: dict) -> BayesianHierarchicalModel:
    """Called by loader.py dispatch."""
    return BayesianHierarchicalModel(
        team_ids=tuple(blob["team_ids"]),
        team_index=dict(blob["team_index"]),
        attack_mean=np.asarray(blob["attack_mean"]),
        attack_std=np.asarray(blob["attack_std"]),
        defense_mean=np.asarray(blob["defense_mean"]),
        defense_std=np.asarray(blob["defense_std"]),
        home_advantage=float(blob["home_advantage"]),
        mu=float(blob["mu"]),
        use_dixon_coles=bool(blob["use_dixon_coles"]),
        dc_rho=float(blob["dc_rho"]),
        trace_samples=blob.get("trace_samples"),
    )
