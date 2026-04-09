"""Model inspection objects for programmatic access to model metadata.

This module provides :class:`Summary` and :class:`Info`, returned by
``model.summary()`` and ``model.info()`` respectively, along with the
:class:`~typing.TypedDict` schemas for training and dataset metadata.

Both classes auto-print when their return value is discarded (e.g.
``model.summary()``) and stay silent when stored in a variable
(e.g. ``s = model.summary()``).  See :class:`AutoPrintMixin` for details.
"""

import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import numpy as np
import torch

if TYPE_CHECKING:
    from leaspy.models.base import BaseModel

__all__ = [
    "AutoPrintMixin",
    "DatasetInfo",
    "Info",
    "Summary",
    "TrainingInfo",
    "VisitsPerSubject",
    "compute_bic",
    "get_axis_labels",
    "get_number_of_parameters",
]


# ---------------------------------------------------------------------------
# TypedDict schemas for metadata
# ---------------------------------------------------------------------------


class VisitsPerSubject(TypedDict, total=False):
    """Per-subject visit distribution statistics."""

    median: float
    min: int
    max: int
    iqr: float


class DatasetInfo(TypedDict, total=False):
    """Statistics of the training dataset, computed during ``fit()``."""

    n_subjects: int
    n_scores: int
    n_visits: int
    n_observations: int
    visits_per_subject: VisitsPerSubject
    n_missing: int
    pct_missing: float
    n_events: int


class TrainingInfo(TypedDict, total=False):
    """Metadata about the training process, captured during ``fit()``."""

    algorithm: str
    seed: int
    n_iter: int
    n_burn_in_iter: int
    converged: bool
    duration: str


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_WIDTH = 80


def get_axis_labels(
    axis_name: Optional[str],
    size: int,
    feature_names: Optional[list[str]] = None,
) -> Optional[list[str]]:
    """Resolve human-readable labels for a parameter axis.

    Parameters
    ----------
    axis_name : str or None
        Semantic axis name (``"feature"``, ``"source"``, ``"cluster"``,
        ``"basis"``).
    size : int
        Number of elements along the axis.
    feature_names : list[str], optional
        Feature names used when *axis_name* is ``"feature"``.

    Returns
    -------
    list[str] or None
        Labels for the axis, or ``None`` if no meaningful labels are available.
    """
    if axis_name is None:
        return None

    if axis_name == "feature":
        if feature_names is not None:
            feats = feature_names[:size]
            return [f[:8] if len(f) <= 8 else f[:7] + "." for f in feats]
        return [f"f{i}" for i in range(size)]
    elif axis_name == "source":
        return [f"s{i}" for i in range(size)]
    elif axis_name == "cluster":
        return [f"c{i}" for i in range(size)]
    elif axis_name == "event":
        return None
    elif axis_name == "basis":
        return [f"b{i}" for i in range(size)]
    else:
        return None


def _wrap_text(label: str, text: str, indent: int = 0) -> list[str]:
    """Wrap *text* with a bold *label* prefix to fit within ``_WIDTH``."""
    prefix = f"{label}: " if label else ""
    initial_indent = " " * indent + prefix
    subsequent_indent = " " * (indent + 4)
    wrapper = textwrap.TextWrapper(
        width=_WIDTH,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapper.wrap(text)


# ---------------------------------------------------------------------------
# Auto-print mixin
# ---------------------------------------------------------------------------


class AutoPrintMixin:
    """Mixin that auto-prints when the object is discarded.

    Relies on CPython reference counting: when the return value of e.g.
    ``model.summary()`` is not assigned, the object is immediately
    garbage-collected, triggering ``__del__`` which prints it.

    When stored (``s = model.summary()``), any public attribute access
    sets ``_printed = True``, suppressing the ``__del__`` output.

    Subclasses must define a ``_printed: bool`` field (via dataclass)
    and a ``__str__`` method.
    """

    def __del__(self):
        if not object.__getattribute__(self, "_printed"):
            print(str(self))

    def __repr__(self) -> str:
        object.__setattr__(self, "_printed", True)
        return str(self)

    def __getattribute__(self, name: str):
        value = object.__getattribute__(self, name)
        # Suppress auto-print once any public attribute is accessed
        if not name.startswith("_") and name != "help":
            object.__setattr__(self, "_printed", True)
        return value


# ---------------------------------------------------------------------------
# Metric utilities
# ---------------------------------------------------------------------------


def get_number_of_parameters(model: "BaseModel") -> int:
    """Calculate the number of free parameters of the model.

    Uses the theoretical formula:
    ``P = 3F + (F-1)*S + S*K + 4K``
    where *F* = features, *S* = sources, *K* = clusters.

    Parameters
    ----------
    model
        A fitted Leaspy model instance.

    Returns
    -------
    int
        Number of free parameters.
    """
    n_features = getattr(model, "dimension", 0) or 0
    n_sources = getattr(model, "source_dimension", 0) or 0
    n_clusters = getattr(model, "n_clusters", 1) or 1

    return (
        (n_features * 3)
        + ((n_features - 1) * n_sources)
        + (n_sources * n_clusters)
        + (n_clusters * 4)
    )


def compute_bic(
    nll: float,
    num_params: int,
    n_subjects: int,
) -> Optional[float]:
    """Calculate the Bayesian Information Criterion (BIC).

    ``BIC = 2 * nll + P * log(N)``

    Parameters
    ----------
    nll : float
        Negative log-likelihood (``nll_attach``).
    num_params : int
        Number of free parameters.
    n_subjects : int
        Number of subjects used for model fitting.

    Returns
    -------
    float or None
        The computed BIC, or ``None`` if inputs are invalid.
    """
    if n_subjects <= 0:
        return None
    return 2 * nll + num_params * np.log(n_subjects)


# ---------------------------------------------------------------------------
# Parameter display registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ModelDisplayMeta:
    """Display metadata for a model class (used by Summary).

    Registered in ``_DISPLAY_REGISTRY`` keyed by class name.
    ``_get_display_meta`` walks the model's MRO to find the closest entry.
    """

    individual_prior_params: tuple[str, ...]
    noise_params: tuple[str, ...]
    param_axes: dict[str, tuple[str, ...]]


_BASE_PARAM_AXES: dict[str, tuple[str, ...]] = {
    "log_g_mean": ("feature",),
    "log_g_std": ("feature",),
    "log_v0_mean": ("feature",),
    "betas_mean": ("basis", "source"),
    "mixing_matrix": ("source", "feature"),
    "noise_std": ("feature",),
}

_DISPLAY_REGISTRY: dict[str, _ModelDisplayMeta] = {
    "McmcSaemCompatibleModel": _ModelDisplayMeta(
        individual_prior_params=(
            "tau_mean", "tau_std", "xi_mean", "xi_std",
            "sources_mean", "sources_std", "zeta_mean",
        ),
        noise_params=("noise_std",),
        param_axes=_BASE_PARAM_AXES,
    ),
    "TimeReparametrizedMixtureModel": _ModelDisplayMeta(
        individual_prior_params=(
            "tau_std", "xi_std", "sources_std", "tau_mean", "xi_mean",
        ),
        noise_params=("noise_std",),
        param_axes={
            **_BASE_PARAM_AXES,
            "tau_mean": ("cluster",),
            "tau_std": ("cluster",),
            "xi_mean": ("cluster",),
            "xi_std": ("cluster",),
            "sources_mean": ("source", "cluster"),
            "sources_std": ("source", "cluster"),
            "probs": ("cluster",),
        },
    ),
    "JointModel": _ModelDisplayMeta(
        individual_prior_params=(
            "tau_mean", "tau_std", "xi_mean", "xi_std",
            "sources_mean", "sources_std", "zeta_mean",
        ),
        noise_params=("noise_std",),
        param_axes={
            **_BASE_PARAM_AXES,
            "n_log_nu_mean": ("event",),
            "log_rho_mean": ("event",),
            "zeta_mean": ("source", "event"),
        },
    ),
}


def _get_display_meta(model) -> Optional[_ModelDisplayMeta]:
    """Return display metadata for *model* by walking its MRO.

    Returns the entry for the most specific registered class, or ``None``
    if the model's class hierarchy has no entry in ``_DISPLAY_REGISTRY``.
    """
    for cls in type(model).__mro__:
        if cls.__name__ in _DISPLAY_REGISTRY:
            return _DISPLAY_REGISTRY[cls.__name__]
    return None


def _build_param_categories(
    model, meta: _ModelDisplayMeta
) -> dict[str, list[str]]:
    """Categorize model parameters into display groups using *meta*."""
    ind_priors = set(meta.individual_prior_params)
    noise = set(meta.noise_params)
    all_params = set(model.parameters.keys()) if model.parameters else set()
    pop = all_params - ind_priors - noise

    def sort_key(name: str) -> tuple[int, str, str]:
        val = model.parameters[name]
        axes = meta.param_axes.get(name, ())
        primary_axis = axes[0] if axes else ""
        n_cols = 1
        if val.ndim == 1 and axes:
            if get_axis_labels(primary_axis, len(val), model.features) is not None:
                n_cols = len(val)
        elif val.ndim == 2:
            n_cols = val.shape[1]
        return (n_cols, primary_axis, name)

    return {
        "population": sorted((k for k in pop if k in all_params), key=sort_key),
        "individual_priors": sorted(
            (k for k in ind_priors if k in all_params), key=sort_key
        ),
        "noise": sorted((k for k in noise if k in all_params), key=sort_key),
    }


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class Info(AutoPrintMixin):
    """Model configuration and training context (no parameter values).

    Returned by ``model.info()``.  Auto-prints when discarded; provides
    programmatic access when stored in a variable.

    Examples
    --------
    >>> model.info()              # prints info
    >>> i = model.info()          # store for programmatic access
    >>> i.n_subjects              # 150
    >>> i.pct_missing             # 2.5
    >>> i.help()                  # list available attributes
    """

    name: str
    model_type: str
    dimension: Optional[int] = None
    features: Optional[list[str]] = None
    source_dimension: Optional[int] = None
    n_clusters: Optional[int] = None
    obs_models: Optional[list[str]] = None
    n_total_params: Optional[int] = None
    bic: Optional[float] = None
    training_info: TrainingInfo = field(default_factory=dict)
    dataset_info: DatasetInfo = field(default_factory=dict)
    leaspy_version: Optional[str] = None
    _printed: bool = field(default=False, repr=False)

    # -- Factory -------------------------------------------------------------

    @classmethod
    def from_model(cls, model: "BaseModel") -> "Info":
        """Build an :class:`Info` from a model instance."""
        # Observation model names
        obs_model_names = None
        if hasattr(model, "obs_models"):
            obs_model_names = [om.to_string() for om in model.obs_models]

        # Parameter count & BIC
        n_total_params = None
        bic = None
        if getattr(model, "parameters", None):
            n_total_params = get_number_of_parameters(model)
            fm = getattr(model, "fit_metrics", None) or {}
            nll_val = fm.get("nll_attach", fm.get("nll_tot"))
            n_subjects = model.dataset_info.get("n_subjects")
            if nll_val is not None and n_subjects is not None:
                bic = compute_bic(float(nll_val), n_total_params, n_subjects)

        # Leaspy version
        try:
            from leaspy import __version__ as version
        except ImportError:
            version = None

        return cls(
            name=model.name,
            model_type=model.__class__.__name__,
            dimension=model.dimension,
            features=model.features,
            source_dimension=getattr(model, "source_dimension", None),
            n_clusters=getattr(model, "n_clusters", None),
            obs_models=obs_model_names,
            n_total_params=n_total_params,
            bic=bic,
            training_info=dict(model.training_info),
            dataset_info=dict(model.dataset_info),
            leaspy_version=version,
        )

    # -- Convenience properties: training ------------------------------------

    @property
    def algorithm(self) -> Optional[str]:
        """Algorithm name used for training."""
        return self.training_info.get("algorithm")

    @property
    def seed(self) -> Optional[int]:
        """Random seed used for training."""
        return self.training_info.get("seed")

    @property
    def n_iter(self) -> Optional[int]:
        """Number of iterations."""
        return self.training_info.get("n_iter")

    @property
    def n_burn_in_iter(self) -> Optional[int]:
        """Number of burn-in (memory-less) iterations."""
        return self.training_info.get("n_burn_in_iter")

    @property
    def converged(self) -> Optional[bool]:
        """Whether training converged."""
        return self.training_info.get("converged")

    @property
    def duration(self) -> Optional[str]:
        """Training duration."""
        return self.training_info.get("duration")

    # -- Convenience properties: dataset -------------------------------------

    @property
    def n_subjects(self) -> Optional[int]:
        """Number of subjects in the training dataset."""
        return self.dataset_info.get("n_subjects")

    @property
    def n_visits(self) -> Optional[int]:
        """Total number of visits."""
        return self.dataset_info.get("n_visits")

    @property
    def n_scores(self) -> Optional[int]:
        """Number of scored features."""
        return self.dataset_info.get("n_scores")

    @property
    def n_observations(self) -> Optional[int]:
        """Total number of observed data points."""
        return self.dataset_info.get("n_observations")

    @property
    def pct_missing(self) -> Optional[float]:
        """Percentage of missing observations."""
        return self.dataset_info.get("pct_missing")

    @property
    def n_missing(self) -> Optional[int]:
        """Number of missing observations."""
        return self.dataset_info.get("n_missing")

    @property
    def visits_per_subject(self) -> Optional[VisitsPerSubject]:
        """Per-subject visit distribution statistics."""
        return self.dataset_info.get("visits_per_subject")

    @property
    def n_events(self) -> Optional[int]:
        """Number of observed events (joint models only)."""
        return self.dataset_info.get("n_events")

    # -- Display -------------------------------------------------------------

    def __str__(self) -> str:
        lines = []
        sep = "=" * _WIDTH

        lines.append(sep)
        lines.append(f"{'Model Information':^{_WIDTH}}")
        lines.append(sep)

        # Statistical Model
        lines.append("Statistical Model")
        lines.append("-" * _WIDTH)
        lines.append(f"Type: {self.model_type}")
        lines.append(f"Name: {self.name}")
        lines.append(f"Dimension: {self.dimension}")
        if self.source_dimension is not None:
            lines.append(f"Source Dimension: {self.source_dimension}")
        if self.obs_models:
            lines.append(f"Observation Models: {', '.join(self.obs_models)}")
        if self.n_total_params is not None:
            lines.append(f"Parameters: {self.n_total_params}")
        if self.bic is not None:
            lines.append(f"BIC: {self.bic:.2f}")
        if self.n_clusters is not None:
            lines.append(f"Clusters: {self.n_clusters}")

        # Training Dataset
        if self.dataset_info:
            lines.append("")
            lines.append("Training Dataset")
            lines.append("-" * _WIDTH)
            di = self.dataset_info
            lines.append(f"Subjects: {di.get('n_subjects', 'N/A')}")
            lines.append(f"Visits: {di.get('n_visits', 'N/A')}")
            lines.append(f"Scores (Features): {di.get('n_scores', 'N/A')}")
            lines.append(f"Total Observations: {di.get('n_observations', 'N/A')}")
            if "visits_per_subject" in di:
                vps = di["visits_per_subject"]
                lines.append(
                    f"Visits per Subject: Median {vps['median']:.1f} "
                    f"[Min {vps['min']}, Max {vps['max']}, IQR {vps['iqr']:.1f}]"
                )
            if "n_missing" in di:
                lines.append(
                    f"Missing Data: {di['n_missing']} "
                    f"({di.get('pct_missing', 0):.2f}%)"
                )
            if "n_events" in di:
                lines.append(f"Events Observed: {di['n_events']}")

        # Training Details
        if self.training_info:
            lines.append("")
            lines.append("Training Details")
            lines.append("-" * _WIDTH)
            ti = self.training_info
            lines.append(f"Algorithm: {ti.get('algorithm', 'N/A')}")
            if "seed" in ti:
                lines.append(f"Seed: {ti['seed']}")
            lines.append(f"Iterations: {ti.get('n_iter', 'N/A')}")
            if "n_burn_in_iter" in ti:
                n_b = ti['n_burn_in_iter']
                n_t = ti.get('n_iter') or 1
                lines.append(f"  Burn-in: {n_b}/{n_t} ({100*n_b/n_t:.0f}%)")
                lines.append(f"  Burn-out: {n_t - n_b}")
            if ti.get("converged") is not None:
                lines.append(f"Converged: {ti['converged']}")
            if "duration" in ti:
                lines.append(f"Duration: {ti['duration']}")

        # Leaspy Version
        if self.leaspy_version:
            lines.append("")
            lines.append(f"Leaspy Version: {self.leaspy_version}")

        lines.append(sep)
        return "\n".join(lines)

    def help(self) -> None:
        """Print available attributes and their meanings."""
        help_text = f"""
Info Help
{'=' * 60}

The Info object provides access to model configuration and training context.

Usage:
    model.info()          # Print model information
    i = model.info()      # Store to access individual attributes

Available Attributes:

  Model:
    name              Model name (str)
    model_type        Model class name (str)
    dimension         Number of features (int)
    features          Feature names (list[str])
    source_dimension  Number of sources (int or None)
    n_clusters        Number of clusters (int or None)
    obs_models        Observation model names (list[str] or None)
    n_total_params    Number of free parameters (int)
    bic               Bayesian Information Criterion (float or None)

  Training:
    algorithm         Algorithm name (str)
    seed              Random seed (int)
    n_iter            Number of iterations (int)
    n_burn_in_iter    Number of burn-in iterations (int)
    converged         Whether training converged (bool or None)
    duration          Training duration (str)

  Dataset:
    n_subjects        Number of subjects (int)
    n_visits          Total visits (int)
    n_scores          Number of scored features (int)
    n_observations    Total observations (int)
    pct_missing       Percent missing data (float)
    n_missing         Count of missing observations (int)
    visits_per_subject  Visit distribution stats (dict)
    n_events          Observed events, joint models only (int or None)

  Other:
    training_info     Full training metadata (TrainingInfo)
    dataset_info      Full dataset statistics (DatasetInfo)
    leaspy_version    Leaspy version (str)

Examples:
    >>> i = model.info()
    >>> i.algorithm              # 'mcmc_saem'
    >>> i.n_subjects             # 150
    >>> i.pct_missing            # 2.5
"""
        print(help_text)
        object.__setattr__(self, "_printed", True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class Summary(AutoPrintMixin):
    """Structured summary of a Leaspy model including parameter values.

    Returned by ``model.summary()``.  Auto-prints when discarded; provides
    programmatic access when stored in a variable.

    Examples
    --------
    >>> model.summary()           # prints the formatted summary
    >>> s = model.summary()       # store for programmatic access
    >>> s.algorithm               # 'mcmc_saem'
    >>> s.get_param('tau_std')    # tensor([10.5])
    >>> s.help()                  # list available attributes
    """

    name: str
    model_type: str
    dimension: Optional[int] = None
    features: Optional[list[str]] = None
    source_dimension: Optional[int] = None
    n_clusters: Optional[int] = None
    obs_models: Optional[list[str]] = None
    n_total_params: Optional[int] = None
    nll: Optional[float] = None
    bic: Optional[float] = None
    training_info: TrainingInfo = field(default_factory=dict)
    dataset_info: DatasetInfo = field(default_factory=dict)
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    leaspy_version: Optional[str] = None
    _param_axes: dict = field(default_factory=dict, repr=False)
    _feature_names: Optional[list[str]] = field(default=None, repr=False)
    _printed: bool = field(default=False, repr=False)

    # -- Factory -------------------------------------------------------------

    @classmethod
    def from_model(cls, model: "BaseModel") -> "Summary":
        """Build a :class:`Summary` from a model instance."""
        from leaspy.exceptions import LeaspyModelInputError

        if not model.is_initialized:
            raise LeaspyModelInputError(
                "Model is not initialized. Call fit() first."
            )
        if model.parameters is None or len(model.parameters) == 0:
            raise LeaspyModelInputError(
                "Model has no parameters. Call fit() first."
            )

        # NLL
        nll = None
        fm = getattr(model, "fit_metrics", None) or {}
        if nll_val := fm.get("nll_tot"):
            nll = float(nll_val)

        # Parameter count & BIC
        n_total_params = get_number_of_parameters(model)
        bic = None
        nll_bic = fm.get("nll_attach", fm.get("nll_tot"))
        n_subjects = model.dataset_info.get("n_subjects")
        if nll_bic is not None and n_subjects is not None:
            bic = compute_bic(float(nll_bic), n_total_params, n_subjects)

        # Observation model names
        obs_model_names = None
        if hasattr(model, "obs_models"):
            obs_model_names = [om.to_string() for om in model.obs_models]

        # Leaspy version
        try:
            from leaspy import __version__ as version
        except ImportError:
            version = None

        # Group parameters by category
        _meta = _get_display_meta(model)
        params_by_category = {}
        if _meta is not None:
            cats = _build_param_categories(model, _meta)
            cat_names = {
                "population": "Population Parameters",
                "individual_priors": "Individual Parameters",
                "noise": "Noise Model",
            }
            for cat_key, display_name in cat_names.items():
                param_names = cats.get(cat_key, [])
                if param_names:
                    params_by_category[display_name] = {
                        name: model.parameters[name]
                        for name in param_names
                        if name in model.parameters
                    }
        else:
            params_by_category["Parameters"] = dict(model.parameters)

        return cls(
            name=model.name,
            model_type=model.__class__.__name__,
            dimension=model.dimension,
            features=model.features,
            source_dimension=getattr(model, "source_dimension", None),
            n_clusters=getattr(model, "n_clusters", None),
            obs_models=obs_model_names,
            n_total_params=n_total_params,
            nll=nll,
            bic=bic,
            training_info=dict(model.training_info),
            dataset_info=dict(model.dataset_info),
            parameters=params_by_category,
            leaspy_version=version,
            _param_axes=_meta.param_axes if _meta is not None else {},
            _feature_names=model.features,
        )

    # -- Convenience properties ----------------------------------------------

    @property
    def sources(self) -> Optional[list[str]]:
        """Source names (e.g. ``['s0', 's1']``) or ``None``."""
        if self.source_dimension is None:
            return None
        return [f"s{i}" for i in range(self.source_dimension)]

    @property
    def clusters(self) -> Optional[list[str]]:
        """Cluster names (e.g. ``['c0', 'c1']``) or ``None``."""
        if self.n_clusters is None:
            return None
        return [f"c{i}" for i in range(self.n_clusters)]

    @property
    def algorithm(self) -> Optional[str]:
        """Algorithm name used for training."""
        return self.training_info.get("algorithm")

    @property
    def seed(self) -> Optional[int]:
        """Random seed used for training."""
        return self.training_info.get("seed")

    @property
    def n_iter(self) -> Optional[int]:
        """Number of iterations."""
        return self.training_info.get("n_iter")

    @property
    def converged(self) -> Optional[bool]:
        """Whether training converged."""
        return self.training_info.get("converged")

    @property
    def n_subjects(self) -> Optional[int]:
        """Number of subjects in the training dataset."""
        return self.dataset_info.get("n_subjects")

    @property
    def n_visits(self) -> Optional[int]:
        """Total number of visits."""
        return self.dataset_info.get("n_visits")

    @property
    def n_observations(self) -> Optional[int]:
        """Total number of observations."""
        return self.dataset_info.get("n_observations")

    def get_param(self, name: str) -> Optional[Any]:
        """Get a parameter value by name, searching across all categories.

        Parameters
        ----------
        name : str
            Parameter name (e.g. ``'betas_mean'``, ``'tau_std'``).

        Returns
        -------
        value
            The parameter value (typically a ``torch.Tensor``), or ``None``.
        """
        for category_params in self.parameters.values():
            if name in category_params:
                return category_params[name]
        return None

    # -- Display -------------------------------------------------------------

    def __str__(self) -> str:
        lines = []
        sep = "=" * _WIDTH

        # Header
        lines.append(sep)
        lines.append(f"{'Model Summary':^{_WIDTH}}")
        lines.append(sep)
        lines.append(f"Model Name: {self.name}")
        lines.append(f"Model Type: {self.model_type}")

        if self.features is not None:
            feat_str = ", ".join(self.features)
            lines.extend(
                _wrap_text(f"Features ({self.dimension})", feat_str)
            )

        if self.source_dimension is not None:
            sources = [f"Source {i} (s{i})" for i in range(self.source_dimension)]
            lines.extend(
                _wrap_text(
                    f"Sources ({self.source_dimension})",
                    ", ".join(sources),
                )
            )

        if self.n_clusters is not None:
            clusters = [f"Cluster {i} (c{i})" for i in range(self.n_clusters)]
            lines.extend(
                _wrap_text(
                    f"Clusters ({self.n_clusters})",
                    ", ".join(clusters),
                )
            )

        if self.obs_models:
            lines.extend(
                _wrap_text("Observation Models", ", ".join(self.obs_models))
            )

        if self.nll is not None:
            lines.append(f"Neg. Log-Likelihood: {self.nll:.4f}")
        if self.n_total_params is not None:
            lines.append(f"Parameters: {self.n_total_params}")
        if self.bic is not None:
            lines.append(f"BIC: {self.bic:.2f}")

        # Training Metadata
        if self.training_info:
            lines.append("")
            lines.append("Training Metadata")
            lines.append("-" * _WIDTH)
            ti = self.training_info
            lines.append(f"Algorithm: {ti.get('algorithm', 'N/A')}")
            if "seed" in ti:
                lines.append(f"Seed: {ti['seed']}")
            lines.append(f"Iterations: {ti.get('n_iter', 'N/A')}")
            if ti.get("converged") is not None:
                lines.append(f"Converged: {ti['converged']}")

        # Data Context
        if self.dataset_info:
            lines.append("")
            lines.append("Data Context")
            lines.append("-" * _WIDTH)
            di = self.dataset_info
            lines.append(f"Subjects: {di.get('n_subjects', 'N/A')}")
            lines.append(f"Visits: {di.get('n_visits', 'N/A')}")
            lines.append(f"Total Observations: {di.get('n_observations', 'N/A')}")

        # Leaspy Version
        if self.leaspy_version:
            lines.append("")
            lines.append(f"Leaspy Version: {self.leaspy_version}")

        lines.append(sep)

        # Parameters by category
        for category, params in self.parameters.items():
            if params:
                lines.append("")
                lines.append(category)
                lines.append("-" * _WIDTH)
                lines.extend(self._format_parameter_group(params))

        lines.append(sep)
        return "\n".join(lines)

    def help(self) -> None:
        """Print available attributes and their meanings."""
        help_text = f"""
Summary Help
{'=' * 60}

The Summary object provides access to model metadata and parameters.

Usage:
    model.summary()       # Print the formatted summary
    s = model.summary()   # Store to access individual attributes

Available Attributes:

  Model Information:
    name              Model name (str)
    model_type        Model class name, e.g., 'LogisticModel' (str)
    dimension         Number of features (int)
    features          List of feature names (list[str])
    sources           Source names, e.g., ['s0', 's1'] (list[str] or None)
    clusters          Cluster names, e.g., ['c0', 'c1'] (list[str] or None)
    source_dimension  Number of sources (int or None)
    n_clusters        Number of clusters (int or None)
    obs_models        Observation model names (list[str] or None)

  Training:
    algorithm         Algorithm name, e.g., 'mcmc_saem' (str)
    seed              Random seed used (int)
    n_iter            Number of iterations (int)
    converged         Whether training converged (bool or None)
    nll               Negative log-likelihood (float or None)
    n_total_params    Number of free parameters (int)
    bic               Bayesian Information Criterion (float or None)

  Dataset:
    n_subjects        Number of subjects in training data (int)
    n_visits          Total number of visits (int)
    n_observations    Total number of observations (int)

  Parameters:
    parameters        All parameters grouped by category (dict)
    get_param(name)   Get a specific parameter by name

  Other:
    training_info     Full training metadata (TrainingInfo)
    dataset_info      Full dataset statistics (DatasetInfo)
    leaspy_version    Leaspy version used (str)

Examples:
    >>> s = model.summary()
    >>> s.algorithm              # 'mcmc_saem'
    >>> s.seed                   # 42
    >>> s.n_subjects             # 150
    >>> s.get_param('tau_std')   # tensor([10.5])
"""
        print(help_text)
        object.__setattr__(self, "_printed", True)

    # -- Private formatting helpers ------------------------------------------

    def _format_parameter_group(self, params: dict[str, Any]) -> list[str]:
        """Format a group of parameters for display."""
        lines = []
        for name, value in params.items():
            if isinstance(value, torch.Tensor):
                lines.append(self._format_tensor(name, value))
            else:
                lines.append(f"  {name:<18} {value}")
        return lines

    def _format_tensor(self, name: str, value: torch.Tensor) -> str:
        """Format a tensor parameter with axis labels."""
        param_axes = object.__getattribute__(self, "_param_axes")
        feature_names = object.__getattribute__(self, "_feature_names")
        axes = param_axes.get(name, ())

        if value.ndim == 0:
            return f"  {name:<18} {value.item():.4f}"

        elif value.ndim == 1:
            n = len(value)
            if n > 10:
                return f"  {name:<18} Tensor of shape ({n},)"

            axis_name = axes[0] if len(axes) >= 1 else None
            col_labels = get_axis_labels(axis_name, n, feature_names)

            if col_labels:
                header = " " * 20 + "  ".join(f"{lbl:>8}" for lbl in col_labels)
                values = f"  {name:<18}" + "  ".join(
                    f"{v.item():>8.4f}" for v in value
                )
                return header + "\n" + values
            else:
                val_str = "[" + ", ".join(f"{v.item():.4f}" for v in value) + "]"
                return f"  {name:<18} {val_str}"

        elif value.ndim == 2:
            rows, cols = value.shape
            if rows > 8 or cols > 8:
                return f"  {name:<18} Tensor of shape {tuple(value.shape)}"

            row_axis = axes[0] if len(axes) >= 1 else None
            col_axis = axes[1] if len(axes) >= 2 else None
            row_labels = get_axis_labels(row_axis, rows, feature_names)
            col_labels = get_axis_labels(col_axis, cols, feature_names)

            result = [f"  {name}:"]
            if col_labels:
                header = " " * 20 + "  ".join(f"{lbl:>8}" for lbl in col_labels)
                result.append(header)

            for i, row in enumerate(value):
                row_lbl = row_labels[i] if row_labels else f"[{i}]"
                row_str = (
                    f"            {row_lbl:<8}"
                    + "  ".join(f"{v.item():>8.4f}" for v in row)
                )
                result.append(row_str)

            return "\n".join(result)

        else:
            return f"  {name:<18} Tensor of shape {tuple(value.shape)}"
