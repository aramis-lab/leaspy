---
orphan: true
---
# Covariate model architecture in Leaspy

This document describes what was done to implement `CovariateLogisticModel` in Leaspy.

There are two independent parts:

1. **Model side**: how `CovariateLogisticModel` was built starting from the structure of `LogisticModelSchiratti`.
2. **Data side**: how covariates are read, validated, and injected into the model.

---

## 1. Model side: from Schiratti's parametrization to the covariate model

### 1.1 Why start from Schiratti's parametrization and not the standard Leaspy model

The standard Leaspy logistic model (`LogisticModel` / `TimeReparametrizedModel` / `RiemannianManifoldModel`) simplifies the original parametrization of Schiratti et al. (2017): instead of keeping a population-level `t0` separate from an individual time-shift `τᵢ`, it absorbs `t0` directly into the mean of the prior on `τᵢ` (`τᵢ ~ N(τ, σ²_τ)`, with `τ` replacing `t0`).

This simplification is a problem for the covariate model: we want a covariate's effect to act on an explicit **population-level parameter** `t0` (`t0_patient = t0 + covariates @ δ_t0`), not on the mean of an individual random effect. So the original parametrization, where `t0` exists as its own `ModelParameter` distinct from `τᵢ`, had to be reintroduced. That is the role of `LogisticModelSchiratti` (and its supporting files `time_reparametrized_Schiratti.py`, `riemannian_manifold_Schiratti.py`): an architectural prerequisite, not a scientific contribution in itself.

### 1.2 Important nuance: the covariate model does *not* inherit from the Schiratti classes

`CovariateLogisticModel` and `LogisticModelSchiratti` are **two separate, parallel class hierarchies**, both ultimately inheriting from `McmcSaemCompatibleModel`, but **not from each other**. Concretely:

- `TimeReparametrizedModelSchiratti(McmcSaemCompatibleModel)` vs. `CovariateTimeReparametrizedModel(McmcSaemCompatibleModel)`
- `RiemannianManifoldModelSchiratti(TimeReparametrizedModelSchiratti)` vs. `CovariateRiemannianManifoldModel(CovariateTimeReparametrizedModel)`
- `LogisticModelSchiratti(..., RiemannianManifoldModelSchiratti)` vs. `CovariateLogisticModel(..., CovariateRiemannianManifoldModel)`

In other words, the covariate files reuse the *structure* (the same 3-level class layout, the same method names, the same overall logic for `get_variables_specs`, sufficient statistics, `metric`, etc.) of the Schiratti files, but as an independent copy that was then edited to add the covariate-specific pieces — not as a subclass that overrides only what changes. This is why the two hierarchies need to be kept in sync manually if a shared behavior changes (e.g. a bugfix in `_center_xi_realizations` would need to be applied in both `riemannian_manifold_Schiratti.py` and `covariate_riemannian_manifold.py`).

### 1.3 File correspondence

| Schiratti (prerequisite, independent hierarchy)                             | Covariate (parallel hierarchy)                                              | Role                                                                                                                 |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `time_reparametrized_Schiratti.py` (`TimeReparametrizedModelSchiratti`) | `covariate_time_reparametrized.py` (`CovariateTimeReparametrizedModel`) | Defines`t0`, `τ`, `ξ`, the time reparametrization, and (covariate side) `δ_t0`, `γ_t0`, `t0_patient` |
| `riemannian_manifold_Schiratti.py` (`RiemannianManifoldModelSchiratti`) | `covariate_riemannian_manifold.py` (`CovariateRiemannianManifoldModel`) | Defines`v0`, the metric, and (covariate side) `δ_v0`, `γ_v0`, `v0_patient`, `metric_patient`             |
| `logistic_Schiratti.py` (`LogisticModelSchiratti`)                      | `covariate_logistic.py` (`CovariateLogisticModel`)                      | Defines`g`, the logistic model formula, initialization, and (covariate side) `δ_g`, `γ_g`, `g_patient`     |

At each level, the `covariate_*` file reproduces the equivalent Schiratti file almost line for line, and only a small, identifiable set of methods actually differ. These are the methods that had to be touched to make the covariate mechanism work — everything else (e.g. `_center_xi_realizations`, `_center_tau_realizations`, `compute_sufficient_statistics`) is copied unchanged:

* **`covariate_time_reparametrized.py`** — 2 methods differ from `time_reparametrized_Schiratti.py`:
  * `time_reparametrization`: takes `t0_patient` instead of `t0` (`alpha * (t - t0_patient - tau)`).
  * `get_variables_specs`: extended with the `δ_t0` / `γ_t0` / `t0_patient` block (§1.4).
* **`covariate_riemannian_manifold.py`** — 3 methods differ from `riemannian_manifold_Schiratti.py`, and one is added:
  * `__init__` (`default_variables_to_track`): the list of tracked variables gains `delta_t0`, `delta_g`, `delta_v0`, `gamma_t0`, `gamma_g`, `gamma_v0`.
  * `get_variables_specs`: extended with the `δ_v0` / `γ_v0` / `v0_patient` block (§1.4).
  * `model_with_sources` (abstract signature): its parameters change from `metric, v0, g` (population-only, in the Schiratti version) to `metric_patient, v0_patient, g_patient` (patient-specific) — the actual body is only implemented one level below, in `covariate_logistic.py`, but the abstract signature already commits to the patient-specific parameters at this level.
  * An abstract `metric_patient` static method is also declared here (implemented in `covariate_logistic.py`).
* **`covariate_logistic.py`** — 2 methods differ from `logistic_Schiratti.py`, one is added:
  * `get_variables_specs`: extended with the `δ_g` / `γ_g` / `g_patient` block (§1.4).
  * `model_with_sources`: the formula itself is rewritten to consume `metric_patient`, `v0_patient`, `g_patient` instead of `metric`, `v0`, `g`.
  * `metric_patient` (new static method): `(g_patient + 1)**2 / g_patient` — the patient-specific counterpart of the existing `metric` method, needed because the metric now has to be recomputed per patient from `g_patient` rather than once at the population level from `g`.

### 1.4 What had to be added at each level — the pattern repeated 3 times

For each population parameter affected by covariates (`t0`, `g`, `v0`), the same pattern is repeated:

1. **A selection mask γ**: `gamma_t0` / `gamma_g` / `gamma_v0`, a Bernoulli(`pi_*`) latent variable — `pi_*` being a fixed hyperparameter (prior probability that a given covariate has an effect, e.g. 0.5).
2. **A covariate effect δ**: `delta_t0` / `delta_g` / `delta_v0`, a `MultivariateNormal(delta_*_mean, delta_*_sigma)` latent variable, with a **conditional prior** (`nll_prior`) centered on `δ_*_cond_mean = γ_* ⊙ δ_*_mean`.
3. **A masked effect**: `delta_*_masked = γ_* ⊙ δ_*` (via `Prod`) — this masked term is what actually enters the model computation, not raw `delta_*`.
4. **A patient-specific parameter**: `t0_patient`, `log_g_patient`, `log_v0_patient`, computed via `Affine`/`AffineMatrix`: `base + covariates @ delta_masked` (or `.T`, depending on shape).
5. **A final derived variable** exposed to the model computation: `g_patient = exp(log_g_patient)`, `v0_patient = exp(log_v0_patient)`.

Concretely, in `covariate_time_reparametrized.py`:

```python
delta_t0_mean=ModelParameter.for_pop_mean_condi("delta_t0", "gamma_t0", shape=(self.nb_cov,)),
delta_t0_sigma=Hyperparameter(torch.eye(self.nb_cov) * 1.0),
pi_t0=Hyperparameter(0.2 * torch.ones(self.nb_cov)),
gamma_t0=PopulationLatentVariable(Bernoulli("pi_t0")),
delta_t0_cond_mean=LinkedVariable(Prod("gamma_t0", "delta_t0_mean")),
delta_t0=PopulationLatentVariable(
    MultivariateNormal("delta_t0_mean", "delta_t0_sigma"),
    sampling_kws={"scale": 1},
    nll_prior=MultivariateNormal("delta_t0_cond_mean", "delta_t0_sigma"),
),
delta_t0_masked=LinkedVariable(Prod("gamma_t0", "delta_t0")),
t0_patient=LinkedVariable(Affine("t0", "delta_t0_masked", "covariates")),
```

and the time reparametrization now uses `t0_patient` (`alpha * (t - t0_patient - tau)`), unlike the Schiratti version which uses `t0` directly.

The same pattern is repeated in `covariate_riemannian_manifold.py` for `g`/`v0` (using `AffineMatrix` instead of `Affine`, since `g` and `v0` have one dimension per feature while `t0` is scalar), and propagates down to the final logistic model formula in `covariate_logistic.py`, where `model_with_sources` uses `v0_patient`, `g_patient`, `metric_patient` (patient-specific) instead of `v0`, `g`, `metric` (population-only, as in the Schiratti version).

#### Why two means in the definition of delta_*: `delta_t0_mean` vs. `delta_t0_cond_mean`

This distinction is the crux of the whole conditional-prior mechanism, so it is worth detailing:

- **`delta_g_mean`** is the actual `ModelParameter` — the quantity re-estimated at every M-step of MCMC-SAEM, via `ModelParameter.for_pop_mean_condi` (see §1.5). Its update rule is `delta_mean = E[γ⊙δ] / (E[γ] + ε)`: it only averages `δ` over the (stochastic-approximation-weighted) episodes where `γ=1`. This means `delta_g_mean` tracks the *magnitude of the effect when it is active*, and is **not** dragged toward 0 just because some MCMC iterations sampled `γ=0`.
- **`delta_g_cond_mean` = `γ_g ⊙ delta_g_mean`** is a `LinkedVariable` (a pure function of `γ_g` and `delta_g_mean`, not a model parameter of its own), recomputed at every iteration from the *current* sampled value of `γ_g`. It defines the mean of `nll_prior`: the regularization term (`nll_regul`) added to the data-fit term whenever a candidate value of `delta_g` is evaluated — during MCMC sampling of `delta_g` itself, and in the overall likelihood computation.

Why not simply use `delta_g_mean` as the mean of the `nll_prior` directly, skipping `delta_g_cond_mean`? Because the two variables serve different, and partly conflicting, purposes:

- The **regularization term** (`delta_g_cond_mean`) needs to depend on the *current* value of `γ_g` at every iteration: when `γ_g=0` the effective prior mean must collapse to 0, so that the NLL penalizes any non-zero `δ` heavily — this is what implements the "spike" behaviour (covariate effectively switched off) of the spike-and-slab-like construction. If the prior mean were `delta_g_mean` (unconditional) instead, `δ` would remain free to drift to any value even when `γ_g=0`, and the mask would have no real regularizing effect on `δ`.
- The **model parameter** (`delta_g_mean`), on the other hand, must **not** be forced toward 0 by episodes where `γ_g` happened to be sampled as 0 — otherwise, every time `γ_g` flips to 0 for a while (which happens often during burn-in, or simply by chance under the Bernoulli prior), the accumulated estimate of the effect size would be diluted or erased, making it much harder for `γ_g` to be resampled back to 1 later (since the model would have "forgotten" that the effect used to be large). Restricting the sufficient-statistics update to `γ=1` episodes (via `for_pop_mean_condi`) keeps `delta_g_mean` informative about the effect's magnitude across the whole run, independently of the current state of the mask.

In short: `delta_g_mean` is *what the effect looks like when it's on*, estimated only from "on" episodes; `delta_g_cond_mean` is *what the prior currently expects*, which switches to 0 whenever the mask says the effect is off. Separating them avoids conflating "learning the size of an effect" with "deciding whether the effect is present."

### 1.5 Technical building blocks required for this pattern to work

These pieces did not exist in Leaspy before and had to be added as supporting infrastructure:

**a) `ModelParameter.for_pop_mean_condi` (`src/leaspy/variables/specs.py`)**
A new factory method, mirroring the existing `for_pop_mean`, but for a parameter whose prior is *conditional* on a binary mask (`p(δ|γ) = N(γ⊙δ_mean, Σ)`). It defines:

- the **sufficient statistics** to collect: `γ` and `γ⊙δ` (this is the `S17_new` / `S19` / `S21` referenced in the thesis notes);
- the **M-step update rule**: `delta_mean = E[γ⊙δ] / (E[γ] + ε)` (implemented in `compute_pop_mean_cond_from_suff_stats`, `src/leaspy/models/utilities.py`).

This function is what makes it possible to express the pattern from §1.4 in a single line per parameter, instead of hand-writing the sufficient-statistics logic separately for `t0`, `g`, and `v0`.

**b) `MultivariateNormalFamily` (`src/leaspy/variables/distributions.py`)**
A new stateless distribution family that did not exist in Leaspy before this model. Unlike the existing `NormalFamily` (independent, scalar-per-coordinate Gaussian), `δ_t0`/`δ_g`/`δ_v0` need a genuine multivariate Gaussian prior with a full covariance matrix `Σ` over the covariate directions — and, for `g`/`v0`, batched over features (covariance of shape `(K, N_cov, N_cov)`). `MultivariateNormalFamily` implements the NLL and its gradient for this case by hand (via a Cholesky decomposition of `Σ`, to stay numerically stable and support the batched case), and is exposed as `MultivariateNormal = SymbolicDistribution.bound_to(MultivariateNormalFamily)`, used for `delta_t0`, `delta_g`, `delta_v0` in `get_variables_specs()`.

**c) `BernoulliFamily` (`src/leaspy/variables/distributions.py`)**
Another new stateless distribution family, added alongside `MultivariateNormalFamily`. Provides the NLL and its gradient for a Bernoulli variable, which lets γ be treated like any other latent variable in the framework (computation of `nll_attach`, `nll_regul`, etc.), exposed as `Bernoulli = SymbolicDistribution.bound_to(BernoulliFamily)` and used in `get_variables_specs()`.

**d) `BernoulliDiscreteSampler` (`src/leaspy/samplers/gibbs.py`)**
Every other population variable in Leaspy is sampled via a continuous Gaussian random-walk Metropolis-Hastings step, which does not make sense for a discrete two-valued variable. This new sampler directly evaluates the total NLL at both possible values (0 and 1) for each coordinate of γ, and samples exactly from the conditional posterior — a true Gibbs step, with no rejection. Routing to this sampler is automatic in `algo_with_samplers.py`: any variable whose `dist_family` is `BernoulliFamily` is sent there instead of the default Gaussian sampler.

**e) `Affine` / `AffineMatrix` (`src/leaspy/utils/functional/_functions.py`, `_utils.py`)**
Two new named functions (in the sense of Leaspy's `NamedInputFunction` framework):

- `Affine(base, delta, covariates)` → `base + covariates @ delta` — used for `t0_patient` (scalar base, one δ per covariate).
- `AffineMatrix(base, delta, covariates)` → `base + covariates @ delta.T` — used for `log_g_patient` / `log_v0_patient` (base per feature, δ of shape `(K, N_cov)`).

These two functions are what materialize, in the computation graph, the move from a population parameter to a patient-specific one.

### 1.6 Initialization from a standard logistic model

`CovariateLogisticInitializationMixin` (in `covariate_logistic.py`) adds an optional `init_from_model` argument to the constructor: if provided, the parameters shared with an already-fitted standard logistic model (`t0_mean`, `log_g_mean`, `log_v0_mean`, `betas_mean`, `tau_std`, `xi_std`, `noise_std`) are directly reused, while the parameters specific to the covariate model are set to their neutral starting values: `delta_*_mean=0` and `gamma_*=1` (masks all start "active"). This speeds up and stabilizes MCMC-SAEM convergence, by avoiding re-learning the population-average trajectory from scratch. Without `init_from_model`, the "from scratch" initialization (empirical computation of patient slopes/values/times) is identical to `LogisticModelSchiratti`'s, with the same addition of `delta_*_mean=0` and `gamma_*=1`.

---

## 2. Data side: integrating covariates into the input pipeline

The covariate reading pipeline follows the same design pattern already used by the other dataframe readers in Leaspy (`EventDataframeDataReader`, `VisitDataframeDataReader`): a subclass of `AbstractDataframeDataReader` implementing the same contract (`_check_headers`, `_set_index`, `_clean_dataframe`, `_load_individuals_data`). What's new is the covariate-specific content of these methods, not the overall reading architecture. It touches 5 files in `src/leaspy/io/data/`.

### 2.1 `covariate_dataframe_data_reader.py` — `CovariateDataframeDataReader` (new)

This is the entry point: it turns a pandas `DataFrame` (with ID, TIME, feature columns, plus covariate columns) into `IndividualData` objects. It delegates the reading of longitudinal visits to the existing `VisitDataframeDataReader`, and adds the covariate-specific handling:

- **`_clean_dataframe_covariates`**: this method carries all the validation logic. It performs, in order:

  1. Checks that the dataframe columns match the declared `covariate_names`.
  2. Rejects missing values (NaN) on any covariate.
  3. Casts to `float` (covariates can be binary *or* continuous — see §2.2).
  4. Checks that there is exactly one covariate value per patient (covariates are assumed **constant over time**, not visit-dependent), then deduplicates (`groupby("ID").first()`).
  5. **Scale warning**: for a covariate that isn't already binary {0,1}, a `warning` is raised if its mean/std deviate too much from a standardized scale (mean ≈ 0, std ≈ 1) — the model's priors on δ and the MCMC proposal step sizes are calibrated for that scale.
  6. **Identifiability condition (a)**: each covariate must have at least 2 distinct values (otherwise no effect is estimable).
  7. **Identifiability condition (b)**: the `[intercept, covariates]` matrix must be full rank (checked via SVD). Since the model has a free intercept per feature (`t0`, `log_g`, `log_v0`), covariates must not be collinear with each other or with a constant column — otherwise the covariate effect and the intercept are not separately identifiable. On failure, the error explicitly lists the covariates involved in the detected linear dependency.
- **`_clean_dataframe`**: orchestrates the cleaning of visits (delegated) and of covariates (`_clean_dataframe_covariates`), then checks that the set of patients with visits exactly matches the set of patients with a covariate, before joining the two (`df_visit.join(df_covariate)`).
- **`_load_individuals_data_covariates`**: for each patient, converts their covariate row to a list and attaches it via `IndividualData.add_covariates(...)`.

### 2.2 `int` → `float` typing (`data.py`, `dataset.py`, `individual_data.py`)

The covariate typing (the `covariates` parameter, the `Dataset.covariates` attribute, the `IndividualData.add_covariates` method) was changed from `int`/`torch.IntTensor` to `float`/`torch.FloatTensor` across the three files involved. This is necessary to support **continuous covariates** (e.g. a standardized N(0,1) score) in addition to binary ones — the original integer typing was inherited from a context where only discrete (mostly binary) covariates were considered.

### 2.3 Routing: making the reader reachable

* **`factory.py`** (`src/leaspy/io/data/factory.py`): adds `DataframeDataReaderFactoryInput.COVARIATE = "covariate"` and registers `CovariateDataframeDataReader` in `dataframe_data_reader_factory`, so that `Data.from_dataframe(df, data_type="covariate", ...)` routes correctly to this reader.
* **`Data._from_reader`** : if the reader has a `covariate_names` attribute (which `CovariateDataframeDataReader` does), it is copied onto the `Data` object.

### 2.4 The bridge between the data pipeline and the model — the actual connection point

Everything described in §2.1–2.3 only gets covariate values into `Dataset.covariates`, a plain tensor sitting on the data side. **On its own, this tensor is invisible to the model** — nothing in §1 (the `Affine`/`AffineMatrix` formulas, `t0_patient`, `g_patient`, etc.) can reference it unless it is explicitly exposed as a named variable inside the model's own state. This is the single connection point where the two sides of this document actually meet, and it lives entirely on the model side, in `covariate_time_reparametrized.py` (the base of the covariate hierarchy — see §1):

* **`covariates=DataVariable()`** , declared in `CovariateTimeReparametrizedModel.get_variables_specs()`, declares a variable named `covariates` inside the model's variable graph — a placeholder that says "this value comes from the dataset, not from a prior or a computation."
* **`put_data_variables()`** , overridden in the same file, is what actually fills that placeholder at run time:

python

```python
  def put_data_variables(self, state: State, dataset: Dataset) -> None:
      super().put_data_variables(state, dataset)
      covariates_tensor = dataset.covariates.clone().detach().to(torch.float32)
      state["covariates"] = WeightedTensor(covariates_tensor)
```

It copies `dataset.covariates` into the model's `state` under the key `"covariates"`. From that point on, `"covariates"` is a name the symbolic formulas can reference — which is exactly what `Affine("t0", "delta_t0_masked", "covariates")` and `AffineMatrix(...)` do (§1.4).

**`nb_cov`** is the other piece needed to make this bridge:

python

```python
@property
def nb_cov(self) -> int:
    covariates = self.dataset.covariate_names
    ...
```

It is a model property, read dynamically from `self.dataset.covariate_names` (itself populated by the reader, §2.1). Every `shape=(self.nb_cov,)` or `shape=(self.dimension, self.nb_cov)` used when declaring `delta_t0`, `delta_g`, `delta_v0` in `get_variables_specs()` (§1.4) depends on it. This is what allows the model to work with any number of covariates without hard-coding it anywhere — the shapes of `δ`, `γ`, `Σ` all derive from `nb_cov` at model-construction time.
