# TimeReparametrizedModel

**Module:** `leaspy.models.time_reparametrized`
**Inherits from:** [`McmcSaemCompatibleModel`](McmcSaemCompatibleModel.md)

This class introduces the mechanism that makes Leaspy a *disease progression* model rather than a generic regression: **time reparametrization**. It allows each patient to have their own disease onset time and progression speed, so the algorithm can align heterogeneous trajectories onto a common disease timeline.

## The Core Idea: Reparametrized Time

In a cohort, patients don't start declining at the same age or progress at the same rate. To handle this, `TimeReparametrizedModel` transforms each patient's real age into a normalized **reparametrized time**:

$$rt_i(t) = \alpha_i \cdot (t - \tau_i)$$

Where:
- $t$ is the patient's real age at a visit
- $\tau_i$ is the **time shift** — when the patient's disease effectively "starts" (earlier or later than average)
- $\alpha_i = e^{\xi_i}$ is the **acceleration factor** — how fast the patient progresses ($\alpha > 1$ means faster, $\alpha < 1$ means slower)
- $\xi_i$ is the **log-acceleration**, the actual latent variable sampled by MCMC (log-space ensures $\alpha > 0$)

> For the mathematical formulation, see [Temporal Random Effects](../../models.md#temporal-random-effects).

This is implemented as a static method with keyword-only arguments, which allows the DAG to automatically wire it as a `LinkedVariable`:

```python
@staticmethod
def time_reparametrization(*, t, alpha, tau):
    return alpha * (t - tau)
```

## Variables Defined

`get_variables_specs()` extends the parent's specs with the temporal (and optionally spatial) variables. Here's what this class adds:

### Always present (temporal variability)

| Variable | Type | Description |
|---|---|---|
| `tau_mean` | `ModelParameter` | Prior mean of $\tau$, learned by M-step |
| `tau_std` | `ModelParameter` | Prior std of $\tau$, learned by M-step |
| `xi_std` | `ModelParameter` | Prior std of $\xi$, learned by M-step |
| `xi` | `IndividualLatentVariable` | Log-acceleration per patient, $\xi_i \sim \mathcal{N}(\text{xi\_mean}, \text{xi\_std})$ |
| `tau` | `IndividualLatentVariable` | Time shift per patient, $\tau_i \sim \mathcal{N}(\text{tau\_mean}, \text{tau\_std})$ |
| `alpha` | `LinkedVariable` | $\alpha_i = e^{\xi_i}$, computed via `Exp("xi")` |
| `rt` | `LinkedVariable` | Reparametrized time, computed via `time_reparametrization` |

> Note: `xi_mean` is a `Hyperparameter` fixed at 0 (defined in the parent class), because the average acceleration should be 1 ($e^0 = 1$). Only its std is learned.

### Conditional (spatial variability, when `source_dimension >= 1`)

When the model has multiple features and `source_dimension > 0`, patients can also differ in *which* features decline faster. This is captured by **sources**:

| Variable | Type | Description |
|---|---|---|
| `sources_mean` | `Hyperparameter` | Fixed at 0 (prior mean of sources) |
| `sources_std` | `Hyperparameter` | Fixed at 1.0 (prior std of sources) |
| `betas_mean` | `ModelParameter` | Population-level spatial mixing coefficients, shape `(dimension-1, source_dimension)` |
| `betas_std` | `Hyperparameter` | Fixed at 0.01 (tight prior on betas) |
| `sources` | `IndividualLatentVariable` | Per-patient spatial components |
| `betas` | `PopulationLatentVariable` | Population-level mixing coefficients, sampled by MCMC |
| `mixing_matrix` | `LinkedVariable` | `orthonormal_basis @ betas.T` |
| `space_shifts` | `LinkedVariable` | `sources @ mixing_matrix` — per-patient spatial deviations |

## What This Class Implements

`TimeReparametrizedModel` fills in several abstract methods from `McmcSaemCompatibleModel`:

- **`put_individual_parameters(state, dataset)`**: Initializes $\xi_i$ and $\tau_i$ by sampling from their priors if not already set in the state.
- **`_audit_individual_parameters(individual_parameters)`**: Validates that the parameter dict contains exactly `{xi, tau}` (or `{xi, tau, sources}` if the model has sources), checks size consistency, and tensorizes them.
- **`_load_hyperparameters(hyperparameters)`**: Loads `features`, `dimension`, and `source_dimension` from a saved model dict, with validation.

It also overrides:
- **`_validate_compatibility_of_dataset(dataset)`**: Auto-sets `source_dimension` if it was left as `None`.
- **`to_dict()`**: Adds `source_dimension` and optionally the `mixing_matrix` to the serialized output.

## What Comes Next

`TimeReparametrizedModel` defines *when* each patient is on the disease timeline, but not *what the disease curve looks like*. The shape of the progression curve — the geometric structure — is added by [`RiemanianManifoldModel`](RiemanianManifoldModel.md), which introduces the metric, velocity, and the model equation itself.
