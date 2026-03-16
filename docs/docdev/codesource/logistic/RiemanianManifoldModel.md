# RiemanianManifoldModel

**Module:** `leaspy.models.riemanian_manifold`
**Inherits from:** [`TimeReparametrizedModel`](TimeReparametrizedModel.md)

While [`TimeReparametrizedModel`](TimeReparametrizedModel.md) defines *when* each patient progresses (time shifts and acceleration), `RiemanianManifoldModel` defines the geometric structure that governs *how* multiple biomarkers evolve together. It ensures that the progression curve is a geodesic (shortest path) on a Riemannian manifold, so that multivariate trajectories remain geometrically consistent.

This class is **largely abstract** — it sets up the geometric framework but delegates the actual curve shape to concrete subclasses like `LogisticModel` or `JointModel`.

## Abstract Methods: What Subclasses Must Provide

`RiemanianManifoldModel` does not know what the progression curve looks like. It forces subclasses to define two things:

- **`metric(*, g)`** (abstract, staticmethod): The Riemannian metric tensor, computed from the parameter $g$. This is what determines the geometry of the space. For example, `LogisticModel` returns $(g + 1)^2 / g$, while `LinearModel` returns a constant metric of 1 (Euclidean space).

- **`model_with_sources(*, rt, space_shifts, metric, v0, g)`** (abstract, classmethod): The actual equation that computes biomarker values from reparametrized time, spatial shifts, and population parameters.

- **`model_no_sources(*, rt, metric, v0, g)`** (concrete): Delegates to `model_with_sources` with `space_shifts=torch.zeros((1, 1))`, effectively removing spatial effects.

## Centering $\xi$ (Speed Factor): Identifiability Fix

In a mixed-effects model on a manifold, there's a non-identifiability problem: if the population velocity $v_0$ increases while individual accelerations $\xi_i$ decrease by the same amount, the resulting trajectories are identical — the model can't distinguish between the two.

To fix this, `RiemanianManifoldModel` overrides `compute_sufficient_statistics()` to center $\xi$ before each M-step:

1. Compute $\bar{\xi} = \text{mean}(\xi_i)$
2. Center: $\xi_i \leftarrow \xi_i - \bar{\xi}$
3. Compensate: $\log v_0 \leftarrow \log v_0 + \bar{\xi}$

**Why this works**: The mean $\bar{\xi}$ removed from individuals is transferred to $\log v_0$, so the overall model output doesn't change — it's just a redistribution between population and individual parameters. After this operation, the average $\xi_i$ is exactly 0, which means the "typical" patient progresses at exactly the population speed $v_0$.

This operation is safe because it only changes the *magnitude* of $v_0$, not its *direction* — so the orthonormal basis (which depends on the direction of $v_0$) doesn't need to be recomputed.

## Variables Defined

`get_variables_specs()` extends the parent's specs with the geometric variables:

| Variable | Type | Description |
|---|---|---|
| `xi_mean` | `Hyperparameter` | Fixed at 0.0 (prior mean of $\xi$, because average acceleration = $e^0 = 1$) |
| `log_v0_mean` | `ModelParameter` | Prior mean of log-velocities, shape `(dimension,)`, learned by M-step |
| `log_v0_std` | `Hyperparameter` | Fixed at 0.01 (tight prior on `log_v0`) |
| `log_v0` | `PopulationLatentVariable` | Log-velocity vector, sampled from a Normal prior with mean `log_v0_mean` and std `log_v0_std` |
| `v0` | `LinkedVariable` | Velocity vector, $v_0 = e^{\log v_0}$ via `Exp("log_v0")` |
| `metric` | `LinkedVariable` | Riemannian metric, delegates to the abstract `metric` method |
| `model` | `LinkedVariable` | Model output — delegates to `model_with_sources` or `model_no_sources` depending on `source_dimension` |

### Additional variables when `source_dimension >= 1`

| Variable | Type | Description |
|---|---|---|
| `metric_sqr` | `LinkedVariable` | Squared metric via `Sqr("metric")`, used to build the orthonormal basis |
| `orthonormal_basis` | `LinkedVariable` | Basis orthogonal to $v_0$ w.r.t. the metric, shape `(dimension, dimension-1)`, built via Householder decomposition from `v0` and `metric_sqr` |

## What Comes Next

`RiemanianManifoldModel` defines the geometric rules but leaves the curve equation abstract. The [LogisticModel](LogisticModel.md) (or `LinearModel`) implements `metric` and `model_with_sources`, giving the manifold its concrete shape — e.g., a sigmoid surface for the logistic case.
