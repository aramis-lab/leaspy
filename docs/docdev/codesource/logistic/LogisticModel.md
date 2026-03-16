# LogisticModel

**Module:** `leaspy.models.logistic`
**Inherits from:** [`LogisticInitializationMixin`](LogisticInitializationMixin.md), [`RiemanianManifoldModel`](RiemanianManifoldModel.md)

`LogisticModel` is the concrete class that gives the progression curve its shape: a **logistic sigmoid**. It implements the two abstract methods left open by `RiemanianManifoldModel` — `metric` and `model_with_sources` — and adds the parameter $g$ that controls the curve's position.

This is the most commonly used model in Leaspy, suitable for biomarkers that follow an S-shaped trajectory from normality (0) to pathology (1).

## The Metric

The Riemannian metric for the logistic manifold is:

$$G_k = \frac{(1 + g_k)^2}{g_k}$$

In code:
```python
@staticmethod
def metric(*, g):
    return (g + 1) ** 2 / g
```

$g_k$ is a per-feature population parameter that controls the value of the logistic curve at the reference time. The metric $G_k$ scales the contribution of velocity and space shifts in logit space so that individual reparametrizations produce curves of the same shape — purely translated, not distorted.

## The Model Equation

`model_with_sources` computes biomarker values in two steps:

**1. Logit computation:**

$$\text{logit}_{i,t,k} = G_k \cdot \left( v_{0,k} \cdot rt_{i,t} + \delta_{i,k} \right) - \ln(g_k)$$

Where:
- $G_k$ is the metric (population-level, per feature)
- $v_{0,k}$ is the population velocity (per feature)
- $rt_{i,t}$ is the reparametrized time (per patient, per visit)
- $\delta_{i,k}$ is the space shift from sources (per patient, per feature) — zero if no sources
- $\ln(g_k)$ is the offset that sets the inflection point

In code:
```python
w_model_logit = metric * (v0 * rt + space_shifts) - torch.log(g)
```

**2. Sigmoid activation:**

$$y_{i,t,k} = \sigma(\text{logit}_{i,t,k}) = \frac{1}{1 + e^{-\text{logit}_{i,t,k}}}$$

This maps the logit to $(0, 1)$, producing the final biomarker estimate. The code uses `WeightedTensor` to handle missing data — NaN observations are masked before the sigmoid and re-masked after.

## Variables Defined

`get_variables_specs()` adds the logistic-specific parameter $g$ to the parent's variables:

| Variable | Type | Description |
|---|---|---|
| `log_g_mean` | `ModelParameter` | Prior mean of $\log g$, shape `(dimension,)`, learned by M-step |
| `log_g_std` | `Hyperparameter` | Fixed at 0.01 (tight prior on `log_g`) |
| `log_g` | `PopulationLatentVariable` | Log of the position parameter, sampled from a Normal prior with mean `log_g_mean` and std `log_g_std` |
| `g` | `LinkedVariable` | Position parameter, $g = e^{\log g}$ via `Exp("log_g")` |

> The log-space parameterization ensures $g > 0$ and provides numerically stable inference.

## Initialization (Mixin)

Non-linear models are sensitive to starting values. `LogisticModel` inherits `_compute_initial_values_for_model_parameters()` from `LogisticInitializationMixin`, which computes initial parameter values from the dataset:

1. Extracts per-patient statistics: mean slopes, mean values, and mean ages
2. Transforms them into model parameters:
   - `log_g_mean` $= \ln(1/\bar{y} - 1)$ — log-odds of mean feature values
   - `log_v0_mean` — log-transformed slopes
   - `tau_mean` — mean age across patients
   - `tau_std`, `xi_std` — from model defaults
   - `betas_mean` — initialized to zeros (if sources)
3. Clamps values to $[0.01, 0.99]$ to avoid boundary issues in the log-odds transform
4. Supports two modes: `DEFAULT` (uses means) and `RANDOM` (samples from estimated distributions)

> See [`LogisticInitializationMixin`](LogisticInitializationMixin.md) for more details.

## What Comes Next

This concludes the model definition hierarchy. From here:
- To understand how noisy observations connect to the model output, see [Observation Models](ObservationModel.md).
- To see how the full variable graph looks, see the [DAG](DAG.md).
