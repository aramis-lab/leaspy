# LogisticInitializationMixin

**Module:** `leaspy.models.logistic`
**Used by:** [`LogisticModel`](LogisticModel.md)

> This is a helper mixin for `LogisticModel`. You rarely interact with it directly unless you are customizing how the model estimates its starting parameters.

Before running MCMC-SAEM, we need reasonable starting values for the model parameters. Random initialization often leads to poor convergence in non-linear models. This mixin provides data-driven heuristics to compute good initial values.

## Why a Separate Mixin?

Initialization logic is kept separate from the model definition for two reasons:
1. **Separation of concerns**: The model class defines the mathematics; the mixin handles the estimation heuristics.
2. **Reusability**: Other models (e.g., `SharedSpeedLogisticModel`) can reuse or override the initialization without duplicating model code.

## The Method: `_compute_initial_values_for_model_parameters`

This is the only method in the mixin. It is called during `StatefulModel.initialize()` → `_initialize_model_parameters()`, after the State and DAG are built but before the algorithm starts.

### Step 1: Extract Patient Statistics

The dataset is converted to a pandas DataFrame, then three distributions are computed using helper functions from `leaspy.models.utilities`:

- **`compute_patient_slopes_distribution(df)`**: For each feature, runs a linear regression per patient (time vs value) and returns the mean and std of slopes across patients.
- **`compute_patient_values_distribution(df)`**: Returns mean and std of each feature's values across the dataset.
- **`compute_patient_time_distribution(df)`**: Returns mean and std of visit ages.

### Step 2: Choose Initialization Mode

The `initialization_method` attribute (set at model creation, default `"default"`) controls how statistics are used:

- **`DEFAULT`**: Uses patient means directly (slopes_mu, values_mu, time_mu). Betas initialized to zeros.
- **`RANDOM`**: Samples from normal distributions centered on patient means with estimated standard deviations.

### Step 3: Transform into Model Parameters

| Statistic | Transformation | Parameter |
|---|---|---|
| Feature values | $\ln(1/\bar{y} - 1)$ (log-odds) | `log_g_mean` |
| Slopes | `get_log_velocities()` — clamps to $\geq 0.01$, then takes $\ln$ | `log_v0_mean` |
| Mean age | Used directly | `tau_mean` |
| — | Model defaults (5.0, 0.5) | `tau_std`, `xi_std` |
| — | Zeros, shape `(dimension-1, source_dimension)` | `betas_mean` (if sources) |

Feature values are clamped to $[0.01, 0.99]$ before the log-odds transform to avoid numerical issues at the boundaries.

All parameters are rounded to ~$10^{-5}$ precision via `torch_round()` and converted to `float32`.

### Step 4: Observation Model Adjustment

If the observation model is a `FullGaussianObservationModel`, the `noise_std` parameter is expanded to match the shape expected by the observation model (scalar or per-feature diagonal).

## Utility Functions

The helper functions used in Step 1 live in `leaspy.models.utilities`:

| Function | What it does |
|---|---|
| `compute_patient_slopes_distribution(df)` | Linear regression per patient per feature → mean/std of slopes |
| `compute_patient_values_distribution(df)` | Mean/std of feature values across dataset |
| `compute_patient_time_distribution(df)` | Mean/std of visit ages |
| `get_log_velocities(velocities, features)` | Clamps negative velocities (with warning), returns $\ln(\text{velocities})$ |
| `torch_round(t, tol)` | Rounds tensor to precision `1/tol` (default ~$10^{-5}$) |
