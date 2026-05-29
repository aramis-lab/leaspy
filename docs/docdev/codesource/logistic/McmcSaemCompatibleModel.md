# McmcSaemCompatibleModel

**Module:** `leaspy.models.mcmc_saem_compatible`
**Inherits from:** [`StatefulModel`](StatefulModel.md)

`McmcSaemCompatibleModel` is the **bridge** between the generic MCMC-SAEM algorithm and a specific mathematical model. While [`StatefulModel`](StatefulModel.md) provides the variables (the `State`), this class defines the **interface** and methods required to optimize them.

It guarantees that the algorithm can perform its two main iterative steps without knowing the underlying model details:
1.  **E-Step**: Sampling individual latent variables (like $\tau_i, \xi_i$).
2.  **M-Step**: Updating population parameters (like $g, v_0$).

## The Optimization Contract

The class enforces three core methods that drive the MCMC-SAEM loop:

1.  **`put_individual_parameters(state, dataset)`** (E-Step)
    Used to initialize or inject individual latent variables into the state. Concrete models (like `LogisticModel`) implement this to map the dataset inputs into the model's specific latent variables.

2.  **`compute_sufficient_statistics(state)`** (M-Step, Part 1)
    After sampling, the algorithm needs to summarize the current state. This method aggregates **sufficient statistics** from all `ModelParameter` nodes in the DAG. It computes the minimal set of numbers (sums, counts) needed to update parameters, rather than storing every sample.

3.  **`update_parameters(state, sufficient_statistics, burn_in)`** (M-Step, Part 2)
    Updates the population parameters using the computed statistics. This is done in a **batch** operation: all new values are computed first, then applied simultaneously to avoid order-dependent inconsistencies.

### Burn-in vs Maximization

The MCMC-SAEM loop is split into two sequential phases — burn-in runs first, then maximization takes over for the remaining iterations:

- **Burn-in phase** (default: first 90% of iterations): Uses a memoryless update rule — each iteration's sufficient statistics directly replace the previous values. This allows the algorithm to move quickly toward a good region of the parameter space without being held back by poor early estimates.
- **Maximization phase** (remaining iterations): Switches to recursive averaging (the "SA" in SAEM), where new statistics are blended with a running average using a decaying step size. This produces more stable, converged parameter estimates.

The split is configured via `n_burn_in_iter_frac` (default `0.9`, i.e. 90% burn-in). It can be passed directly to `model.fit()`:

```python
model.fit(data, "mcmc_saem", n_burn_in_iter_frac=0.8)
```

## Other Responsibilities

Beyond the optimization loop, this class handles:
*   **Observation Models**: Wraps the noise model (e.g., Gaussian noise), providing the likelihood function (`nll_attach`) needed to accept or reject samples.
*   **Individual Trajectories**: The `compute_individual_trajectory` method allows predicting a patient's score **after** the model is trained, effectively powering the `personalize` feature.

## The Algorithm Loop

The following diagram illustrates how the MCMC-SAEM algorithm interacts with this interface at each iteration:

```{image} ../../../_static/images/mcmc.png
:alt: mcmc
:align: center
:width: 80%
```

With the optimization interface defined, the next step is to see how concrete models build on it — starting with [`TimeReparametrizedModel`](TimeReparametrizedModel.md), which adds the time-shifting mechanism shared by all temporal models in Leaspy.
