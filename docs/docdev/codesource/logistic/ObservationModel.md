# Observation Models

**Modules:** `leaspy.models.obs_models`

While the **Logistic Model** defines the ideal, noise-free disease trajectory, real-world data is messy. The **Observation Model** bridges this gap by defining the probability of observing specific data points given the model's prediction:

$$ P(y_{observed} | y_{model}) $$

## The Abstract Base: `ObservationModel`

This interface describes how data "attaches" to the model, formally defining the **Negative Log-Likelihood (NLL)** that algorithms minimize. Its main responsibilities are:

1.  **Data Connection**: Extracts relevant features from the raw `Dataset`.
2.  **Likelihood Computation**: Defines the statistical distribution of the residuals (Gaussian, Bernoulli, or Weibull depending on the model).
3.  **Variable Generation**: Creates the `nll_attach` variables in the computational graph (DAG).

## The Standard: `GaussianObservationModel`

Leaspy primarily uses a Gaussian observation model. It assumes that the observed data is simply the model's prediction plus random noise:

$$ y_{observed} = y_{model}(t) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2) $$

> **Note**: The word "Logistic" in `LogisticModel` refers to the *shape of the mean prediction* $y_{model}(t)$ (a sigmoid curve). The noise on top of that prediction is always Gaussian. These are two independent choices.

### Handling Noise (`FullGaussianObservationModel`)

The estimation of the noise level $\sigma$ is integrated directly into the model fitting (in the M-step). We support two different noise structures, regardless of how many features the model has:

*   **Scalar Noise (Homoscedastic)**: The model estimates a **single global $\sigma$** shared by all features. This constrains the model to assume that every biomarker has the same noise level.
*   **Diagonal Noise (Heteroscedastic)**: The model estimates a **distinct $\sigma_k$ for each feature**. This is crucial for multivariate models where some sources of data might be much noisier than others.

**Why does this matter?**
The noise level acts as a natural "weighting" mechanism. The actual NLL that the algorithm minimizes per observation is (from `NormalFamily._nll`):

$$
\text{NLL}_k = \frac{1}{2}\left(\frac{y_k - \mu_k}{\sigma_k}\right)^2 + \ln(\sigma_k) + \frac{1}{2}\ln(2\pi)
$$

Notice the **two competing terms**:
*   $\frac{1}{2}\left(\frac{y_k - \mu_k}{\sigma_k}\right)^2$ — pushes $\sigma_k$ **down** to penalize large residuals.
*   $\ln(\sigma_k)$ — pushes $\sigma_k$ **up**, preventing it from trivially going to $0$ or being used to ignore any feature.

This balance is what allows $\sigma_k$ to be meaningfully estimated during fitting. If the model learns that Feature A is very noisy (large $\sigma_A$) and Feature B is clean (small $\sigma_B$), it will prioritize fitting Feature B accurately, while being more forgiving of deviations in Feature A.