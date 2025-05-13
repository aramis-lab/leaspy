# Models Evaluation

## Convergence diagnosis

*We need to talk with Sofia in June to complete this part.*

*At first glance, we would talk about looking at the convergence graphs saved in the logs (all the plots in convergence.pdf). WDYT?* 


## Likelihood based metrics

*Sofia and Gabrielle*

## Fit metrics

In leaspy, 3 negative log likelihood (nll) are stored in the model’s json file, in the `fit_metrics` part:
- `nll_attach`: Corresponds to the nll attach to the data
- `nll_regul_ind_sum`: Corresponds to the nll from the random effects
- `nll_to`: Corresponds to the total nll: nll_attach, nll_regul_ind_sum and the nll linked to the individual parameters (v0, xi, tau), that is not reported directly in the json file.

It is the last nll which is used for computing fit metric as BIC, AIC, DIC…


### Bayesian Approach

#### WAIC
WAIC (Watanabe – Akaike information criterion), defined by Watanabe in 2010 \cite{watanabe2010asymptotic}, is a metric permitting comparing different Bayesian models. Lower values of WAIC correspond to better performance of the model.  
For example, the version by Vehtari, Gelman, and Gabry (2017) \cite{vehtari2017practical} can be used in the Leaspy framework, with β = 1 and multiplied by -2 to be on the deviance scale. To compute it, the probability of observation given the parameters should be computed for each iteration. Two versions can be computed using the conditional or the marginal likelihood.  
The marginal likelihood is more robust \cite{millar2018bayesian}, but it is harder to compute as the integral must be estimated. It is usually estimated using Laplace’s approximation, which corresponds to a Taylor expansion \cite{daxberger2021laplace}.  

$$
\text{WAIC} = -2 \cdot \sum_{i=1}^{n} \log(p(y_i | \hat{\theta}_i))
$$

Where \( p(y_i | \hat{\theta}_i) \) is the probability of the observation given the estimated parameters \(\hat{\theta}_i\), and \(n\) is the number of data points.

**Package:** `leaspy`

### Frequentist Approach

#### AIC
AIC (Akaike Information Criterion) is a robust metric for model selection. It integrates the goodness-of-fit and the complexity of the model (number of features and number of patients). It has a penalty term for the number of parameters in the model, thus penalizing more complex models with unnecessary features. Lower AIC values indicate a better model.\cite{akaike1974new}

$$
\text{AIC} = 2 \cdot (\text{nb\_features}) - 2 \cdot \log(\text{likelihood})
$$

**Package:** `statsmodels.api`

### BIC
BIC (Bayesian Information Criterion) is similar to the AIC metric, but it also integrates the number of patients. It penalizes both the number of features and the number of patients.\cite{schwarz1978estimating}

$$
\text{BIC} = \log(\text{nb\_patient}) \cdot \text{features} - 2 \cdot \log(\text{likelihood})
$$

**Package:** `statsmodels.api`


## Prediction metrics

These metrics could be computed either on predictions or in reconstructions. Note that all this could be stratified by the different groups you have in your dataset. You might want these metrics to be independent from it.

### Repeated Measures

This section describes the main metrics used to evaluate the quality of predictions from the Leaspy non-linear mixed-effects model.

#### Mean Absolute Error (MAE)

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

MAE is more robust to outliers and provides a straightforward sense of typical prediction error.

#### Mean Square Error (MSE)

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

MSE penalizes larger errors more heavily than MAE, making it more sensitive to outliers. \cite{willmott2005advantages}\cite{chai2014root}

**Package:** `sklearn.metrics`

#### Residual Q-Q Plot
A graphical tool to assess whether residuals follow a normal distribution, which is an assumption in many mixed-effects models.\cite{fox2015applied}

**Package:** `statsmodels.api`

#### Coefficient of Determination (R²)
R² indicates how well the model explains the variance in the observed data. Higher values (closer to 1) suggest better performance.

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

In mixed-effects models, multiple R² variants exist (e.g., marginal vs. conditional R²) to account for fixed and random effects. \cite{nakagawa2012method}

**Package:** `sklearn.metrics`

### Events

For the joint model, it is interesting to have prediction metrics on survival as well:

#### Integrated Brier Score (IBS)
The Integrated Brier Score (IBS) is a robust metric used to evaluate the predictive accuracy of survival models. In survival analysis, IBS quantifies how well a model predicts the timing of events by integrating the Brier Score, a measure of prediction error for probabilistic outcomes, across all observed time points. It compares the model’s predicted survival probabilities against actual observed outcomes, penalizing discrepancies between predictions and reality. IBS accounts for censored data, ensuring reliability in real-world datasets where not all events are fully observed. IBS describes how closely predictions match reality. A low IBS indicates superior predictive performance, as it reflects smaller cumulative errors over time.\cite{graf1999assessment}

**Package:** `scikit-survival`

#### Cumulative Dynamic AUC
Cumulative AUC (or time-dependent AUC) evaluates the model’s ability to discriminate between subjects who experience an event by a specific time and those who do not, based on their predicted risk scores. It focuses on ranking accuracy, ensuring high-risk subjects receive higher predicted probabilities than low-risk ones at each time point. Cumulative AUC describes how well risks are ordered. A high cumulative AUC means a strong discriminatory ability.  \cite{uno2007evaluating} \cite{hung2010estimation} \cite{lambert2014summary}

**Package:** `scikit-survival`

#### Avoid using C-index
The C-index or Concordance index, similarly to the cumulative AUC, is a metric assessing the discriminatory ability of a survival model. However, this metric is criticized because it is a global metric that averages performance over the entire study period, hiding time-specific weaknesses. It also depends on the censoring distribution. Therefore, it is more convenient to use time-dependent AUC and the Brier Score presented above. \cite{blanche2019cindex}

## References

```{bibliography}
:filter: docname in docnames
```