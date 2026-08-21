# Models

(introduction-to-spatio-temporal-models)=

## Introduction to Spatio-Temporal Models

(temporal-random-effects)=

### Temporal Random Effects

Individual temporal variability for patient $i$ is modeled with the [latent disease age](latent-disease-age) $\psi_i(t)$ :

$$
\psi_i(t) = e^{\xi_i}(t - \tau_i) + t_0
$$

where:

- $ \xi_i $ is the [individual log speed factor](individual-log-speed-factor)
- $ \tau_i $ is the [estimated reference time](estimated-reference-time)
- $ t_0 $ is the [population reference time](population-reference-time)

The longitudinal $ \gamma_i(t)$ and survival $S_i(t)$ processes are derived from $ \psi_i(t) $.

*Key Hypothesis*: Longitudinal and survival processes are linked by a shared latent disease age.

### Spatial Random Effects

Disease presentation variability is captured by [space-shifts](space-shift) : $\mathbf{w}_i = \mathbf{A} \mathbf{s}_i$ where:

- $\mathbf{A}$: [mixing matrix](mixing-matrix)  (dimension reduction with $N_s \leq K-1 $ independent sources:  $N_s$ being the number of sources and $K$ the number of outcomes).
- $ \mathbf{s}_i$: Independent sources

For identifiability, $ \mathbf{A} $ is defined as a linear combination of an orthonormal basis $ (\mathbf{B}_k)_{1 \leq k \leq K} $ orthogonal to $ \text{Span}(\mathbf{v}_0) $ {cite}`schirattiBayesianMixedEffectsModel`, so that:

$$
\mathbf{A} = (\mathbf{B}\beta)^T
$$

with $\beta$ the matrix of coefficients.

Each event ( $l$ ) has a [survival shift](survival-shift):

$$
u_{i,l} = \sum_{m=1}^{N_s} \zeta_{l,m} s_{i,m}
$$

*Interpretation*: Space shifts ($w_{i,k}$)  are more interpretable than sources ($s_i $), as they encapsulate total spatial variability effects.

(logistic-model)=

## Logistic Model

### Definition

The **logistic model** is a statistical model used to describe **sigmoidal trajectories** of longitudinal outcomes over time, typically biomarkers or clinical scores. This model is particularly useful when outcomes exhibit a **monotonic progression** from one asymptote to another, also called as bounded outcome scores, which is common in disease progression {cite}`lesaffreLogisticTransformBounded2007b`.
The logistic model is governed by parameters, which control its **speed**, **inflection point**, and **asymmetry**.

In Leaspy, the logistic model is implemented using a **non-linear mixed-effects** framework, where the disease evolution is described through a latent time variable (latent disease age) shared across outcomes, and individual variations are modeled through subject-specific parameters {cite}`durrleman2013toward`.

This model provides interpretable components:

- A population-level trajectory defined by a logistic function,
- Individual deviations from this curve through time reparametrization and spatial shifts,
- The ability to simulate, estimate, and personalize trajectories for unseen individuals.

(logistic-data)=

### Data

A logistic model is relevant when you have:

- **Longitudinal repeated measurements** of one or more continuous outcomes (e.g., cognitive scores, clinical scores, biomarkers)
- **Monotonic or sigmoidal progression** of outcomes over time (typically increasing or decreasing to a plateau)
- **No event or survival information required**, unlike joint models

To fit a logistic model, you need a dataframe with the following columns:

- `ID`: Patient identifier
- `TIME`: Time of measurement
- One or more columns representing the longitudinal outcomes (e.g., `OUTCOME_1`, `OUTCOME_2`, ...)

For the importation of dataframe:

```python
dataset = dataframe.set_index(["ID", "TIME"]).sort_index()
print(dataset.head())

                        OUTCOME_1  OUTCOME_2
      ID    TIME
132-S2-0  81.661          0.44444    0.04000
          82.136          0.60000    0.56000
          82.682          0.39267    0.04000
          83.139          0.58511    0.30000
          83.691          0.57044    0.05040

data_logistic = Data.from_dataframe(dataset, "visit")
```

### Mathematical background

The logistic trajectory for outcome $k$, and from the latent disease age $\psi_i(t)$ is defined by:

$$
\gamma_{i,k}(t) = \left[ 1 + g_k \times \exp\left( -\frac{(1+g_k)^2}{g_k} \left( v_{0,k}(\psi_i(t) - t_0) + w_{i,k} \right) \right) \right]^{-1}
$$

where:

- $\gamma_{i,k}(t)$ is the modeled outcome value,
- $t_0$ is the population reference time
- $v_{0,k}$ is the speed of progression for outcome $k$ at reference time $t_0$,
- $w_{i,k}$ is the individual space shift for outcome $k$.
- $\frac{1}{1+g_k}$ is the value of the logistic curve at $t_0$
- $\psi_i(t)$ is the latent disease age, please have a look to part [introduction to spatio-temporal models](introduction-to-spatio-temporal-models) for more details.

(joint-model)=

## Joint Model

### Definition

Joint models are a class of statistical models that simultaneously analyze {term}`longitudinal data` and {term}`survival data` {cite}`alsefri_bayesian_2020, ibrahim_basic_2010`. Unlike traditional approaches that treat these processes separately, joint models integrate them into a unified framework, recognizing that they often share underlying biological mechanisms—for example, a slowly progressing biomarker may signal an increased risk of a clinical event. By linking the two submodels—typically through shared random effects {cite}`rizopoulos_bayesian_2011` or latent processes {cite}`proust-lima_joint_2014`, fraitly {cite}`rondeau_frailtypack_2012`, models account for their interdependence, reducing biases from informative dropout or measurement error {cite}`tsiatisJOINTMODELINGLONGITUDINAL`.

In Leaspy, the joint model {cite}`ortholand_joint_2024` is implemented as a longitudinal spatio-temporal model, and a survival model, that are linked through a shared latent disease age, and, in the case of multiple longitudinal outcomes, spatial random effects ([see description in first paragraph of this page](introduction-to-spatio-temporal-models)). This approach allows for the incorporation of both temporal and spatial random effects, providing a more comprehensive understanding of the underlying disease process.

(joint-data)=

### Data

A joint model is relevant when you have:

- **Longitudinal measurements** as repeated biomarker readings, clinical scores
- **Time-to-event outcomes** as survival, dropout, or failure events
- **A suspected association** between the longitudinal process and event risk

You must have one dataframe with the following columns:

- `ID`: Patient identifier
- `TIME`: Time of measurement
- `EVENT_TIME`: Time of event
- `EVENT_BOOL`: Event indicator (1 if event occurred, 0 if censored and 2 if competing event)

For one patient, the event time and event bool are the same for each row.

For the importation of dataframe:

```python
dataset = dataframe.set_index(["ID", "TIME"]).sort_index()
print(dataset.head())

                                OUTCOME_1  OUTCOME_2  EVENT_TIME  EVENT_BOOL
        ID              TIME                                                     
132-S2-0              81.661      0.44444    0.04000        84.0           1
           82.13600000000001      0.60000    0.00000        84.0           1
                      82.682      0.39267    0.04000        84.0           1
                      83.139      0.58511    0.00000        84.0           1
                      83.691      0.57044    0.00000        84.0           1
184-S2-0              80.994      0.32600    0.00000        83.4           0
           81.47199999999998      0.35556    0.00000        83.4           0
           82.08800000000001      0.40000    0.00000        83.4           0
                      82.488      0.31111    0.00000        83.4           0

data_joint = Data.from_dataframe(dataset, "joint")
```

### Mathematical background

#### Longitudinal Submodel

The longitudinal submodel that can be used here is the logistic model, please have a look to part [description logistic model](#logistic-model) for more details.

#### Survival Submodel

**Cause-Specific Weibull Hazards** (for competing risks):

This submodel captures how variations in the progression of longitudinal disease outcomes influence the probability and timing of multiple clinical events, while accounting for censoring and competing risks. To achieve this, the leaspy joint model uses a cause-specific hazard structure.

For each event $l$ and patient $i$, we model a cause-specific hazard $h_{i,l}(t)$ {cite}`prentice_regression_1978, cheng_prediction_1998`. This framework allows us to estimate the risk of each event separately and account for the presence of competing risks (where one event precludes others).

A Weibull distribution is used to model time-to-event data due to its flexibility in representing:

- Increasing, decreasing, or constant hazard shapes,
- Dependence on two interpretable parameters:
  - Scale parameter $\nu_l$,
  - Shape parameter $\rho_l$.

The hazard is modulated via a Cox-proportional hazard framework to incorporate the effect of longitudinal biomarkers using survival shifts.

Let $u_{i} = \zeta s_{i}$, $\zeta$ being the survival shift and $s$ sources. The cause-specific hazard $h_{i,l}(t)$ for event $l$ and subject $i$ is defined as:

$$
h_{i,l}(t) = h_{0,i,l}(t) \cdot \exp(u_{i,l})
$$

Expanding the baseline hazard $h_{0,i,l}(t)$, we get:

$$
h_{i,l}(t) = \frac{\rho_l e^{\xi_i}}{\nu_l} \cdot \left( \frac{e^{\xi_i} (t - \tau_i)}{\nu_l} \right)^{\rho_l - 1} \cdot \exp(u_{i,l})
$$

**Where**:

- $\nu_l$: Scale parameter of the Weibull distribution for event $l$,
- $\rho_l$: Shape parameter of the Weibull distribution for event $l$,
- $\xi_i$: Subject-specific parameter,
- $\tau_i$:  Estimated reference time for individual $i$,

Survival function $S_{i,l}(t)$ is then defined as:

$$
S_{i,l}(t) = \exp\left( -\left( \frac{e^{\xi_i (t - \tau_i)}}{\nu_l} \right)^{\rho_l} \exp(u_{i,l}) \right)
$$

Finally, {term}`CIF` for event $l$ and subject $i$ is defined as:

$$
\text{CIF}_{i,l}(t) = \int_0^t h_{i,l}(x) \prod_{q=1}^L S_{i,q}(x) \, dx \
= \int_0^t \rho_l e^{\xi_i} \left( \frac{e^{\xi_i (x - \tau_i)}}{\nu_l} \right)^{\rho_l - 1} \exp(u_{i,l}) S_{i,l}(x) \prod_{q=1}^L S_{i,q}(x) \, dx
$$

(joint-model-summary)=

#### Model summary

For patient $i$, outcome $k$, and event $l$:

$$
\begin{cases}
\psi_i(t) = e^{\xi_i}(t - \tau_i) + t_0 \\
\mathbf{w}_i = \mathbf{A} \mathbf{s}_i \\
\mathbf{u}_i = \zeta \mathbf{s}_i \\
\gamma_{i,k}(t) = \left[ 1 + g_k \exp\left( -\frac{v_{0,k}(1+g_k)^2}{g_k} e^{\xi_i}(t - \tau_i) + w_{i,k} \right) \right]^{-1} \\
S_{i,l}(t) = \exp\left( -\left( \frac{e^{\xi_i (t - \tau_i)}}{\nu_l} \right)^{\rho_l} \exp(u_{i,l}) \right)
\end{cases}
$$

In practice, create a `JointModel` to fit, personalize, and estimate joint trajectories. To generate synthetic longitudinal and time-to-event data, use the dedicated [`joint_simulate` algorithm](joint-simulation), which also handles censoring and competing events.

```python
leaspy_joint = JointModel(nb_events=2, source_dimension=3)
leaspy_joint.fit(data_joint, nb_iter=1000, nb_burnin=500)
```

For estimation, it is the {term}`CIF` that is outputted by the model. Note that for prediction purposes, the {term}`CIF` is corrected using the survival probability at the time of the last observed visit, following common practice in other packages {cite}`andrinopoulou_combined_2017`.

(mixture-model)=

## Mixture Model

### Definition

Mixture models are a class of statistical models that represent a population as a combination of underlying subpopulations, each described by its own probability distribution {cite}`mclachlan_finite_2000`. Standard mixed effects models assume a certain homogeneity, in the sense that inter-patient variability is considered as random variation around a fixed reference. Mixture models explicitly recognize that observed data may arise from distinct latent groups—for example, patients may cluster into different onset times, rates of disease progression or more advanced clinical scores. This formulation captures heterogeneity in the data, allowing for flexible modeling of complex, multimodal patterns. Estimation is typically carried out using an adaptation of {term}`MCMC-SAEM` for mixture models as described in {cite}`mclachlan_finite_2000`, which iteratively refine both the component parameters and the posterior probabilities of membership. By providing a probabilistic clustering framework, mixture models not only identify hidden structure but also quantify uncertainty in subgroup assignment, making them widely applicable in biomedical sciences.

In Leaspy the mixture model is implemented as an adaptation of the spatio-temporal logistic model where the individual parameters (`tau`, `xi`and `sources`) come from a mixture of gaussian distributions with a number of components defined by the user.

(mixture-data)=

### Data

The same rules apply as for the standard [logistic model](#logistic-model).

### Mathematical background

To build a mixture model, we add another layer upon the hierarchical structure of the model. We suppose that each subgroup $c$ can be described by a different set of $(\overline{\tau}^c, \overline{\xi}^c, \overline{s}^c)$ and has a respective probability of occurrence $\pi^c$. We suppose that the population parameters come from normal distributions as in the standard model, while the individual parameters are sampled from a mixture of gaussians with mixing probabilities $\pi^c$.

The individual parameters (random effects):

$$
\begin{cases}
\xi_i \sim \sum_{c=1}^{n_c} \pi^c \mathcal{N} (\overline\xi^c, \sigma_{\xi}^c) \\
\tau_i \sim \sum_{c=1}^{n_c} \pi^c \mathcal{N} (\overline\tau^c, \sigma_{\tau}^c)\\
s_{il} \sim \sum_{c=1}^{n_c} \pi^c \mathcal{N} (\overline s^c, 1)
\end{cases}
$$

(mixture-model-summary)=

### Model summary

To use the mixture model in Leaspy you need to choose the number of cluster you wish to estimate beforehand.

```python
from leaspy.models import LogisticMultivariateMixtureModel

leaspy_mixture = LogisticMultivariateMixtureModel(source_dimension=1, n_clusters=2, dimension=3)
leaspy_mixture.fit(data_logistic,  "mcmc_saem", n_iter=1000)
```

(covariate-model)=

## Covariate Model

### Definition

The **covariate model** extends the standard [logistic model](#logistic-model) to take patient-level covariates into account (e.g. gender, genotype, age at inclusion, socio-economic status). In the standard logistic model, as in classical mixed-effects models more generally, inter-patient variability is modeled as a random perturbation around a fixed population reference, even though some of that variability is known to stem from identifiable covariates. The covariate model instead estimates covariate effects **jointly** with the rest of the model, directly on the population parameters that govern the trajectory: the population reference time $t_0$, and the feature-specific position and velocity parameters $g_k$ and $v_{0,k}$.

Rather than assuming every covariate has an effect, the model performs **variable selection**: for each population parameter and each covariate, a binary latent mask decides whether that covariate has an effect at all, and, if so, an effect size is estimated.

In Leaspy, the covariate model is implemented as an extension of the spatio-temporal logistic model (in its original [Schiratti et al.] parametrization {cite}`schirattiBayesianMixedEffectsModel`, with an explicit population-level $t_0$), where $t_0$, $g_k$, and $v_{0,k}$ each become patient-specific through a linear covariate effect.

(covariate-data)=

### Data

A covariate model is relevant when you have:

- **Longitudinal repeated measurements** of one or more continuous outcomes, as for the [logistic model](#logistic-data)
- **One or more patient-level covariates**, constant over time (e.g. genotype, baseline score, age at inclusion, sex)
- **A hypothesis that some of these covariates affect the disease trajectory** (its onset, its speed, or the feature values), which you want to test and quantify jointly with the rest of the model, rather than post-hoc

To fit a covariate model, you need a dataframe with the following columns:

- `ID`: Patient identifier
- `TIME`: Time of measurement
- One or more columns representing the longitudinal outcomes (e.g., `OUTCOME_1`, `OUTCOME_2`, ...)
- One or more columns representing the covariates (e.g., `COVARIATE_1`, `COVARIATE_2`, ...), constant across all visits of a given patient

Covariates can be binary or continuous. Non-binary covariates should be standardized (mean $\approx 0$, standard deviation $\approx 1$) beforehand, since the model's priors and MCMC proposal steps are calibrated for that scale.

For the importation of dataframe:

```python
dataset = dataframe.set_index(["ID", "TIME"]).sort_index()
print(dataset.head())

                        OUTCOME_1  OUTCOME_2  COVARIATE_1  COVARIATE_2
      ID    TIME
132-S2-0  81.661          0.44444    0.04000            1        -0.32
          82.136          0.60000    0.56000            1        -0.32
          82.682          0.39267    0.04000            1        -0.32
          83.139          0.58511    0.30000            1        -0.32
          83.691          0.57044    0.05040            1        -0.32

data_covariate = Data.from_dataframe(
    dataset,
    "covariate",
    factory_kws={"covariate_names": ["COVARIATE_1", "COVARIATE_2"]},
)
```

### Mathematical background

Let $\mathbf{c}_i \in \mathbb{R}^{N_c}$ denote the vector of $N_c$ covariates for patient $i$. For each population parameter $x \in \{t_0, g_k, v_{0,k}\}$, the model introduces:

- an effect vector $\delta_x \in \mathbb{R}^{N_c}$, quantifying the size of each covariate's effect,
- a binary selection mask $\gamma_x \in \{0,1\}^{N_c}$, deciding, covariate by covariate, whether that effect is active.

The population parameters become patient-specific through the masked, linear effect $(\gamma_x \odot \delta_x)^\top \mathbf{c}_i$, where $\odot$ denotes the element-wise (Hadamard) product:

$$
\begin{cases}
t_0(\mathbf{c}_i) = t_0 + (\gamma_{t_0} \odot \delta_{t_0})^\top \mathbf{c}_i \\
g_k(\mathbf{c}_i) = \exp\left(\tilde{g}_k + (\gamma_{g_k} \odot \delta_{g})^\top \mathbf{c}_i\right) \\
v_{0,k}(\mathbf{c}_i) = \exp\left(\tilde{v}_k + (\gamma_{v_k} \odot \delta_{v})^\top \mathbf{c}_i\right)
\end{cases}
$$

where $\tilde g_k = \log(g_k)$ and $\tilde v_k = \log(v_{0,k})$, as in the standard logistic model. When $\gamma_{x,c}=0$, the $c$-th covariate has no effect on parameter $x$; when $\gamma_{x,c}=1$, the full effect $\delta_{x,c}$ is applied.

> **Note — back to the Schiratti parametrization.** Because the covariate effect on $t_0$ needs an explicit population-level reference time to act on, the covariate model reintroduces the original [Schiratti et al.](#logistic-model) parametrization, where $t_0$ and $\tau_i$ are kept separate. This differs from the standard Leaspy logistic model, which simplifies this parametrization by absorbing $t_0$ into the mean of $\tau_i$'s prior. Concretely, the latent disease age is here:
>
> $$
> \psi_i(t) = e^{\xi_i}\left(t - t_0(\mathbf{c}_i) - \tau_i\right) + t_0(\mathbf{c}_i)
> $$
>
> instead of $\psi_i(t) = e^{\xi_i}(t - \tau_i)$ in the standard logistic model.

The resulting trajectory for outcome $k$ and patient $i$ follows the same logistic form as the [standard logistic model](#logistic-model), but evaluated at the patient-specific parameters:

$$
\gamma_{i,k}(t) = \left[ 1 + g_k(\mathbf{c}_i) \times \exp\left( -\frac{(1+g_k(\mathbf{c}_i))^2}{g_k(\mathbf{c}_i)} \left( v_{0,k}(\mathbf{c}_i)\left(\psi_i(t) - t_0(\mathbf{c}_i)\right) + w_{i,k} \right) \right) \right]^{-1}
$$

**Priors.** Population-level parameters keep their usual Gaussian priors. The mask and the covariate effect follow:

$$
\gamma_{x,c} \sim \text{Bernoulli}(\pi), \qquad \delta_x \sim \mathcal{N}\left(\gamma_x \odot \bar\delta_x, \, \Sigma_{\delta_x}\right)
$$

where $\pi$ and $\Sigma_{\delta_x}$ are fixed hyperparameters, and $\bar\delta_x$ is the corresponding population-level model parameter.

(covariate-model-summary)=

### Model summary

For patient $i$ and outcome $k$, with covariate vector $\mathbf{c}_i$:

$$
\begin{cases}
t_0(\mathbf{c}_i) = t_0 + (\gamma_{t_0} \odot \delta_{t_0})^\top \mathbf{c}_i \\
g_k(\mathbf{c}_i) = \exp\left(\tilde{g}_k + (\gamma_{g_k} \odot \delta_{g})^\top \mathbf{c}_i\right) \\
v_{0,k}(\mathbf{c}_i) = \exp\left(\tilde{v}_k + (\gamma_{v_k} \odot \delta_{v})^\top \mathbf{c}_i\right) \\
\psi_i(t) = e^{\xi_i}\left(t - t_0(\mathbf{c}_i) - \tau_i\right) + t_0(\mathbf{c}_i) \\
\mathbf{w}_i = \mathbf{A} \mathbf{s}_i \\
\gamma_{i,k}(t) = \left[ 1 + g_k(\mathbf{c}_i) \exp\left( -\dfrac{(1+g_k(\mathbf{c}_i))^2}{g_k(\mathbf{c}_i)} \left( v_{0,k}(\mathbf{c}_i)(\psi_i(t) - t_0(\mathbf{c}_i)) + w_{i,k} \right) \right) \right]^{-1}
\end{cases}
$$

In practice, once your data is loaded with `Data.from_dataframe(dataset, "covariate", factory_kws={"covariate_names": [...]})` as shown in the [Data](#covariate-data) section above, you can fit the covariate model:

```python
from leaspy.models import CovariateLogisticModel

leaspy_covariate = CovariateLogisticModel(source_dimension=1)
leaspy_covariate.fit(data_covariate, "mcmc_saem", n_iter=100000)
```

To speed up and stabilize convergence, it is recommended to first fit a standard [logistic model](#logistic-model) (in its `Schiratti` parametrization, i.e. `LogisticModelSchiratti`) and pass it to the covariate model via `init_from_model`. This initializes the parameters shared between both models (e.g. `t0`, `g`, `v0`) from the already-fitted logistic model, so that only the covariate-specific effects need to be learned from scratch:

```python
from leaspy.models import LogisticModelSchiratti, CovariateLogisticModel

# 1. Fit a standard logistic model first
model_init = LogisticModelSchiratti(name="logistic-init", source_dimension=1)
model_init.fit(data, "mcmc_saem", n_iter=1000)

# 2. Fit the covariate model, initialized from the model above
leaspy_covariate = CovariateLogisticModel(
    source_dimension=1, init_from_model=model_init
)
leaspy_covariate.fit(data_covariate, "mcmc_saem", n_iter=100000)
```

Each estimated mask $\gamma_x$ can be inspected after fitting to determine which covariates were found to have a non-negligible effect on which population parameter (onset $t_0$; feature position $g_k$; feature speed $v_{0,k}$), and the corresponding $\delta_x$ gives the size and direction of that effect.
