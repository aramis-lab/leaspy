# Models

## Introduction to Spatio-Temporal Models

### Temporal Random Effects
Individual variability for patient $i$ is modeled with the [latent disease age](./notations.md#latent-disease-age) $\psi_i(t)$ :  

$$
\psi_i(t) = e^{\xi_i}(t - \tau_i) + t_0
$$

where:
- $ \xi_i $ is the [individual log speed factor](./notations.md#individual-log-speed-factor)
- $ \tau_i $ is the [estimated reference time](./notations.md#estimated-reference-time)
- $ t_0 $ is the [population reference time](./notations.md#population-reference-time)


The longitudinal $ \gamma_i(t)$ and survival $S_i(t)$ processes are derived from $ \psi_i(t) $.  

*Key Hypothesis*: Longitudinal and survival processes are linked by a shared latent disease age.

### Spatial Random Effects
Disease presentation variability is captured by [space-shifts](./notations.md#space-shift) : $\mathbf{w}_i = \mathbf{A} \mathbf{s}_i$ where:  
- $\mathbf{A}$: [mixing matrix](./notations.md#mixing-matrix)  (dimension reduction with $N_s \leq K-1 $ independent sources:  $N_s$ being the number of sources and $K$ the number of [space shifts](./notations.md#space-shift)).
- $ \mathbf{s}_i$: Independent sources  

For identifiability, $ \mathbf{A} $ is defined as a linear combination of an orthonormal basis $ (\mathbf{B}_k)_{1 \leq k \leq K} $ orthogonal to $ \text{Span}(\mathbf{v}_0) $ {cite}`schirattiBayesianMixedEffectsModel`, so that:

$$
\mathbf{A} = (\mathbf{B}\beta)^T
$$

with $\beta$ the matrix of coefficients.

Each event ( $l$ ) has a [space shift](./notations.md#space-shift):

$$
u_{i,l} = \sum_{m=1}^{N_s} \zeta_{l,m} s_{i,m}
$$

*Interpretation*: Space shifts ($w_{i,k}$)  are more interpretable than sources ($s_i $), as they encapsulate total spatial variability effects.

## Logistic Model
### Definition
### Data
   - *"When you have this type of data, it is relevant to use this model"*
### Mathematical background

Logistic trajectories for outcome $k$, and from the latent disease age $\psi_i(t)$ is defined by:

$$
\gamma_{i,k}(t) = \left[ 1 + g_k x \exp\left( -\frac{(1+g_k)^2}{g_k} \left( v_{0,k}(\psi_i(t) - t_0) + w_{i,k} \right) \right) \right]^{-1}
$$

where:  
- $ t_0 $: Population reference time
- $ v_{0,k} $: Logistic curve speed at $ t_0 $
- $ \frac{1}{1+g_k} $: Logistic curve value at $ t_0 $


### References

## Joint Model

### Definition

Joint models are a class of statistical models that simultaneously analyze [longitudinal data](./glossary.md#longitudinal-data) and [survival data](./glossary.md#survival-data) {cite}`alsefri_bayesian_2020, ibrahim_basic_2010`. Unlike traditional approaches that treat these processes separately, joint models integrate them into a unified framework, recognizing that they often share underlying biological mechanisms—for example, a slowly progressing biomarker may signal an increased risk of a clinical event. By linking the two submodels—typically through shared random effects {cite}`rizopoulos_bayesian_2011` or latent processes {cite}`proust-lima_joint_2014`, fraitly {cite}`rondeau_frailtypack_2012`, models account for their interdependence, reducing biases from informative dropout or measurement error {cite}`tsiatisJOINTMODELINGLONGITUDINAL`.

In Leaspy, the joint model {cite}`ortholand_joint_2024` is implemented as a longitudinal spatio-temporal model, and a survival model, that are linked through a shared latent disease age, and, in the case of multiple longitudinal outcomes, spatial random effects ([see description in first paragraph of this page](./models.md#Introduction-to-spatio-temporal-models)). This approach allows for the incorporation of both temporal and spatial random effects, providing a more comprehensive understanding of the underlying disease process.


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

                                FEATURE_1  FEATURE_2  EVENT_TIME  EVENT_BOOL
  ID       TIME                                                           
  132-S2-0 81.661               0.44444    0.04000        84.0           1
          82.13600000000001    0.60000    0.00000        84.0           1
          82.682               0.39267    0.04000        84.0           1
          83.139               0.58511    0.00000        84.0           1
          83.691               0.57044    0.00000        84.0           1
  184-S2-0 80.994               0.32600    0.00000        83.4           0
          81.47199999999998    0.35556    0.00000        83.4           0
          82.08800000000001    0.40000    0.00000        83.4           0
          82.488               0.31111    0.00000        83.4           0

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

Finally, [CIF](./glossary.md#cif) for event $l$ and subject $i$ is defined as:

$$
\text{CIF}_{i,l}(t) = \int_0^t h_{i,l}(x) \prod_{q=1}^L S_{i,q}(x) \, dx \
= \int_0^t \rho_l e^{\xi_i} \left( \frac{e^{\xi_i (x - \tau_i)}}{\nu_l} \right)^{\rho_l - 1} \exp(u_{i,l}) S_{i,l}(x) \prod_{q=1}^L S_{i,q}(x) \, dx
$$

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

In practice in leaspy, to use the joint model, you need to precise "joint" in Leaspy object creation, then you can use it to fit, personnalize, estimate and simulate.

```python
leaspy_joint = Joint(nb_events=2, source_dimension=3)
leaspy_joint.fit(data_joint, nb_iter=1000, nb_burnin=500)
```
For estimation, it is the [CIF](./glossary.md#cif) that is outputted by the model. Note that for prediction purposes, the [CIF](./glossary.md#cif) is corrected using the survival probability at the time of the last observed visit, following common practice in other packages {cite}`andrinopoulou_combined_2017`.

### References

```{bibliography}
:filter: docname in docnames
```


## Mixture Model
## Covariate Model

