# Models

## Introduction to spatio-temporal models

### Temporal Random Effects
Individual variability for patient $i$ is modeled vwith the latent disease age $\psi_i(t)$:  

$$
\psi_i(t) = e^{\xi_i}(t - \tau_i) + t_0
$$

where:
- $ e^{\xi_i} $ : Individual speed factor
- $ e^{\xi_i} $ : Individual speed factor
- $ \tau_i $ : Estimated reference time
- $ t_0 $ : Population reference time

The longitudinal $ \gamma_i(t)$ and survival $S_i(t)$ processes are derived from $ \psi_i(t) $.  

*Key Hypothesis*: Longitudinal and survival processes are linked by a shared latent disease age.

### Spatial Random Effects
Disease presentation variability is captured by $\mathbf{w}_i = \mathbf{A} \mathbf{s}_i$ where:  
- $\mathbf{A}$: Mixing matrix (dimension reduction with $N_s \leq K-1 $ independent sources)  
- $ \mathbf{s}_i$: Independent sources  

For identifiability, $ \mathbf{A} $ is defined as a linear combination of an orthonormal basis $ (\mathbf{B}_k)_{1 \leq k \leq K} $ orthogonal to $ \text{Span}(\mathbf{v}_0) $ , so that:

$$
\mathbf{A} = (\mathbf{B}\beta)^T
$$

with $\beta$ the matrix of coefficients.

Each event ( $l$ ) has a survival shift:

$$
u_{i,l} = \sum_{m=1}^{N_s} \zeta_{l,m} s_{i,m}
$$

*Interpretation*: Space shifts ( $w_{i,k}$)  are more interpretable than sources ( $s_i $), as they encapsulate total spatial variability effects.

## Logistic Model
### Definition
### Data
   - *"When you have this type of data, it is relevant to use this model"*
### Mathematical background

### References

## Joint Model

### Definition

Joint models are a class of statistical models that simultaneously analyze longitudinal data (repeated measurements over time) and survival data (time-to-event outcomes) {cite}`alsefri_bayesian_2020, ibrahim_basic_2010`. Unlike traditional approaches that treat these processes separately, joint models integrate them into a unified framework, recognizing that they often share underlying biological mechanisms—for example, a slowly progressing biomarker may signal an increased risk of a clinical event. By linking the two submodels—typically through shared random effects {cite}`rizopoulos_bayesian_2011` or latent processes {cite}`proust-lima_joint_2014`, or fraitly {cite}`rondeau_frailtypack_2012` models account for their interdependence, reducing biases from informative dropout or measurement error.

In Leaspy, the joint model {cite}`ortholand:tel-04770912` is implemented as a longitudinal spatio-temporal model, and a survival model, that are linked through a shared latent disease age. This approach allows for the incorporation of both temporal and spatial random effects, providing a more comprehensive understanding of the underlying disease process.


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

```{python}
dataset = dataframe.set_index(["ID", "TIME"]).sort_index()
data_joint = Data.from_dataframe(dataset, "joint")
```

  
### Mathematical background
#### Longitudinal Submodel

Logistic trajectories for outcome $k$, and from the latent disease age $\psi_i(t)$ is defined by:

$$
\gamma_{i,k}(t) = \left[ 1 + g_k x \exp\left( -\frac{(1+g_k)^2}{g_k} \left( v_{0,k}(\psi_i(t) - t_0) + w_{i,k} \right) \right) \right]^{-1}
$$

where:  
- $ t_0 $: Population reference time
- $ v_{0,k} $: Logistic curve speed at $ t_0 $
- $ \frac{1}{1+g_k} $: Logistic curve value at $ t_0 $

#### Survival Submodel
**Cause-Specific Weibull Hazards** (for competing risks):

This submodel captures how variations in the disease longitudinal outcomes progression, influence the probability and timing of multiple clinical events, while accounting for censoring and competing risks. To do so, we adopt a cause-specific structure For each event $l$ and patient $i$, we model a cause-specific hazard $ \h_{i_l}(t) $ {cite}`prentice_regression_1978, cheng_prediction_1998` because it allows us to model the risk of each event separately, while still accounting for the presence of other events. This is particularly useful in the context of competing risks, where the occurrence of one event may preclude the occurrence of another. A Weibull distribution is used to model the time-to-event data, as it is flexible and can accommodate various hazard shapes (increasing, decreasing, or constant) depending on the parameters chosen. The Weibull distribution is defined by two parameters: a scale parameter $\nu_l$ and a shape parameter $\rho_l$ and is modulated with a Cox-proportional hazard effect of the sources on the hazard using survival shifts:
$
\h_{i_l}(t)=\h_{0_i_l}(t) x e^{\u_i,l}
$

$$
\text{CIF}_{i,l}(t) = \int_0^t h_{i,l}(x) \prod_{q=1}^L S_{i,q}(x) \, dx \
= \int_0^t \rho_l e^{\xi_i} \left( \frac{e^{\xi_i (x - \tau_i)}}{\nu_l} \right)^{\rho_l - 1} \exp(u_{i,l}) S_{i,l}(x) \prod_{q=1}^L S_{i,q}(x) \, dx
$$

#### Model summary
For patient $( i )$, outcome $( k )$, and event $( l )$:

$$
\begin{cases}
\psi_i(t) = e^{\xi_i}(t - \tau_i) + t_0 \\
\mathbf{w}_i = \mathbf{A} \mathbf{s}_i \\
\mathbf{u}_i = \zeta \mathbf{s}_i \\
\gamma_{i,k}(t) = \left[ 1 + g_k \exp\left( -\frac{v_{0,k}(1+g_k)^2}{g_k} e^{\xi_i}(t - \tau_i) + w_{i,k} \right) \right]^{-1} \\
S_{i,l}(t) = \exp\left( -\left( \frac{e^{\xi_i (t - \tau_i)}}{\nu_l} \right)^{\rho_l} \exp(u_{i,l}) \right)
\end{cases}
$$

In practice in leaspy, to use joint model, you need to precise "joint" in Leaspy object creation, then you can use it to fit, personnalize, estimate and simulate.

```{python}
leaspy_joint = Leaspy("joint", nb_events=2, dimension=9, source_dimension=7)
leaspy_joint.fit(data_joint, settings=algo_settings_joint_fit)
```

### References

```{bibliography}
:filter: docname in docnames
```


## Mixture Model
## Covariate Model

