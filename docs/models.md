# Models

## Logistic Model
### Data
   - *"When you have this type of data, it is relevant to use this model"*
### Mathematical background
### References

## Joint Model

### Definition

Joint models are a class of statistical models that simultaneously analyze longitudinal data (repeated measurements over time) and survival data (time-to-event outcomes). Unlike traditional approaches that treat these processes separately, joint models integrate them into a unified framework, recognizing that they often share underlying biological mechanisms—for example, a slowly progressing biomarker may signal an increased risk of a clinical event. By linking the two submodels—typically through shared random effects or latent processes, models account for their interdependence, reducing biases from informative dropout or measurement error.

### Data
A joint model is relevant when you have:
- **Longitudinal measurements** as repeated biomarker readings, clinical scores
  
- **Time-to-event outcomes** as survival, dropout, or failure events
- **A suspected association** between the longitudinal process and event risk
  
### Mathematical background

#### Temporal Random Effects
Individual variability for patient $i$ is modeled vwith the latent disease age $\psi_i(t)$:  

$$
\psi_i(t) = e^{\xi_i}(t - \tau_i) + t_0
$$

where:
- $ \e^{\xi_i} $ : Individual speed factor
- $ \tau_i $ : Estimated reference time
- $ t_0 $ : Population reference time

The longitudinal $ \gamma_i(t)$ and survival $S_i(t)$ processes are derived from $ \psi_i(t) $.  

*Key Hypothesis*: Longitudinal and survival processes are linked by a shared latent disease age.

#### Spatial Random Effects
Disease presentation variability is captured by $\mathbf{w}_i = \mathbf{A} \mathbf{s}_i$ where:  
- $\mathbf{A}$: Mixing matrix (dimension reduction with $N_s \leq K-1 $ independent sources)  
- $ \mathbf{s}_i$: Independent sources  

For identifiability, $ \mathbf{A} $ is defined as a linear combination of an orthonormal basis $ (\mathbf{B}_k)_{1 \leq k \leq K} $ orthogonal to $ \text{Span}(\mathbf{v}_0) $ , so that:

$$
\mathbf{A} = (\mathbf{B}\beta)^T
$$

Each event ( $l$ ) has a survival shift:

$$
u_{i,l} = \sum_{m=1}^{N_s} \zeta_{l,m} s_{i,m}
$$

*Interpretation*: Space shifts ( $w_{i,k}$) and survival shifts ( $u_{i,l} $) are more interpretable than sources ( $s_i $), as they encapsulate total spatial variability effects.

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

Hazard for event $ l $:  

$$
h_{i,l}(t) = h_{0,i,l}(t) \exp(u_{i,l}) \
=\rho_l e^{\xi_i} \left( \frac{e^{\xi_i (t - \tau_i)}}{\nu_l} \right)^{\rho_l - 1} \exp(u_{i,l})
$$

Survival function: 

$$
S_{i,l}(t) = \exp - \int_0^t h_{i,l}(x)dx \
= \exp\left( -\left( \frac{e^{\xi_i (t - \tau_i)}}{\nu_l} \right)^{\rho_l} \exp(u_{i,l}) \right)
$$

Cumulative Incidence Function (CIF):

$$
\text{CIF}_{i,l}(t) = \int_0^t h_{i,l}(x) \prod_{q=1}^L S_{i,q}(x) \, dx \
= \int_0^t \rho_l e^{\xi_i} \left( \frac{e^{\xi_i (x - \tau_i)}}{\nu_l} \right)^{\rho_l - 1} \exp(u_{i,l}) S_{i,l}(x) \prod_{q=1}^L S_{i,q}(x) \, dx
$$

*Interpretation*: Survival shifts $(u_{i,l})$ act proportionally on hazards (similar to Hazard Ratios).  

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


### References


## Mixture Model
## Covariate Model

