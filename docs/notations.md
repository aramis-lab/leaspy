# Notations

This page contains the definitions and notations of the variables used in Leaspy.

## Notations, concepts, and names in DAG

The following table displays the relationships between a variable mathematical notation, the associated concept, and the string name used in the DAG if this variable is handled this way.

| Notation   | Concept                                                | Name in the DAG |
|------------|--------------------------------------------------------|-----------------|
| $ \tau_i $ | [estimated reference time](#estimated-reference-time)  | `tau`            |
|  $ \xi_i $ | [individual log speed factor](#individual-log-speed-factor) | `xi`             |
| $ \gamma_i(t) $ | [individual trajectory](#individual-trajectory) | `model`             |
| $ \psi_i(t) $ | [latent disease age](#latent-disease-age) | `rt`             |
| $ \mathbf{A} $ | [mixing matrix](#mixing-matrix) | `mixing_matrix` |
| $ t_0 $ | [population reference time](#population-reference-time) | `tau_mean`             |
| $ \mathbf{s}_i$ | [sources](#sources) | `sources` |
| $ w_{i,k} $ | [space shift](#space-shift) | `space_shifts`  |
| $ u_{i,l} $ | [survival shift](#survival-shift) | `survival_shifts`             |

## Concept definitions

This section explains each mathematical concept and provides mathematical definitions when needed.

(estimated-reference-time)=
### Estimated reference time

The *estimated reference time* for a given individual $ i $ is denoted as $ \tau_i $. It follows $\tau_i \sim \mathcal{N}(t_0, \sigma^2_{\tau})$.

(individual-log-speed-factor)=
### Individual log speed factor

The *individual log speed factor* for a given individual $ i $ is denoted as $ \xi_i $ . It follows $\xi_i \sim \mathcal{N}(0, \sigma^2_{\xi})$.

(individual-trajectory)=
### Individual trajectory

The *individual trajectory* for a given individual $ i $ is denoted as $ \gamma_i(t) $ and represents the disease progression of the patient $i$. It can be indexed by $k$ when $K$ outcomes are estimated.

(latent-disease-age)=
### Latent disease age

The *latent disease age*, for a given individual $ i $, is denoted as $ \psi_i(t) $.

It represents a transformation from chronological time $t$ to latent disease age that is related to its stage in the disease and is defined as:

$$
\psi_i(t) = e^{\xi_i}(t - \tau_i) + t_0
$$

where :

- $ e^{\xi_i} $ is the [individual speed factor](#individual-log-speed-factor) of individual $ i $.
- $ \tau_i $ is the [estimated reference time](#estimated-reference-time) of individual $ i $.
- $ t_0 $ is the [population reference time](#population-reference-time).

(mixing-matrix)=
### Mixing matrix

The *mixing matrix* is denoted as $ \mathbf{A} $ and is defined as a matrix that describes the mixing of different sources in the model.

(population-reference-time)=
### Population reference time

The *population reference time* is denoted as $ t_0 $ and is defined as *the reference time for the entire population*.

(sources)=
### Sources

The *sources*, for a given individual $ i $, are denoted as $ \mathbf{s}_i$ and defined as the different origins of information or data that contribute to the individual's disease progression model.

(space-shift)=
### Space shift

The *space shift*, are more interpretable than sources $\mathbf{s}_i$, as they encapsulate total spatial variability effects.
for a given individual $ i $ and a given longitudinal outcome $ k $, is denoted as $ w_{i,k} $ and defined as:

$$
w_{i,k} = \sum_{j=1}^{N_w} \eta_{k,j} s_{i,j}
$$

where:

- $ N_w $ is the number of spatial sources.
- $ \eta_{k,j} $ is the weight for spatial source $ j $ in outcome $ k $.
- $ s_{i,j} $ is the contribution of spatial source $ j $ for individual $ i $.

(survival-shift)=
### Survival shift

The *survival shift*, for a given individual $ i $ and a given event $ l $ is denoted as $ u_{i,l} $ and defined as:

$$
u_{i,l} = \sum_{m=1}^{N_s} \zeta_{l,m} s_{i,m}
$$

where:

- $ N_s $ is the number of survival sources.
- $ \zeta_{l,m} $ is the weight for survival source $ m $ in event $ l $.
- $ s_{i,m} $ is the contribution of survival source $ m $ for individual $ i $.

