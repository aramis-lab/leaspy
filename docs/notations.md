# Notations

This page contains the definitions and notations of the variables used in Leaspy.

## Notations, concepts, and names in DAG

The following table displays the relationships between a variable mathematical notation, the associated concept, and the string name used in the DAG if this variable is handled this way.

| Notation   | Concept                                                | Name in the DAG |
|------------|--------------------------------------------------------|-----------------|
| $ \tau_i $ | [estimated reference time](#estimated-reference-time)  | `tau`            |
|  $ \xi_i $ | [individual log speed factor](#individual-speed-factor) | `xi`             |
| $ \gamma_i(t) $ | [individual trajectory](#individual-trajectory) | `model`             |
| $ \psi_i(t) $ | [latent disease age](#latent-disease-age) | `rt`             |
| $ \mathbf{A} $ | [mixing matrix](#mixing-matrix) | `mixing_matrix` |
| $ t_0 $ | [population reference time](#population-reference-time) | `tau_mean`             |
| $ \mathbf{s}_i$ | [sources](#sources) | `sources` |
| $ w_{i,k} $ | [space shift](#space-shift) | `space_shifts`  |
| $ u_{i,l} $ | [survival shift](#survival-shift) | `survival_shifts`             |

## Concept definitions

This section explains each mathematical concept and provides mathematical definitions when needed.

### Estimated reference time

The *estimated reference time* for a given individual $ i $ is denoted as $ \tau_i $. It follows $\tau_i \sim \mathcal{N}(t_0, \sigma^2_{\tau})$.

### Individual log speed factor

The *individual log speed factor* for a given individual $ i $ is denoted as $ \xi_i $ . It follows $\xi_i \sim \mathcal{N}(0, \sigma^2_{\xi})$.

### Individual trajectory

The *individual trajectory* for a given individual $ i $ is denoted as $ \gamma_i(t) $ and represents the disease progression of the patient $i$. It can be indexed by $k$ when $K$ outcomes are estimated.  

### Latent disease age

The *latent disease age*, for a given individual $ i $, is denoted as $ \psi_i(t) $.

It represents a transformation from chronological time $t$ to latent disease age that is related to its stage in the disease and is defined as:

$$
\psi_i(t) = e^{\xi_i}(t - \tau_i) + t_0
$$

where :

- $ e^{\xi_i} $ is the [individual speed factor](#individual-speed-factor) of individual $ i $.
- $ \tau_i $ is the [estimated reference time](#estimated-reference-time) of individual $ i $.
- $ t_0 $ is the [population reference time](#population-reference-time).

### Mixing matrix

The *mixing matrix* is denoted as $ \mathbf{A} $ and is defined as **TODO**.

### Population reference time

The *population reference time* is denoted as $ t_0 $ and is defined as **TODO**.

### Sources

The *sources*, for a given individual $ i $, are denoted as $ \mathbf{s}_i$ and defined as **TODO**.

### Space shift

The *space shift*, for a given individual $ i $ and a given XX $ k $, is denoted as $ w_{i,k} $ and defined as **TODO**.

### Survival shift

The *survival shift*, for a given individual $ i $ and a given event $ l $ is denoted as $ u_{i,l} $ and defined as:

$$
u_{i,l} = \sum_{m=1}^{N_s} \zeta_{l,m} s_{i,m}
$$

where:

- $ N $ is **TODO**.
- $ \zeta_{l,m} $ is **TODO**.
- $ s_{i,m} $ is **TODO**.

