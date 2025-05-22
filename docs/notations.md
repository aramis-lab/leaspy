# Notations

This page contains the definitions and notations of the variables used in Leaspy.

## Notations, concepts, and names in DAG

The following table displays the relationships between a variable mathematical notation, the associated concept, and the string name used in the DAG if this variable is handled this way.

| Notation   | Concept                                                | Name in the DAG |
|------------|--------------------------------------------------------|-----------------|
| $ \tau_i $ | [estimated reference time](#estimated-reference-time)  | `tau`            |
|  $ e^{\xi_i} $ | [individual speed factor](#individual-speed-factor) | ???             |
| $ \gamma_i(t) $ | [individual trajectory](#individual-trajectory) | ???             |
| $ \psi_i(t) $ | [latent disease age](#latent-disease-age) | ???             |
| $ \mathbf{A} $ | [mixing matrix](#mixing-matrix) | `mixing_matrix` |
| $ t_0 $ | [population reference time](#population-reference-time) | ???             |
| $ \mathbf{s}_i$ | [sources](#sources) | `sources` |
| $ w_{i,k} $ | [space shift](#space-shift) | `space_shifts`  |
| $ u_{i,l} $ | [survival shift](#survival-shift) | ???             |

## Concept definitions

This section explains each mathematical concept and provides mathematical definitions when needed.

### Estimated reference time

The *estimated reference time* for a given individual $ i $ is denoted as $ \tau_i $ and is defined as **TODO**.

### Individual speed factor

The *individual speed factor* for a given individual $ i $ is denoted as $ e^{\xi_i} $ and is defined as **TODO**.

### Individual trajectory

The *individual trajectory* for a given individual $ i $ is denoted as $ \gamma_i(t) $ and represents **TODO**.

### Latent disease age

The *latent disease age*, for a given individual $ i $, is denoted as $ \psi_i(t) $.

It represents **TODO** and is defined as:

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

