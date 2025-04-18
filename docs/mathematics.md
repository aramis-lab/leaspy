# Mathematical Background

## Mixed effect models

*Talk about mixed-effect models and their ability to handle missing values*


In this article, Einsein {cite}`einstein1905`.
## Riemanian framework

### Disease progression represented as trajectory
As presented in the figure the model draws a parallel between a clinical and a Riemannian point of view of the disease progression. The idea is to see the variability of the disease progression as a Riemannian manifold where the longitudinal observations $$y_{i,j,k}$$  are aligned in an individual trajectory $$\gamma_i$$ that traverses the manifold. 

![intuition](./_static/images/intuition.png)


__From clinical to Riemannian point of view__
_On the left, the progression of four clinical outcomes for one patient is represented depending on the age of the patient. The graph displays the individual progression of one patient on a grid detailing the typical progression of the disease, as it is done in health diaries for BMI curves. This represents how a clinician is used to see the progression of the patient. On the right, the same patient progression is represented but this time in a disease space (manifold) built thanks to the knowledge extracted from the four clinical outcomes. This represents the Riemannian point of view of the progression of the patient._

### Trajectory shape defined by Riemmanian metric
The shape of the disease progression (linear, logistic ...) is defined by the choice of the Riemannian metric ($$G(p)$$) applied to the manifold. For instance, the manifold $$\mathbb{R}^n$$ equipped with Euclidean metric gives straight lines trajectories and thus straight lines disease progression.  For a K-dimensional dataset, we used the product manifold of a 1-dimensional metric. The disease trajectory is a geodesic if and only if it satisfy a differential equation with the metric, further described in \cite{koval_learning_2020} (p.169). This enable to get from the metric formulation the shape of the curve in time.

In the case of logistic progression, often used for clinical scores, with curvilinearity, and potential floor or ceiling effects (\cite{gordon_progression_2010}), the metric is $G(p) = \frac{1}{p^2(1-p)^2}$ in a manifold $(0, 1)$.


### Individual variability define as initial conditions
To separate an average disease progression from the individual progression, a mixed-effects model structure is added to the trajectories.  Any trajectory $$\gamma$$ (geodesic) can be defined by the two parameters of its initial condition at a time $$t_0$$: the initial position $$\gamma(t_0) = p$$ and the initial speed $$\dot{\gamma}(t_0) = v_0$$. The average trajectory  $$\gamma_0$$ is thus parametrized by its initial conditions ($$t_0, p, v_0$$). This average longitudinal process is further described in section \ref{als_fixed_effects}.

From there, the individual trajectory $$\gamma_i(t)$$ could be defined. First, a temporal variation is enable with varying degrees of individual earliness $\tau_i$ and speed $$e^{\xi_i}$$, using a latent disease age $$\psi_i(t)$$. In terms of initial definition, these variations could be seen as ($$t_0 + \tau_i, p, v_0e^{\xi_i}$$). Subsequently, variation in terms of disease presentation, which corresponds to modifying the order of degradation of the various outcomes, is allowed. From a Riemannian point of view, the trajectory is spatially adjusted playing on the initial position $p$. It is done thanks to a vector in the tangent space of the trajectory that modified the trajectory in the sense of the Exp-parallelisation. All these individual parameters are further described in section [TODO?]


![intuition](./_static/images/pop_to_ind.png)
__Temporal and spatial random effects: from population to individual progression (adapted from \cite{koval_ad_2021})__
_This figure presents from two points of view (clinical and Riemannian) how the three types of random effects (two temporal and one spatial) enable to modify the population average progression to calibrate the patient observations. Clinical: Two normalised clinical scores (blue and orange) (0: the healthiest value, +1: the maximum pathological change) depending on the age of the patients The scatter represents the real observed values for one patient at different visits. Riemannian: The same two normalised scores are represented but this time depending on each other. The scatter represents the same real observed values as in the clinical version. The black cross on the curve corresponds to what is modelled at the visit ages of the patient._
- Population progression (1.a., 1.b.): Population average trajectory compared to the observed values of the patient. 
- Time Shift (2.a., 2.b.): The progression starts earlier due to the individual estimated reference time, Clinical graph: the curves are shifted on the left, Riemannian graph: black crosses are shifted on the right following the trajectory (for the same age the patient is more advanced).
- Speed factor (3.a., 3.b.): The progression speed increases, Clinical graph: the curves become steeper, Riemannian graph: black crosses get further from each other on the trajectory (for the same time of follow-up a wider portion of the trajectory is observed).
- Space Shift (4.a., 4.b.): the blue curves progress before the orange curve, Clinical graph: the curves are shifted in opposite directions, Riemannian graph: most of the blue (resp. orange) score value is observed for an orange (resp. blue) value of 0 (resp. 1)
- Individual progression (5.a., 5.b.): The modelled curves fit the observations, Riemannian graph: the black crosses are close to the observed values._

## References

```{bibliography}
:filter: docname in docnames
```