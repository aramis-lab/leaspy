# Mathematical Background

## Mixed effect models

*Talk about mixed-effect models and their ability to handle missing values*



## Riemanian framework

In this article, Einsein {cite}`einstein1905`.

The Spatiotemporal model was first introduced by \cite{schiratti_mixed-effects_2015} and was more broadly used for longitudinal modelling of neurodegenerative diseases (\cite{schiratti_methods_2017, koval_learning_2020}) as described in chapter \ref{sota_chap}. As presented in Figure \ref{fig:als_intuition} the model draws a parallel between a clinical and a Riemannian point of view of the disease progression. The idea is to see the variability of the disease progression as a Riemannian manifold where the longitudinal observations $y_{i,j,k}$  are aligned in an individual trajectory $\gamma_i$ that traverses the manifold. 

\begin{figure}[h!]
\begin{center}
    \includegraphics[width=\linewidth]{Images/3_als/intuition.pdf}
\end{center}
    \caption{From clinical to Riemannian point of view} \label{fig:als_intuition}
   \textit{ On the left, the progression of four clinical outcomes for one patient is represented depending on the age of the patient. The graph displays the individual progression of one patient on a grid detailing the typical progression of the disease, as it is done in health diaries for BMI curves. This represents how a clinician is used to see the progression of the patient. On the right, the same patient progression is represented but this time in a disease space (manifold) built thanks to the knowledge extracted from the four clinical outcomes. This represents the Riemannian point of view of the progression of the patient.}
\end{figure}

The shape of the disease progression (linear, logistic ...) is defined by the choice of the Riemannian metric applied to the manifold. For instance, the manifold $\mathbb{R}^n$ equipped with Euclidean metric gives straight lines trajectories and thus straight lines disease progression. As we studied mainly clinical scores, with curvilinearity, and potential floor or ceiling effects (\cite{gordon_progression_2010}), we selected a metric that enables modelling the logistic progression of the outcomes. 

\paragraph*{}To separate an average disease progression from the individual progression, a mixed-effects model structure is added to the trajectories.  Any trajectory $\gamma$ (geodesic) can be defined by the two parameters of its initial condition at a time $t_0$: the initial position $\gamma(t_0) = p$ and the initial speed $\dot{\gamma}(t_0) = v_0$. The average trajectory  $\gamma_0$ is thus parametrized by its initial conditions ($t_0, p, v_0$). This average longitudinal process is further described in section \ref{als_fixed_effects}.

From there, the individual trajectory $\gamma_i(t)$ could be defined. First, a temporal variation is enable with varying degrees of individual earliness $\tau_i$ and speed $e^{\xi_i}$, using a latent disease age $\psi_i(t)$. In terms of initial definition, these variations could be seen as ($t_0 + \tau_i, p, v_0e^{\xi_i}$). Subsequently, variation in terms of disease presentation, which corresponds to modifying the order of degradation of the various outcomes, is allowed. From a Riemannian point of view, the trajectory is spatially adjusted playing on the initial position $p$. It is done thanks to a vector in the tangent space of the trajectory that modified the trajectory in the sense of the Exp-parallelisation. All these individual parameters are further described in section \ref{als_model_re}.

## References

```{bibliography}
:filter: docname in docnames
```
