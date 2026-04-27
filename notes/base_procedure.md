## 3.1 Method

For the simulation study we followed the ADEMP recommendation ([Morris et al., 2019]) and got inspiration
from the work of ([Lavalley-Morelle et al., 2024]) also using MCMC-SAEM estimator.

### 3.1.1 Aims

The simulation study aimed to validate the Joint Temporal model. For model parameters (θ), we evaluate
the estimation of the model parameters associated with their standard error. For random effects, we assessed
the correlation between the estimated values and the simulated ones.

### 3.1.2 Data-generating mechanism

Data were simulated under the Joint Temporal model structure with the following procedure:

1. We simulated random effects using $\xi_i \sim \mathcal{N}\left(0, \sigma^2_\xi\right)$ and $\tau_i \sim \mathcal{N}\left(t_0, \sigma^2_\tau\right)$.
2. We modelled the age at first visit $t_{i,0}$ as $t_{i,0} = \tau_i + \delta_{f_i}$ with $\delta_{f_i} \sim \mathcal{N}\left(\bar{\delta_f}, \sigma^2_{\delta_f}\right)$.
3. We set a time of follow-up per patient $T_{f_i}$, with $T_{f_i} \sim \mathcal{N}\left(\bar{T_f}, \sigma^2_{T_f}\right)$ and a time between two visits $\delta v_{i,j} \sim \mathcal{N}\left(\bar{\delta_v}, \sigma^2_{\delta_v}\right)$ to simulate $n_i$ visits until $t_{n_i} \leq t_{i,0} + T_{f_i} < t_{n_i+1}$.
4. We set the value of the outcome at each visit using a beta distribution of concentration $p$ and mode $\gamma_0(\psi_i(t_{i,j}))$ so that $y_{i,j} \sim \mathcal{B}\left(\gamma_0(\psi_i(t_{i,j})), p\right)$.
5. For each patient, we simulated the event $T_{e_i}$ through a Weibull distribution using $T_{e_i} \sim e^{-\xi_i}W(\nu, \rho) + \tau_i$.
6. We considered that the event stopped the follow-up and that the follow-up censored the event. Thus all the visits after the event were censored: $t_{i,j} > T_{e_i}$ and events after the last visit were censored: $t_{i,\max(j)} < T_{e_i}$.

We simulated ALS real-like data using an ALS dataset, PRO-ACT, described in part 4.1.1, to get
real-like values for parameters. Note that some parameters values were adjusted, such as the population
estimated reference time, to make sure that no patient was left censored (Table 4 in appendix). Parameters
directly associated with the disease have been extracted from data analysis, using the Longitudinal and AFT
models (Figures 4, 5 in appendix). We simulated M=100 datasets with N=200 patients. The parameters
used for the simulation study are summarised in Table 4 in appendix.

### 3.1.3 Estimands

We initialised the Joint Temporal model with the Longitudinal model trained for 2,000 iterations and a
survival Weibull model. Then, we ran the Joint Temporal model with 70,000 iterations (on average an hour)
with the last 10,000 of the Robbins-Monro convergence phase ([Robbins and Monro, 1951]) to extract the
mean of the posterior.

We validated the estimation of the model parameters $\theta = \{\sigma_\xi, \sigma_\tau, t_0, \tilde{g}, \tilde{v_0}, \tilde{\nu}, \tilde{\rho}, \sigma\}$ extracted by
the Robbins-Monro convergence phase. As we use a Gaussian approximation for the noise, we estimated $\sigma$
using the noisy simulation and the expected perfect curve from the random effect used for simulation. For
the random effects $(\tau_i, \xi_i)$, to reduce the computation complexity, we extracted the mean of the last 10,000
iterations for each individual.

### 3.1.4 Performance metrics

To assess the estimation performances of the estimated model parameters ($\hat{\theta}$) over the M datasets simulated
for the scenario, we reported:

- the Relative Bias: $RB(\hat{\theta}) = \frac{1}{M}\sum_{m=1}^{M}\frac{\hat{\theta}^{(m)}-\theta}{\theta} \times 100$
- Relative Root Mean Square Errors: $RRMSE(\hat{\theta}) = \sqrt{\frac{1}{M}\sum_{m=1}^{M}\left(\frac{\hat{\theta}^{(m)}-\theta}{\theta} \times 100\right)^2}$
- Relative Estimation Errors: $REE(m) = \frac{\hat{\theta}^{(m)}-\theta}{\theta} \times 100$

To assess the Standard Error of the estimated model parameters ($\hat{\theta}$), we reported:

- the relative empirical Standard Error: $SE_{emp}(\hat{\theta}) = \sqrt{\frac{\sum_{m=1}^{M}(\hat{\theta}^{(m)}-\bar{\hat{\theta}})^2}{m-1}}$, $RSE_{emp}(\hat{\theta}) = \frac{SE_{emp}(\hat{\theta})}{\bar{\hat{\theta}}}$
- the coverage rates (CR): defined as the proportion of datasets for which $\theta$ belonged to $[\hat{\theta}-1.96SE(\hat{\theta}),\hat{\theta}+1.96SE(\hat{\theta})]$ with their 95% confidence intervals (CI) computed using the exact Clopper Pearson method.

The estimation of the random effects $(\tau_i, \xi_i)$ was assessed using the intraclass correlation between the mean
of each individual and the true value that enabled the simulation.
