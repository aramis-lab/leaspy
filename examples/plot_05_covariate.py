"""
Covariate Model
===============

This example fits a logistic model with a patient-level covariate.
"""

# %%
# .. warning::
#
#    The covariate model is in beta. We will not provide support for this model.
#    Its methods may change in the next version, which is expected very shortly.

# %%
# We load synthetic longitudinal data containing one binary covariate, ``cov_A``.
import os

import pandas as pd

import leaspy
from leaspy.io.data import Data

data_path = os.path.join(
    os.path.dirname(leaspy.__file__),
    "datasets/data/simulated_data_for_covariate.csv",
)
df = pd.read_csv(data_path).set_index(["ID", "TIME"])
df.head()

# %%
# We identify ``cov_A`` as the covariate when creating the Leaspy data container.
data = Data.from_dataframe(
    df,
    "covariate",
    factory_kws={"covariate_names": ["cov_A"]},
)

# %%
# First, we fit a standard logistic model to initialize the shared parameters.
# This is faster than fitting the covariate model directly, even if it is not strictly necessary.
from leaspy.models import CovariateLogisticModel, LogisticModelSchiratti

logistic_model = LogisticModelSchiratti(
    name="logistic-initialization",
    source_dimension=2,
)
logistic_model.fit(
    data,
    "mcmc_saem",
    seed=0,
    n_iter=100,
    progress_bar=False,
)

# %%
# We then fit the covariate model, initialized from the logistic model.
#
# .. note::
#
#    Only 1000 iterations are run here to keep this example short. In practice,
#    MCMC-SAEM typically needs several thousand iterations to reach convergence,
#    so the values obtained below should not be over-interpreted as a final,
#    converged result. This example only aims at showing *how* to read and
#    interpret the covariate model's outputs once it has been fitted.
covariate_model = CovariateLogisticModel(
    name="covariate-model",
    source_dimension=2,
    init_from_model=logistic_model,
)
covariate_model.fit(
    data,
    "mcmc_saem",
    seed=42,
    n_iter=1000,
    progress_bar=False,
    path="_outputs",
    save_periodicity=50,
    overwrite_logs_folder=True,
)

# %%
# The fitted population and covariate-effect parameters are available below.
# ``covariate_model.parameters`` exposes the quantities that MCMC-SAEM
# directly re-estimates at each iteration (the ``ModelParameter``): the
# population-average parameters ``t0_mean``, ``log_g_mean`` and
# ``log_v0_mean``, common to every patient, and the covariate effect sizes
# ``delta_t0_mean``, ``delta_g_mean`` and ``delta_v0_mean``. For a patient,
# each population parameter is shifted from that common reference by
# ``cov_A * delta`` (in log-scale for ``g`` and ``v0``).
covariate_model.parameters

# %%
# A non-zero ``delta``, however, does not necessarily mean that ``cov_A`` has
# an effect: each ``delta`` is paired with a binary latent variable ``gamma``
# (``gamma_t0``, ``gamma_g``, ``gamma_v0``), sampled at every MCMC-SAEM
# iteration, which acts as an on/off switch on the corresponding ``delta``:
#
# * ``gamma = 1``: the covariate effect is considered active for this
#   parameter, and the corresponding ``delta`` can be interpreted directly,
#   as the shift applied to the population reference value above.
# * ``gamma = 0``: the effect is switched off (in the model, ``delta`` is
#   multiplied by ``gamma`` before being used), so the estimated ``delta``
#   is *not* interpretable — it should be treated as if there were no effect.
#
# ``gamma`` is a latent variable, not a ``ModelParameter``, so it does not
# appear in ``covariate_model.parameters``: it must be read from the
# convergence traces saved during the fit, in
# ``_outputs/parameter_convergence/``. We read the last saved value of each
# ``gamma`` below, and use it to decide whether the matching ``delta`` should
# be interpreted.
convergence_path = "_outputs/parameter_convergence"
gamma_t0 = pd.read_csv(
    f"{convergence_path}/gamma_t0.csv", header=None, index_col=0
).iloc[-1, 0]
gamma_g = pd.read_csv(f"{convergence_path}/gamma_g.csv", header=None, index_col=0).iloc[
    -1
]
gamma_v0 = pd.read_csv(
    f"{convergence_path}/gamma_v0.csv", header=None, index_col=0
).iloc[-1]

delta_t0 = covariate_model.parameters["delta_t0_mean"]
delta_g = covariate_model.parameters["delta_g_mean"]
delta_v0 = covariate_model.parameters["delta_v0_mean"]

print("Effect of cov_A on t0 (population time-shift):")
if gamma_t0 == 1:
    print(f"  active   -> delta_t0 = {delta_t0[0].item():.3f}")
else:
    print("  inactive -> delta_t0 is not interpretable")

print("\nEffect of cov_A on g and v0, feature by feature:")
for k, feature in enumerate(covariate_model.features):
    g_active = gamma_g.iloc[k] == 1
    v0_active = gamma_v0.iloc[k] == 1
    g_value = (
        f"delta_g = {delta_g[k, 0].item():.3f}" if g_active else "not interpretable"
    )
    v0_value = (
        f"delta_v0 = {delta_v0[k, 0].item():.3f}" if v0_active else "not interpretable"
    )
    print(f"  {feature:<8} g:  {'active  ' if g_active else 'inactive'} -> {g_value}")
    print(f"  {feature:<8} v0: {'active  ' if v0_active else 'inactive'} -> {v0_value}")

# %%
# The next example shows how to simulate data from a fitted model:
# see :doc:`plot_06_simulate`.
