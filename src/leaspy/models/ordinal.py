from operator import itemgetter
from typing import Optional

import numpy as np
import torch
from scipy.optimize import minimize

from leaspy.io.data.dataset import Dataset
from leaspy.utils.docs import doc_with_super
from leaspy.utils.functional import Exp
from leaspy.utils.weighted_tensor import (
    TensorOrWeightedTensor,
    WeightedTensor,
    unsqueeze_right,
)
from leaspy.variables.distributions import Normal
from leaspy.variables.specs import (
    LVL_FT,
    Hyperparameter,
    LinkedVariable,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
    VariableNameToValueMapping,
)

from .base import InitializationMethod
from .logistic import LogisticModel

@doc_with_super(if_other_signature="force")
class OrdinalModel(LogisticModel):
    """
    Manifold model for multiple variables of interest (logistic or linear formulation).

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    **kwargs
        Hyperparameters of the model (including `noise_model`)

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        * If hyperparameters are inconsistent
    """
    type = "ordinal"

    def __init__(self, name: str, **kwargs):
        max_levels = kwargs.pop("max_levels", None)
        kwargs.setdefault("obs_models", "ordinal")
        super().__init__(name, **kwargs)
        if max_levels is not None:
            self.max_levels = {k: int(v) for k, v in max_levels.items()}
            self.max_level = max(self.max_levels.values())
        self.tracked_variables.add("deltas")

    def to_dict(self, **kwargs) -> dict:
        d = super().to_dict(**kwargs)
        if hasattr(self, "max_levels"):
            d["max_levels"] = self.max_levels
        return d

    def _get_deltas_mask(self) -> torch.Tensor:
        """Return True for valid deltas and False for padded positions."""
        return torch.tensor(
            [
                [j < max_level - 1 for j in range(self.max_level - 1)]
                for max_level in self.max_levels.values()
            ],
            dtype=torch.bool,
        )

    def initialize(self, dataset: Optional[Dataset] = None) -> None:
        """Overloads base model initialization (in particular to handle internal model State).

        <!> We do not put data variables in internal model state at this stage (done in algorithm)

        Parameters
        ----------
        dataset : :class:`~leaspy.io.data.dataset.Dataset`, optional
            Input dataset from which to initialize the model.
        """
        self.max_levels = dataset.get_max_levels()
        self.max_level = max(self.max_levels.values())
        super().initialize(dataset=dataset)

    def ordinal_time_reparametrization(
        self,
        *,
        t: TensorOrWeightedTensor[float],
        alpha: torch.Tensor,
        tau: torch.Tensor,
        deltas: torch.Tensor,
    ) -> TensorOrWeightedTensor[float]:
        """
        Tensorized time reparametrization formula.

        .. warning::
            Shapes of tensors must be compatible between them.

        Parameters
        ----------
        t : :class:`torch.Tensor`
            Timepoints to reparametrize
        alpha : :class:`torch.Tensor`
            Acceleration factors of individual(s)
        tau : :class:`torch.Tensor`
            Time-shift(s) of individual(s)

        Returns
        -------
        :class:`torch.Tensor`
            Reparametrized time of same shape as `timepoints`
        """
        reparametrized_time = alpha * (t - tau)
        reparametrized_time = unsqueeze_right(
            reparametrized_time, ndim=2
        )  # add dim of ordinal level
        t0 = torch.zeros((self.dimension, 1))
        deltas = torch.cat(
            (t0, deltas), dim=-1
        )  # Add zero for P(X >= 1), parametrized by standard leaspy
        deltas = deltas[None, None, ...]  # add (ind, tpts) dimensions
        reparametrized_time = reparametrized_time - deltas.cumsum(dim=-1)
        return reparametrized_time

    def _compute_initial_values_for_model_parameters(
        self,
        dataset: Dataset,
    ) -> VariableNameToValueMapping:
        """Compute initial values for model parameters and for the ordinal deltas parameters
        and initializes ordinal noise_model attributes.
        """
        from leaspy.models.utilities import torch_round

        df = dataset.to_pandas(apply_headers=True)

        # Ordinal-specific initialization of g, v0 and tau
        # using the anchor trajectory P(Y >= 1)

        time_values = (df.index.get_level_values("TIME").to_numpy(dtype=float))
        time_mu = float(np.mean(time_values))
        time_sigma = float(np.std(time_values))

        log_g_initial = []
        log_v0_initial = []


        for feature in self.features:
            feature_series = (df[feature].dropna())

            times = (feature_series.index.get_level_values("TIME").to_numpy(dtype=float))

            # Binary observations for the anchor cumulative curve
            y_binary = (feature_series.to_numpy(dtype=float) >= 1.0 ).astype(float)
            centered_times = (times - time_mu)

            # Initial intercept based on the empirical binary proportion
            empirical_probability = np.clip(y_binary.mean(),1e-2,1.0 - 1e-2,)

            intercept_initial = np.log(empirical_probability / (1.0 - empirical_probability))

            # Positive initial time slope
            initial_log_slope = np.log(0.1)

            def negative_log_likelihood(parameters_binary,):
                intercept = parameters_binary[0]

                # Enforce a positive progression slope
                slope = np.exp(parameters_binary[1])

                linear_predictor = (intercept + slope * centered_times)

                # Stable binary logistic negative log-likelihood:
                # log(1 + exp(eta)) - y * eta
                nll = np.sum(np.logaddexp(0.0,linear_predictor,) - y_binary * linear_predictor)

                # Very small regularization for numerical stability
                regularization = (1e-6 * np.sum(parameters_binary**2))

                return nll + regularization

            optimization_result = minimize(
                negative_log_likelihood,
                x0=np.array(
                    [
                        intercept_initial,
                        initial_log_slope,
                    ],
                    dtype=float,
                ),
                method="L-BFGS-B",
                bounds=[
                    (None, None),
                    (
                        np.log(1e-4),
                        np.log(1e2),
                    ),
                ],
            )

            if not optimization_result.success:
                raise RuntimeError(
                    "Binary logistic initialization failed "
                    f"for feature '{feature}': "
                    f"{optimization_result.message}"
                )

            intercept = float(optimization_result.x[0])

            log_effective_slope = float(optimization_result.x[1])

            effective_slope = np.exp(log_effective_slope)

            # At t = tau = time_mu:
            # logit P(Y >= 1) = -log(g)
            # Therefore: log(g) = -intercept
            log_g = -intercept

            g = np.exp(log_g)

            # Model effective time slope:
            # effective_slope = ((g + 1)^2 / g) * v0
            metric = ((g + 1.0) ** 2 / g)

            v0 = (effective_slope / metric)
            log_v0 = np.log(np.clip(v0,1e-8,None,))
            log_g_initial.append(log_g)
            log_v0_initial.append(log_v0)


        log_g_initial = torch.tensor(log_g_initial,dtype=torch.float32,)
        log_v0_initial = torch.tensor(log_v0_initial,dtype=torch.float32,)
        tau_initial = torch.tensor([time_mu],dtype=torch.float32,)

        # Build the initial population parameters
        if self.initialization_method == InitializationMethod.DEFAULT:
            log_g = log_g_initial
            log_v0 = log_v0_initial
            t0 = tau_initial
            betas = torch.zeros((self.dimension - 1,self.source_dimension,), dtype=torch.float32,)

        elif self.initialization_method == InitializationMethod.RANDOM:

            # Random perturbations around the pooled binary
            # logistic estimates.
            log_g = (log_g_initial + 0.1 * torch.randn_like(log_g_initial))
            log_v0 = (log_v0_initial + 0.1 * torch.randn_like(log_v0_initial))
            t0 = torch.normal(mean=tau_initial, std=torch.tensor([time_sigma], dtype=torch.float32,),)
            betas = torch.randn((self.dimension - 1,self.source_dimension,),dtype=torch.float32,)

        else:
            raise ValueError(
                "Unsupported initialization method: "
                f"{self.initialization_method}"
            )

        parameters = {
            "log_g_mean": log_g,
            "log_v0_mean": log_v0,
            "tau_mean": t0,
            "tau_std": self.tau_std,
            "xi_std": self.xi_std,
        }

        if self.source_dimension >= 1:
            parameters["betas_mean"] = betas

        parameters = {
            str(parameter_name): torch_round(parameter_value.to(torch.float32))
            for parameter_name, parameter_value
            in parameters.items()
        }

        # Existing ordinal deltas initialization
        deltas = {}
        for feature, s in df.items():  # preserve feature order
            max_level = int(
                s.max()
            )  # possible levels not observed in calibration data do not exist for us
            # we do not model P >= 0 (since constant = 1)
            # we compute stats on P(Y >= k) in our data
            first_age_gte = {}
            for k in range(1, max_level + 1):
                s_gte_k = (s >= k).groupby("ID")
                first_age_gte[k] = (
                    s_gte_k.idxmax().map(itemgetter(1)).where(s_gte_k.any())
                )  # (ID, TIME) tuple -> TIME or nan
            # we do not need a delta for our anchor curve P >= 1
            # so we start at k == 2
            delays = [
                (first_age_gte[k] - first_age_gte[k - 1]).mean(skipna=True).item()
                for k in range(2, max_level + 1)
            ]
            deltas[feature] = torch.log(torch.clamp(torch.tensor(delays), min=0.1))

        # we set the undefined deltas to be infinity to extend validity of formulas for them as well (and to avoid computations)
        deltas_ = float("inf") * torch.ones((len(deltas), self.max_level - 1))
        for i, name in enumerate(deltas):
            deltas_[i, : len(deltas[name])] = deltas[name]
        parameters["log_deltas_mean"] = deltas_
        return parameters

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables, derived variables,
        model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables :
            The specifications of the model's variables.
        """
        d = super().get_variables_specs()
        # del d["rt"]
        d.update(
            # PRIORS
            log_deltas_mean=ModelParameter.for_pop_mean(
                "log_deltas",
                shape=(self.dimension, self.max_level - 1),
                # shape=(self.dimension),
            ),
            log_deltas_std=Hyperparameter(0.1),
            # LATENT VARS
            log_deltas=PopulationLatentVariable(
                Normal("log_deltas_mean", "log_deltas_std"),
                sampling_kws={"mask": self._get_deltas_mask()},
                regularity_mask=self._get_deltas_mask(),
            ),
            # DERIVED VARS
            deltas=LinkedVariable(Exp("log_deltas")),
            ordinal_rt=LinkedVariable(self.ordinal_time_reparametrization),
        )
        return d

    @classmethod
    def model_no_sources(
        cls, *, ordinal_rt: torch.Tensor, metric, v0, g
    ) -> torch.Tensor:
        """Returns a model without source. A bit dirty?"""
        return cls.model_with_sources(
            ordinal_rt=ordinal_rt,
            metric=metric,
            v0=v0,
            g=g,
            space_shifts=torch.zeros((1, 1)),
        )

    @staticmethod
    def metric(*, g: torch.Tensor) -> torch.Tensor:
        """Used to define the corresponding variable."""
        return (g + 1) ** 2 / g

    @classmethod
    def model_with_sources(
        cls,
        *,
        ordinal_rt: TensorOrWeightedTensor[float],
        space_shifts: TensorOrWeightedTensor[float],
        metric: TensorOrWeightedTensor[float],
        v0: TensorOrWeightedTensor[float],
        g: TensorOrWeightedTensor[float],
    ) -> torch.Tensor:
        """Returns a model with sources."""
        # Shape: (Ni, Nt, Nfts)
        pop_s = (None, None, ..., None)
        w_model_logit = metric[pop_s] * (
            v0[pop_s] * ordinal_rt + space_shifts[:, None, ..., None]
        ) - torch.log(g[pop_s])
        model_logit, weights = WeightedTensor.get_filled_value_and_weight(
            w_model_logit, fill_value=0.0
        )
        model = torch.sigmoid(model_logit).nan_to_num(0.0)  # Fill nan with 0.
        return WeightedTensor(model, weights).weighted_value
