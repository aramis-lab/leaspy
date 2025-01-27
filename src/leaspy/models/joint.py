import pandas as pd
import torch
from lifelines import WeibullFitter

from leaspy.exceptions import LeaspyInputError
from leaspy.io.data.dataset import Dataset
from leaspy.models.base import InitializationMethod
from leaspy.models.multivariate import LogisticMultivariateModel
from leaspy.models.obs_models import observation_model_factory
from leaspy.utils.docs import doc_with_super  # doc_with_
from leaspy.utils.functional import Exp, MatMul, Sum
from leaspy.utils.typing import DictParams, KwargsType, Optional
from leaspy.utils.weighted_tensor import WeightedTensor
from leaspy.variables.distributions import Normal
from leaspy.variables.specs import (
    Hyperparameter,
    LinkedVariable,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
    VariablesValuesRO,
)
from leaspy.variables.state import State

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either
# model values or model values + jacobians wrt individual parameters

# TODO refact? subclass or other proper code technique to extract model's concrete
#  formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class JointModel(LogisticMultivariateModel):
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
        * If `name` is not one of allowed sub-type: 'univariate_linear' or 'univariate_logistic'
        * If hyperparameters are inconsistent
    """

    init_tolerance: float = 0.3

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        obs_models_to_string = [o.to_string() for o in self.obs_models]

        if (self.dimension == 1) or (self.source_dimension == 0):
            if "weibull-right-censored-with-sources" in obs_models_to_string:
                raise LeaspyInputError(
                    "You cannot use a weibull with sources for an univariate model"
                )
            if "weibull-right-censored" not in obs_models_to_string:
                self.obs_models += (
                    observation_model_factory(
                        "weibull-right-censored",
                        nu="nu",
                        rho="rho",
                        xi="xi",
                        tau="tau",
                    ),
                )
                obs_models_to_string += ["weibull-right-censored"]
        else:
            if "weibull-right-censored" in obs_models_to_string:
                warnings.warn(
                    "You are using a multivariate model with a weibull model without sources"
                )
            elif "weibull-right-censored-with-sources" not in obs_models_to_string:
                self.obs_models += (
                    observation_model_factory(
                        "weibull-right-censored-with-sources",
                        nu="nu",
                        rho="rho",
                        zeta="zeta",
                        xi="xi",
                        tau="tau",
                        sources="sources",
                    ),
                )
                obs_models_to_string += ["weibull-right-censored-with-sources"]

        variables_to_track = [
            "nu",
            "rho",
            "nll_attach_y",
            "nll_attach_event",
        ]
        if "weibull-right-censored-with-sources" in obs_models_to_string:
            variables_to_track += ["zeta", "survival_shifts"]
        self.tracked_variables = self.tracked_variables.union(set(variables_to_track))

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
        d.update(
            # PRIORS
            n_log_nu_mean=ModelParameter.for_pop_mean(
                "n_log_nu",
                shape=(self.nb_events,),
            ),
            n_log_nu_std=Hyperparameter(0.01),
            log_rho_mean=ModelParameter.for_pop_mean(
                "log_rho",
                shape=(self.nb_events,),
            ),
            log_rho_std=Hyperparameter(0.01),
            # LATENT VARS
            n_log_nu=PopulationLatentVariable(
                Normal("n_log_nu_mean", "n_log_nu_std"),
            ),
            log_rho=PopulationLatentVariable(
                Normal("log_rho_mean", "log_rho_std"),
            ),
            # DERIVED VARS
            nu=LinkedVariable(self.exp_neg_n_log_nu),
            rho=LinkedVariable(
                Exp("log_rho"),
            ),
        )
        d.update(
            nll_attach=LinkedVariable(Sum("nll_attach_y", "nll_attach_event")),
            nll_attach_ind=LinkedVariable(
                Sum("nll_attach_y_ind", "nll_attach_event_ind")
            ),
        )
        if self.source_dimension >= 1:
            d.update(
                zeta_mean=ModelParameter.for_pop_mean(
                    "zeta",
                    shape=(self.source_dimension, self.nb_events),
                ),
                zeta_std=Hyperparameter(0.01),
                zeta=PopulationLatentVariable(
                    Normal("zeta_mean", "zeta_std"),
                    sampling_kws={"scale": 0.5},  # cf. GibbsSampler (for retro-compat)
                ),
                survival_shifts=LinkedVariable(MatMul("sources", "zeta")),
            )

        return d

    @staticmethod
    def exp_neg_n_log_nu(
        *,
        n_log_nu: torch.Tensor,  # TODO: TensorOrWeightedTensor?
    ) -> torch.Tensor:
        return torch.exp(-1 * n_log_nu)

    @classmethod
    def _center_xi_realizations(cls, state: State) -> None:
        """
        Center the ``xi`` realizations in place.

        .. note::
            This operation does not change the orthonormal basis
            (since the resulting ``v0`` is collinear to the previous one)
            Nor all model computations (only ``v0 * exp(xi_i)`` matters),
            it is only intended for model identifiability / ``xi_i`` regularization
            <!> all operations are performed in "log" space (``v0`` is log'ed)

        Parameters
        ----------
        realizations : :class:`.CollectionRealization`
            The realizations to use for updating the :term:`MCMC` toolbox.
        """
        mean_xi = torch.mean(state["xi"])
        state["xi"] = state["xi"] - mean_xi
        state["log_v0"] = state["log_v0"] + mean_xi
        state["n_log_nu"] = state["n_log_nu"] + mean_xi

        # TODO: find a way to prevent re-computation of orthonormal basis since it should not have changed (v0_collinear update)
        # self.update_MCMC_toolbox({'v0_collinear'}, realizations)

    def _load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        self.nb_events = hyperparameters.pop("nb_events", 1)
        super()._load_hyperparameters(hyperparameters)

    def to_dict(self, *, with_mixing_matrix: bool = True) -> KwargsType:
        dict_params = super().to_dict(with_mixing_matrix=with_mixing_matrix)
        dict_params["nb_events"] = self.nb_events
        return dict_params

    def _validate_compatibility_of_dataset(
        self, dataset: Optional[Dataset] = None
    ) -> None:
        super()._validate_compatibility_of_dataset(dataset)
        # Check that there is only one event stored

        if not set(dataset.event_bool.unique().tolist()) == set([False, True]):
            raise LeaspyInputError(
                "You are using a one event model, your event_bool value should only contain 0 and 1, "
                "with at least one censored event and one observed event"
            )

    def _compute_initial_values_for_model_parameters(
        self,
        dataset: Dataset,
        method: InitializationMethod,
    ) -> VariablesValuesRO:
        from leaspy.models.utilities import torch_round

        params = super()._compute_initial_values_for_model_parameters(dataset, method)

        new_parameters = self._estimate_initial_event_parameters(dataset)
        new_rounded_parameters = {
            str(p): torch_round(v.to(torch.float32)) for p, v in new_parameters.items()
        }

        params.update(new_rounded_parameters)
        return params

    def put_individual_parameters(self, state: State, dataset: Dataset):
        df = dataset.to_pandas().reset_index("TIME").groupby("ID").min()

        # Initialise individual parameters if they are not already initialised
        if not state.are_variables_set(("xi", "tau")):
            df_ind = df["TIME"].to_frame(name="tau")
            df_ind["xi"] = 0.0
        else:
            df_ind = pd.DataFrame(
                torch.concat([state["xi"], state["tau"]], axis=1),
                columns=["xi", "tau"],
                index=df.index,
            )

        # Set the right initialisation point fpr barrier methods
        df_inter = pd.concat(
            [df["EVENT_TIME"] - self.init_tolerance, df_ind["tau"]], axis=1
        )
        df_ind["tau"] = df_inter.min(axis=1)

        if self.source_dimension > 0:
            for i in range(self.source_dimension):
                df_ind[f"sources_{i}"] = 0.0

        with state.auto_fork(None):
            state.put_individual_latent_variables(df=df_ind)

    def _estimate_initial_event_parameters(self, dataset: Dataset) -> VariablesValuesRO:
        log_rho_mean = [0] * self.nb_events
        n_log_nu_mean = [0] * self.nb_events

        df_ind = dataset.to_pandas().reset_index("TIME").groupby("ID").min()
        approx_tau = torch.tensor(df_ind["TIME"].values) - self.init_tolerance

        for i in range(self.nb_events):
            wbf = WeibullFitter().fit(
                dataset.event_time[:, i] - approx_tau, dataset.event_bool[:, i]
            )
            log_rho_mean[i] = torch.log(torch.tensor(wbf.rho_))
            n_log_nu_mean[i] = -torch.log(torch.tensor(wbf.lambda_))

        event_params = {
            "log_rho_mean": torch.tensor(log_rho_mean),
            "n_log_nu_mean": torch.tensor(n_log_nu_mean),
        }

        if self.source_dimension > 0:
            event_params["zeta_mean"] = torch.zeros(
                self.source_dimension, self.nb_events
            )

        return event_params

    def compute_individual_trajectory(
        self,
        timepoints,
        individual_parameters: DictParams,
        *,
        skip_ips_checks: bool = False,
    ) -> torch.Tensor:
        """
        Compute scores values at the given time-point(s) given a subject's individual parameters.

        Nota: model uses its current internal state.

        Parameters
        ----------
        timepoints : scalar or array_like[scalar] (:obj:`list`, :obj:`tuple`, :class:`numpy.ndarray`)
            Contains the age(s) of the subject.
        individual_parameters : :obj:`dict`
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like.
        skip_ips_checks : :obj:`bool` (default: ``False``)
            Flag to skip consistency/compatibility checks and tensorization
            of ``individual_parameters`` when it was done earlier (speed-up).

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's scores computed at the given age(s)
            Shape of tensor is ``(1, n_tpts, n_features)``.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If computation is tried on more than 1 individual.
        :exc:`.LeaspyIndividualParamsInputError`
            if invalid individual parameters.
        """
        self._check_individual_parameters_provided(individual_parameters.keys())
        timepoints, individual_parameters = self._get_tensorized_inputs(
            timepoints, individual_parameters, skip_ips_checks=skip_ips_checks
        )

        # TODO? ability to revert back after **several** assignments?
        # instead of cloning the state for this op?
        local_state = self.state.clone(disable_auto_fork=True)

        self._put_data_timepoints(local_state, timepoints)
        local_state.put(
            "event",
            WeightedTensor(timepoints.T, torch.zeros(timepoints.T.shape).bool()),
        )

        for ip, ip_v in individual_parameters.items():
            local_state[ip] = ip_v
        # reshape survival_event from (len(timepoints)) to (1, len(timepoints), 1) so it is compatible with the
        # model shape
        return torch.cat(
            (
                local_state["model"],
                local_state["predictions_event"].expand(
                    (1, timepoints.shape[1], self.nb_events)
                ),
            ),
            2,
        )
