from abc import abstractmethod
from typing import Iterable, Optional, Union

import torch

from leaspy.exceptions import LeaspyIndividualParamsInputError, LeaspyModelInputError
from leaspy.io.data.dataset import Dataset
from leaspy.utils.typing import DictParams, DictParamsTorch, KwargsType
from leaspy.utils.weighted_tensor import TensorOrWeightedTensor, WeightedTensor
from leaspy.variables.specs import (
    LVL_FT,
    DataVariable,
    LatentVariableInitType,
    ModelParameter,
    NamedVariables,
    SuffStatsRO,
    SuffStatsRW,
)
from leaspy.variables.state import State

from .obs_models import ObservationModel
from .stateful import StatefulModel

__all__ = ["McmcSaemCompatibleModel"]


class McmcSaemCompatibleModel(StatefulModel):
    """Defines probabilistic models compatible with an MCMC SAEM estimation.

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.

    obs_models : ObservationModel or Iterable[ObservationModel]
        The noise model for observations (keyword-only parameter).

    fit_metrics : :obj:`dict`
        Metrics that should be measured during the fit of the model
        and reported back to the user.

    **kwargs
        Hyperparameters for the model

    Attributes
    ----------
    is_initialized : :obj:`bool`
        Indicates if the model is initialized.
    name : :obj:`str`
        The model's name.
    features : :obj:`list` of :obj:`str`
        Names of the model features.
    parameters : :obj:`dict`
        Contains the model's parameters
    obs_models : Tuple[ObservationModel, ...]
        The observation model(s) associated to the model.
    fit_metrics : :obj:`dict`
        Contains the metrics that are measured during the fit of the model and reported to the user.
    _state : State
        Private instance holding all values for model variables and their derived variables.
    """

    def __init__(
        self,
        name: str,
        *,
        # TODO? if we'd allow to pass a state there should be a all bunch of checks I guess? only "equality" of DAG is OK?
        # (WIP: cf. comment regarding inclusion of state here)
        # state: Optional[State] = None,
        # TODO? Factory of `ObservationModel` instead? (typically one would need the dimension to instantiate the `noise_std` variable of the right shape...)
        obs_models: Union[ObservationModel, Iterable[ObservationModel]],
        fit_metrics: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        if isinstance(obs_models, ObservationModel):
            obs_models = (obs_models,)
        self.obs_models = tuple(obs_models)
        # load hyperparameters
        # <!> some may still be missing at this point (e.g. `dimension`, `source_dimension`, ...)
        # (thus we sh/could NOT instantiate the DAG right now!)
        self._load_hyperparameters(kwargs)
        # TODO: dirty hack for now, cf. AbstractFitAlgo
        self.fit_metrics = fit_metrics

    @property
    def observation_model_names(self) -> list[str]:
        return [model.to_string() for model in self.obs_models]

    def has_observation_model_with_name(self, name: str) -> bool:
        return name in self.observation_model_names

    def to_dict(self, **kwargs) -> KwargsType:
        """Export model as a dictionary ready for export.

        Returns
        -------
        KwargsType :
            The model instance serialized as a dictionary.
        """
        d = super().to_dict()
        d.update(
            {
                "obs_models": {
                    obs_model.name: obs_model.to_string()
                    for obs_model in self.obs_models
                },
                "fit_metrics": self.fit_metrics,  # TODO improve
            }
        )
        return d

    @abstractmethod
    def _load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """Load model's hyperparameters.

        Parameters
        ----------
        hyperparameters : :obj:`dict` [ :obj:`str`, Any ]
            Contains the model's hyperparameters.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If any of the consistency checks fail.
        """

    @classmethod
    def _raise_if_unknown_hyperparameters(
        cls, known_hps: Iterable[str], given_hps: KwargsType
    ) -> None:
        """Raises a :exc:`.LeaspyModelInputError` if any unknown hyperparameter is provided to the model."""
        # TODO: replace with better logic from GenericModel in the future
        unexpected_hyperparameters = set(given_hps.keys()).difference(known_hps)
        if len(unexpected_hyperparameters) > 0:
            raise LeaspyModelInputError(
                f"Only {known_hps} are valid hyperparameters for {cls.__qualname__}. "
                f"Unknown hyperparameters provided: {unexpected_hyperparameters}."
            )

    @abstractmethod
    def _audit_individual_parameters(
        self, individual_parameters: DictParams
    ) -> KwargsType:
        """
        Perform various consistency and compatibility (with current model) checks
        on an individual parameters dict and outputs qualified information about it.

        TODO? move to IndividualParameters class?

        Parameters
        ----------
        individual_parameters : :obj:`dict` [param: str, Any]
            Contains some un-trusted individual parameters.
            If representing only one individual (in a multivariate model) it could be:
                * {'tau':0.1, 'xi':-0.3, 'sources':[0.1,...]}

            Or for multiple individuals:
                * {'tau':[0.1,0.2,...], 'xi':[-0.3,0.2,...], 'sources':[[0.1,...],[0,...],...]}

            In particular, a sources vector (if present) should always be a array_like, even if it is 1D

        Returns
        -------
        ips_info : :obj:`dict`
            * ``'nb_inds'`` : :obj:`int` >= 0
                Number of individuals present.
            * ``'tensorized_ips'`` : :obj:`dict` [ :obj:`str`, :class:`torch.Tensor` ]
                Tensorized version of individual parameters.
            * ``'tensorized_ips_gen'`` : generator
                Generator providing tensorized individual parameters for
                all individuals present (ordered as is).

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            If any of the consistency/compatibility checks fail.
        """
        raise NotImplementedError

    def _get_tensorized_inputs(
        self,
        timepoints: torch.Tensor,
        individual_parameters: DictParamsTorch,
        *,
        skip_ips_checks: bool = False,
    ) -> tuple[torch.Tensor, DictParamsTorch]:
        from .utilities import tensorize_2D

        if not skip_ips_checks:
            individual_parameters_info = self._audit_individual_parameters(
                individual_parameters
            )
            individual_parameters = individual_parameters_info["tensorized_ips"]
            if (n_individual_parameters := individual_parameters_info["nb_inds"]) != 1:
                raise LeaspyModelInputError(
                    "Only one individual computation may be performed at a time. "
                    f"{n_individual_parameters} was provided."
                )
        # Convert the timepoints (list of numbers, or single number) to a 2D torch tensor
        timepoints = tensorize_2D(timepoints, unsqueeze_dim=0)  # 1 individual
        return timepoints, individual_parameters

    def _check_individual_parameters_provided(
        self, individual_parameters_keys: Iterable[str]
    ) -> None:
        """Check consistency of individual parameters keys provided."""
        ind_vars = set(self.individual_variables_names)
        unknown_ips = set(individual_parameters_keys).difference(ind_vars)
        missing_ips = ind_vars.difference(individual_parameters_keys)
        errs = []
        if len(unknown_ips):
            errs.append(f"Unknown individual latent variables: {unknown_ips}")
        if len(missing_ips):
            errs.append(f"Missing individual latent variables: {missing_ips}")
        if len(errs):
            raise LeaspyIndividualParamsInputError(". ".join(errs))

    def compute_individual_trajectory(
        self,
        timepoints: list[float],
        individual_parameters: DictParams,
        *,
        skip_ips_checks: bool = False,
    ) -> torch.Tensor:
        """Compute scores values at the given time-point(s) given a subject's individual parameters.

        .. note::
            The model uses its current internal state.

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
        for (
            individual_parameter_name,
            individual_parameter_value,
        ) in individual_parameters.items():
            local_state[individual_parameter_name] = individual_parameter_value

        return local_state["model"]

    def compute_prior_trajectory(
        self,
        timepoints: torch.Tensor,
        prior_type: LatentVariableInitType,
        *,
        n_individuals: Optional[int] = None,
    ) -> TensorOrWeightedTensor[float]:
        """
        Compute trajectory of the model for prior mode or mean of individual parameters.

        Parameters
        ----------
        timepoints : :class:`torch.Tensor` [1, n_timepoints]
        prior_type : LatentVariableInitType
        n_individuals : int, optional
            The number of individuals.

        Returns
        -------
        :class:`torch.Tensor` [1, n_timepoints, dimension]
            The group-average values at given timepoints.
        """
        exc_n_ind_iff_prior_samples = LeaspyModelInputError(
            "You should provide n_individuals (int >= 1) if, "
            "and only if, prior_type is `PRIOR_SAMPLES`"
        )
        if n_individuals is None:
            if prior_type is LatentVariableInitType.PRIOR_SAMPLES:
                raise exc_n_ind_iff_prior_samples
            n_individuals = 1
        elif prior_type is not LatentVariableInitType.PRIOR_SAMPLES or not (
            isinstance(n_individuals, int) and n_individuals >= 1
        ):
            raise exc_n_ind_iff_prior_samples
        local_state = self.state.clone(disable_auto_fork=True)
        self._put_data_timepoints(local_state, timepoints)
        local_state.put_individual_latent_variables(
            prior_type, n_individuals=n_individuals
        )

        return local_state["model"]

    def compute_mean_traj(self, timepoints: torch.Tensor):
        """Trajectory for average of individual parameters (not really meaningful for non-linear models)."""
        # TODO/WIP: keep this in BaseModel interface? or only provide `compute_prior_trajectory`, or `compute_mode|typical_traj` instead?
        return self.compute_prior_trajectory(
            timepoints, LatentVariableInitType.PRIOR_MEAN
        )

    def compute_mode_traj(self, timepoints: torch.Tensor):
        """Most typical individual trajectory."""
        return self.compute_prior_trajectory(
            timepoints, LatentVariableInitType.PRIOR_MODE
        )

    def compute_jacobian_tensorized(
        self,
        state: State,
    ) -> DictParamsTorch:
        """
        Compute the jacobian of the model w.r.t. each individual parameter, given the input state.

        This function aims to be used in :class:`.ScipyMinimize` to speed up optimization.

        Parameters
        ----------
        state : :class:`.State`
            Instance holding values for all model variables (including latent individual variables), as well as:
            - timepoints : :class:`torch.Tensor` of shape (n_individuals, n_timepoints)

        Returns
        -------
        :obj:`dict` [ param_name: :obj:`str`, :class:`torch.Tensor` ] :
            Tensors are of shape ``(n_individuals, n_timepoints, n_features, n_dims_param)``.
        """
        # return {
        #     ip: state[f"model_jacobian_{ip}"]
        #     for ip in self.get_individual_variable_names()
        # }
        raise NotImplementedError("This method is currently not implemented.")

    @classmethod
    def compute_sufficient_statistics(cls, state: State) -> SuffStatsRW:
        """
        Compute sufficient statistics from state.

        Parameters
        ----------
        state : :class:`.State`

        Returns
        -------
        :obj:`dict` [ suff_stat: :obj:`str`, :class:`torch.Tensor`]
        """
        suff_stats = {}
        for mp_var in state.dag.sorted_variables_by_type[ModelParameter].values():
            mp_var: ModelParameter  # type-hint only
            suff_stats.update(mp_var.suff_stats(state))

        # we add some fake sufficient statistics that are in fact convergence metrics (summed over individuals)
        # TODO proper handling of metrics
        # We do not account for regularization of pop. vars since we do NOT have true Bayesian priors on them (for now)
        for k in ("nll_attach", "nll_regul_ind_sum"):
            suff_stats[k] = state[k]
        suff_stats["nll_tot"] = (
            suff_stats["nll_attach"] + suff_stats["nll_regul_ind_sum"]
        )  # "nll_regul_all_sum"

        return suff_stats

    @classmethod
    def update_parameters(
        cls,
        state: State,
        sufficient_statistics: SuffStatsRO,
        *,
        burn_in: bool,
    ) -> None:
        """
        Update model parameters of the provided state.

        Parameters
        ----------
        state : :class:`.State`
        sufficient_statistics : dict[suff_stat: str, :class:`torch.Tensor`]
        burn_in : bool
        """
        # <!> we should wait before updating state since some updating rules may depending on OLD state
        # (i.e. no sequential update of state but batched updates once all updated values were retrieved)
        # (+ it would be inefficient since we could recompute some derived values between updates!)
        params_updates = {}
        for mp_name, mp_var in state.dag.sorted_variables_by_type[
            ModelParameter
        ].items():
            mp_var: ModelParameter  # type-hint only
            params_updates[mp_name] = mp_var.compute_update(
                state=state, suff_stats=sufficient_statistics, burn_in=burn_in
            )
        # mass update at end
        for mp, mp_updated_val in params_updates.items():
            state[mp] = mp_updated_val

    def get_variables_specs(self) -> NamedVariables:
        specifications = NamedVariables({"t": DataVariable()})
        single_obs_model = len(self.obs_models) == 1
        for obs_model in self.obs_models:
            specifications.update(
                obs_model.get_variables_specs(named_attach_vars=not single_obs_model)
            )
        return specifications

    @abstractmethod
    def put_individual_parameters(self, state: State, dataset: Dataset):
        raise NotImplementedError()

    def _put_data_timepoints(
        self, state: State, timepoints: TensorOrWeightedTensor[float]
    ) -> None:
        """Put the timepoints variables inside the provided state (in-place)."""
        # TODO/WIP: we use a regular tensor with 0 for times so that 'model' is a regular tensor
        # (to avoid having to cope with `StatelessDistributionFamily` having some `WeightedTensor` as parameters)
        if isinstance(timepoints, WeightedTensor):
            state["t"] = timepoints
        elif isinstance(timepoints, torch.Tensor):
            state["t"] = WeightedTensor(timepoints)
        else:
            raise TypeError(
                f"Time points should be either torch Tensors or WeightedTensors. "
                f"Instead, a {type(timepoints)} was provided."
            )

    def put_data_variables(self, state: State, dataset: Dataset) -> None:
        """Put all the needed data variables inside the provided state (in-place)."""
        self._put_data_timepoints(
            state,
            WeightedTensor(
                dataset.timepoints, dataset.mask.to(torch.bool).any(dim=LVL_FT)
            ),
        )
        for obs_model in self.obs_models:
            state[obs_model.name] = obs_model.getter(dataset)

    def reset_data_variables(self, state: State) -> None:
        """Reset all data variables inside the provided state (in-place)."""
        state["t"] = None
        for obs_model in self.obs_models:
            state[obs_model.name] = None
