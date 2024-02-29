from __future__ import annotations

from abc import abstractmethod
import warnings

import torch

from leaspy.models.base import BaseModel, InitializationMethod
from leaspy.models.obs_models import ObservationModel, ObservationModelFactoryInput, observation_model_factory
from .utilities import cast_value_to_tensor, cast_value_to_2d_tensor, serialize_tensor
from leaspy.io.data.dataset import Dataset

from leaspy.variables.specs import (
    VarName,
    Hyperparameter,
    ModelParameter,
    LinkedVariable,
    DataVariable,
    PopulationLatentVariable,
    IndividualLatentVariable,
    LatentVariableInitType,
    NamedVariables,
    SuffStatsRO,
    SuffStatsRW,
    LVL_FT,
    VariablesValuesRO,
)
from leaspy.variables.dag import VariablesDAG
from leaspy.variables.state import State, StateForkType
from leaspy.utils.weighted_tensor import WeightedTensor, TensorOrWeightedTensor

from leaspy.exceptions import LeaspyIndividualParamsInputError, LeaspyModelInputError
from leaspy.utils.typing import (
    FeatureType,
    KwargsType,
    DictParams,
    DictParamsTorch,
    Union,
    List,
    Dict,
    Set,
    Tuple,
    Iterable,
    Optional,
    Any,
)


#  TODO? refact so to only contain methods needed for the Leaspy api + add another
#  abstract class (interface) on top of it for MCMC fittable models + one for "manifold models"

# TODO: not 100% clear to me whether:
# 1. model should have an internal state? or only provide methods to define suited states (i.e. with the right DAG) and interact with such states
# 2. model methods should have a `state: State` argument, or the state used is always the model internal one?


class AbstractModel(BaseModel):
    """
    Contains the common attributes & methods of the different probabilistic models.

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
    obs_models : Tuple[ObservationModelFactoryInput, ...]
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
        obs_models: Optional[Union[ObservationModelFactoryInput, Iterable[ObservationModelFactoryInput]]] = None,
        dimension: Optional[int] = None,
        features: Optional[List[FeatureType]] = None,
        fit_metrics: Optional[Dict[str, float]] = None,
        variables_to_track: Optional[Iterable[VarName]] = None,
        **kwargs
    ):
        super().__init__(name, dimension=dimension, features=features)
        if obs_models is None:
            obs_models = "gaussian-scalar" if dimension is None else "gaussian-diagonal"
        if isinstance(obs_models, (list, tuple)):
            self.obs_models = tuple(
                [observation_model_factory(model, dimension=dimension, **kwargs) for model in obs_models]
            )
        elif isinstance(obs_models, dict):
            # Not really satisfied... Used for api load
            self.obs_models = tuple(
                [observation_model_factory(obs_models['y'], dimension=dimension)]
            )
        else:
            self.obs_models = (observation_model_factory(obs_models, dimension=dimension, **kwargs),)
        # WIP: cf. comment regarding inclusion of state here
        self._state: Optional[State] = None  # = state
        # TODO: dirty hack for now, cf. AbstractFitAlgo
        self.fit_metrics = fit_metrics
        self.tracked_variables: Set[VarName, ...] = set()
        if variables_to_track:
            self.tracked_variables = self.tracked_variables.union(set(variables_to_track))

    @property
    def state(self) -> State:
        if self._state is None:
            raise LeaspyModelInputError("Model state is not initialized yet")
        return self._state

    @state.setter
    def state(self, new_state: State) -> None:
        if not isinstance(new_state, State):
            raise LeaspyModelInputError("Provided state should be a valid State instance.")
        if self._state is not None and new_state.dag is not self._state.dag:
            raise LeaspyModelInputError("DAG of new state does not match with previous one.")
        # TODO? perform some clean-up steps for provided state (cf. `_terminate_algo` of MCMC algo)
        self._state = new_state

    @property
    def dag(self) -> VariablesDAG:
        return self.state.dag

    @property
    def hyperparameters_names(self) -> Tuple[VarName, ...]:
        return tuple(self.dag.sorted_variables_by_type[Hyperparameter])

    @property
    def parameters_names(self) -> Tuple[VarName, ...]:
        return tuple(self.dag.sorted_variables_by_type[ModelParameter])

    @property
    def population_variables_names(self) -> Tuple[VarName, ...]:
        return tuple(self.dag.sorted_variables_by_type[PopulationLatentVariable])

    @property
    def individual_variables_names(self) -> Tuple[VarName, ...]:
        return tuple(self.dag.sorted_variables_by_type[IndividualLatentVariable])

    @property
    def parameters(self) -> DictParamsTorch:
        """Dictionary of values for model parameters."""
        return {p: self._state[p] for p in self.parameters_names}

    @property
    def hyperparameters(self) -> DictParamsTorch:
        """Dictionary of values for model hyperparameters."""
        return {p: self._state[p] for p in self.hyperparameters_names if p in self._state}

    @property
    def has_sources(self) -> bool:
        return (
            hasattr(self, 'source_dimension')
            and isinstance(self.source_dimension, int)
            and self.source_dimension > 0
        )

    def to_dict(self, **kwargs) -> KwargsType:
        """
        Export model as a dictionary ready for export.

        Returns_tensorize_2D
        -------
        KwargsType :
            The model instance serialized as a dictionary.
        """
        model_export = super().to_dict(**kwargs)
        return {
            **model_export,
            **{
                "obs_models": {
                    obs_model.name: obs_model.to_string()
                    for obs_model in self.obs_models
                },
                # 'obs_models': export_obs_models(self.obs_models),
                "fit_metrics": self.fit_metrics,  # TODO improve
                "tracked_variables": list(self.tracked_variables),
            }
        }

    def load_parameters(self, parameters: KwargsType) -> None:
        if self._state is None:
            self._initialize_state()
        if len(missing_parameters := set(self.parameters_names).difference(set(parameters.keys()))):
            warnings.warn(f"Missing some model parameters: {missing_parameters}")
        super().load_parameters(parameters)
        # derive the population latent variables from model parameters
        # e.g. to check value of `mixing_matrix` we need `v0` and `betas` (not just `log_v0` and `betas_mean`)
        self._state.put_population_latent_variables(LatentVariableInitType.PRIOR_MODE)
        self._check_parameter_consistency()

    def _check_parameter_consistency(self):
        for parameter_name, parameter_value in self.parameters.items():
            try:
                current_value = self._state[parameter_name]
            except Exception as e:
                raise LeaspyModelInputError(
                    f"Impossible to compare value of provided value for {parameter_name} "
                    "- not computable given current state"
                ) from e
            parameter_value = cast_value_to_tensor(
                parameter_value, getattr(self.dag[parameter_name], "shape", None)
            )
            assert (
                parameter_value.shape == current_value.shape,
                (parameter_name, parameter_value.shape, current_value.shape)
            )
            # TODO: WeightedTensor? (e.g. batched `deltas``)
            assert (
                torch.allclose(parameter_value, current_value, atol=1e-4),
                (parameter_name, parameter_value, current_value)
            )

    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set parameter name to given value.

        Parameters
        ----------
        name: str
            The name of the parameter to set.

        value : Any
            The value of the parameter.

        Raises
        ------
        LeaspyModelInputError :
            If the name is not a known model parameter.
        """
        if name not in self.parameters_names:
            raise LeaspyModelInputError(f"Unknown model parameter: {name}.")
        self._state[name] = cast_value_to_tensor(value, self.dag[name].shape)

    def _audit_individual_parameters(self, individual_parameters: DictParams) -> KwargsType:
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
        expected_parameters = set(['xi', 'tau'] + int(self.has_sources)*['sources'])
        given_parameters = set(individual_parameters.keys())
        symmetric_diff = expected_parameters.symmetric_difference(given_parameters)
        if len(symmetric_diff) > 0:
            raise LeaspyIndividualParamsInputError(
                    f"Individual parameters dict provided {given_parameters} "
                    f"is not compatible for {self.name} model. "
                    f"The expected individual parameters are {expected_parameters}."
            )
        # Check number of individuals present (with low constraints on shapes)
        individual_parameters_is_array_like = {
            k: _is_array_like(v) for k, v in individual_parameters.items()
        }
        individual_parameters_size = {
            k: len(v)
            if individual_parameters_is_array_like[k] else 1
            for k, v in individual_parameters.items()
        }
        if self.has_sources:
            if not individual_parameters_is_array_like['sources']:
                raise LeaspyIndividualParamsInputError(
                    f"Sources must be an array_like but {individual_parameters['sources']} was provided."
                )
            tau_xi_scalars = all(individual_parameters_size[k] == 1 for k in ("tau", "xi"))
            if tau_xi_scalars and (individual_parameters_size["sources"] > 1):
                # is 'sources' not a nested array? (allowed iff tau & xi are scalars)
                if not _is_array_like(individual_parameters["sources"][0]):
                    # then update sources size (1D vector representing only 1 individual)
                    individual_parameters_size["sources"] = 1

            # TODO? check source dimension compatibility?
        if len(unique_sizes := set(individual_parameters_size.values())) != 1:
            raise LeaspyIndividualParamsInputError(
                "Individual parameters sizes are not compatible together. "
                f"Sizes are {individual_parameters_size}."
            )
        # properly choose unsqueezing dimension when tensorizing array_like (useful for sources)
        unsqueeze_dim = -1  # [1,2] => [[1],[2]] (expected for 2 individuals / 1D sources)
        if (n_individuals := unique_sizes.pop()) == 1:
            unsqueeze_dim = 0  # [1,2] => [[1,2]] (expected for 1 individual / 2D sources)
        # tensorized (2D) version of ips
        tensorized_individual_parameters = {
            k: cast_value_to_2d_tensor(v, unsqueeze_dim=unsqueeze_dim)
            for k, v in individual_parameters.items()
        }
        return {
            'nb_inds': n_individuals,
            'tensorized_ips': tensorized_individual_parameters,
            'tensorized_ips_gen': (
                {
                    individual_name: individual_value[individual_idx, :].unsqueeze(0)
                    for individual_name, individual_value in tensorized_individual_parameters.items()
                }
                for individual_idx in range(n_individuals)
            ),
        }

    def _get_tensorized_inputs(
        self,
        timepoints: torch.Tensor,
        individual_parameters: DictParamsTorch,
        *,
        skip_individual_parameters_checks: bool = False,
    ) -> Tuple[torch.Tensor, DictParamsTorch]:
        if not skip_individual_parameters_checks:
            individual_parameters_info = self._audit_individual_parameters(individual_parameters)
            individual_parameters = individual_parameters_info["tensorized_ips"]
            if (n := individual_parameters_info["nb_inds"]) != 1:
                raise LeaspyModelInputError(
                    f"Only one individual computation may be performed at a time. {n} was provided."
                )
        # Convert the timepoints (list of numbers, or single number) to a 2D torch tensor
        timepoints = cast_value_to_2d_tensor(timepoints, unsqueeze_dim=0)  # 1 individual
        return timepoints, individual_parameters

    def _check_individual_parameters_provided(self, individual_parameters_keys: Iterable[str]) -> None:
        """Check consistency of individual parameters keys provided."""
        individual_variable_names = set(self.individual_variables_names)
        errors = []
        if len(unknown_individual_parameters := set(individual_parameters_keys).difference(individual_variable_names)):
            errors.append(f"Unknown individual latent variables: {unknown_individual_parameters}.")
        if len(missing_individual_parameters := individual_variable_names.difference(individual_parameters_keys)):
            errors.append(f"Missing individual latent variables: {missing_individual_parameters}.")
        if errors:
            raise LeaspyIndividualParamsInputError(". ".join(errors))

    # TODO: unit tests? (functional tests covered by api.estimate)
    def compute_individual_trajectory(
        self,
        timepoints,
        individual_parameters: DictParams,
        *,
        skip_individual_parameters_checks: bool = False,
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
        skip_individual_parameters_checks : :obj:`bool` (default: ``False``)
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
            timepoints,
            individual_parameters,
            skip_individual_parameters_checks=skip_individual_parameters_checks,
        )
        # TODO? ability to revert back after **several** assignments?
        # instead of cloning the state for this op?
        local_state = self.state.clone(disable_auto_fork=True)
        self._put_data_timepoints(local_state, timepoints)
        for individual_parameter_name, individual_parameter_value in individual_parameters.items():
            local_state[individual_parameter_name] = individual_parameter_value

        return local_state["model"]

    def _compute_prior_trajectory(
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
        elif (
            prior_type is not LatentVariableInitType.PRIOR_SAMPLES
            or not (isinstance(n_individuals, int) and n_individuals >= 1)
        ):
            raise exc_n_ind_iff_prior_samples
        local_state = self.state.clone(disable_auto_fork=True)
        self._put_data_timepoints(local_state, timepoints)
        local_state.put_individual_latent_variables(prior_type, n_individuals=n_individuals)

        return local_state["model"]

    def compute_mean_trajectory(self, timepoints: torch.Tensor) -> TensorOrWeightedTensor[float]:
        """Trajectory for average of individual parameters (not really meaningful for non-linear models)."""
        # TODO/WIP: keep this in BaseModel interface? or only
        #  provide `compute_prior_trajectory`, or `compute_mode|typical_traj` instead?
        return self._compute_prior_trajectory(timepoints, LatentVariableInitType.PRIOR_MEAN)

    def compute_mode_trajectory(self, timepoints: torch.Tensor) -> TensorOrWeightedTensor[float]:
        """Most typical individual trajectory."""
        return self._compute_prior_trajectory(timepoints, LatentVariableInitType.PRIOR_MODE)

    # TODO: unit tests? (functional tests covered by api.estimate)
    def compute_individual_ages_from_biomarker_values(
        self,
        value: Union[float, List[float]],
        individual_parameters: DictParams,
        feature: Optional[FeatureType] = None,
    ) -> torch.Tensor:
        """
        For one individual, compute age(s) at which the given features values
        are reached (given the subject's individual parameters).

        Consistency checks are done in the main :term:`API` layer.

        Parameters
        ----------
        value : scalar or array_like[scalar] (:obj:`list`, :obj:`tuple`, :class:`numpy.ndarray`)
            Contains the :term:`biomarker` value(s) of the subject.

        individual_parameters : :obj:`dict`
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like.

        feature : :obj:`str` (or None)
            Name of the considered :term:`biomarker`.

            .. note::
                Optional for :class:`.UnivariateModel`, compulsory
                for :class:`.MultivariateModel`.

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s).
            Shape of tensor is ``(1, n_values)``.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If computation is tried on more than 1 individual.
        """
        raise NotImplementedError("TODO")
        value, individual_parameters = self._get_tensorized_inputs(
            value,
            individual_parameters,
            skip_individual_parameters_checks=False,
        )
        return self.compute_individual_ages_from_biomarker_values_tensorized(
            value, individual_parameters, feature
        )

    #@abstractmethod
    def compute_individual_ages_from_biomarker_values_tensorized(
        self,
        value: torch.Tensor,
        individual_parameters: DictParamsTorch,
        feature: Optional[FeatureType],
    ) -> torch.Tensor:
        """
        For one individual, compute age(s) at which the given features values are
        reached (given the subject's individual parameters), with tensorized inputs.

        Parameters
        ----------
        value : :class:`torch.Tensor` of shape ``(1, n_values)``
            Contains the :term:`biomarker` value(s) of the subject.

        individual_parameters : DictParamsTorch
            Contains the individual parameters.
            Each individual parameter should be a :class:`torch.Tensor`.

        feature : :obj:`str` (or None)
            Name of the considered :term:`biomarker`.

            .. note::
                Optional for :class:`.UnivariateModel`, compulsory
                for :class:`.MultivariateModel`.

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s).
            Shape of tensor is ``(n_values, 1)``.
        """
        raise NotImplementedError("TODO in child classes")

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
        raise NotImplementedError("TODO")
        return {
            ip: state[f"model_jacobian_{ip}"]
            for ip in self.get_individual_variable_names()
        }

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
        sufficient_statistics = {}
        for model_parameter in state.dag.sorted_variables_by_type[ModelParameter].values():
            model_parameter: ModelParameter
            sufficient_statistics.update(model_parameter.suff_stats(state))

        # we add some fake sufficient statistics that are in fact convergence metrics (summed over individuals)
        # TODO proper handling of metrics
        # We do not account for regularization of pop. vars since we do NOT have true Bayesian priors on them (for now)
        for statistic in ("nll_attach", "nll_regul_ind_sum"):
            sufficient_statistics[statistic] = state[statistic]
        sufficient_statistics["nll_tot"] = (
                sufficient_statistics["nll_attach"] +
                sufficient_statistics["nll_regul_ind_sum"]  # "nll_regul_all_sum"
        )

        return sufficient_statistics

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
        parameter_updates = {}
        for model_parameter_name, model_parameter_value in state.dag.sorted_variables_by_type[ModelParameter].items():
            model_parameter_value: ModelParameter  # type-hint only
            parameter_updates[model_parameter_name] = model_parameter_value.compute_update(
                state=state,
                suff_stats=sufficient_statistics,
                burn_in=burn_in,
            )
        for model_parameter_name, model_parameter_value in parameter_updates.items():
            state[model_parameter_name] = model_parameter_value

    def __str__(self):
        output = "=== MODEL ==="
        output += serialize_tensor(self.parameters)

        # TODO/WIP obs models...
        # nm_props = export_noise_model(self.noise_model)
        # nm_name = nm_props.pop('name')
        # output += f"\nnoise-model : {nm_name}"
        # output += serialize_tensor(nm_props, indent="  ")

        return output

    @staticmethod
    def time_reparametrization(
        *,
        t: TensorOrWeightedTensor[float],
        alpha: torch.Tensor,
        tau: torch.Tensor,
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
            Time-shift(s).

        Returns
        -------
        :class:`torch.Tensor` of same shape as `timepoints`
        """
        return alpha * (t - tau)

    #@abstractmethod
    #def model(self, **kws) -> torch.Tensor:
    #    pass
    #
    #@abstractmethod
    #def model_jacobian(self, **kws) -> torch.Tensor:
    #    pass

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables,
        derived variables, model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables :
            The specifications of the model's variables.
        """
        variable_specs = NamedVariables({
            "t": DataVariable(),
            "rt": LinkedVariable(self.time_reparametrization),
            #"model": LinkedVariable(self.model),  # function arguments may depends on hyperparameters so postpone (e.g. presence of sources or not)
            #"model_jacobian_{ip}": LinkedVariable(self.model_jacobian), for ip in IndividualLatentVariables....
        })
        for observation_model in self.obs_models:
            variable_specs.update(
                observation_model.get_variables_specs(
                    named_attach_vars=(len(self.obs_models) != 1),
                )
            )
        #if not single_obs_model:
        #    raise NotImplementedError(
        #        "WIP: Only 1 noise model supported for now, but to be extended."
        #    )
        #    d.update(
        #        #nll_attach_full=LinkedVariable(Sum(...)),
        #        nll_attach_ind=LinkedVariable(Sum(...)),
        #        nll_attach=LinkedVariable(Sum(...)),
        #        # TODO Same for nll_attach_ind jacobian, w.r.t each observation var???
        #    )

        return variable_specs

    def initialize(self, dataset: Optional[Dataset] = None, method: Optional[InitializationMethod] = None) -> None:
        """
        Overloads base model initialization (in particular to handle internal model State).

        <!> We do not put data variables in internal model state at this stage (done in algorithm)

        Parameters
        ----------
        dataset : :class:`.Dataset`, optional
            Input dataset from which to initialize the model.
        method : InitializationMethod, optional
            The initialization method to be used.
            Default='default'.
        """
        super().initialize(dataset=dataset, method=method)
        self._initialize_state()
        if not dataset:
            return
        # WIP: design of this may be better somehow?
        with self._state.auto_fork(None):
            # Set model parameters
            self._initialize_model_parameters(dataset, method=method)
            # Initialize population latent variables to their mode
            self._state.put_population_latent_variables(LatentVariableInitType.PRIOR_MODE)

    def _initialize_state(self) -> None:
        """
        Initialize the internal state of model, as well as the underlying DAG.

        Note that all model hyperparameters (dimension, source_dimension, ...) should be defined
        in order to be able to do so.

        Returns
        -------
        None
        """
        if self._state is not None:
            raise LeaspyModelInputError("Trying to initialize the model's state again.")
        self.state = State(
            VariablesDAG.from_dict(self.get_variables_specs()),
            auto_fork_type=StateForkType.REF
        )
        self.state.track_variables(self.tracked_variables)

    def put_individual_parameters(self, state: State, dataset: Dataset):
        if not state.are_variables_set(('xi', 'tau')):
            with state.auto_fork(None):
                state.put_individual_latent_variables(
                    LatentVariableInitType.PRIOR_SAMPLES,
                    n_individuals=dataset.n_individuals,
                )

    def _put_data_timepoints(self, state: State, timepoints: TensorOrWeightedTensor[float]) -> None:
        """Put the timepoints variables inside the provided state (in-place)."""
        # TODO/WIP: we use a regular tensor with 0 for times so that 'model' is a regular tensor
        # (to avoid having to cope with `StatelessDistributionFamily` having some `WeightedTensor` as parameters)
        # (but we might need it at some point, especially for `batched_deltas` of ordinal model for instance)
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
                dataset.timepoints,
                dataset.mask.to(torch.bool).any(dim=LVL_FT)
            ),
        )
        for observation_model in self.obs_models:
            state[observation_model.name] = observation_model.getter(dataset)

    def reset_data_variables(self, state: State) -> None:
        """Reset all data variables inside the provided state (in-place)."""
        state["t"] = None
        for observation_model in self.obs_models:
            state[observation_model.name] = None

    def _initialize_model_parameters(self, dataset: Dataset, method: InitializationMethod) -> None:
        """Initialize model parameters (in-place, in `_state`).

        The method also checks that the model parameters whose initial values
        were computed from the dataset match the expected model parameters from
        the specifications (i.e. the nodes of the DAG of type 'ModelParameter').

        If there is a mismatch, the method raises a ValueError because there is
        an inconsistency between the definition of the model and the way it computes
        the initial values of its parameters from a dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to use to compute initial values for the model parameters.

        method : InitializationMethod
            The initialization method to use to compute these initial values.
        """
        model_parameters_initialization = self._compute_initial_values_for_model_parameters(dataset, method=method)
        model_parameters_spec = self.dag.sorted_variables_by_type[ModelParameter]
        if set(model_parameters_initialization.keys()) != set(model_parameters_spec):
            raise ValueError(
                "Model parameters created at initialization are different "
                "from the expected model parameters from the specs:\n"
                f"- From initialization: {sorted(list(model_parameters_initialization.keys()))}\n"
                f"- From Specs: {sorted(list(model_parameters_spec))}\n"
            )
        for model_parameter_name, model_parameter_variable in model_parameters_spec.items():
            model_parameter_initial_value = model_parameters_initialization[model_parameter_name]
            if not isinstance(model_parameter_initial_value, (torch.Tensor, WeightedTensor)):
                try:
                    model_parameter_initial_value = torch.tensor(model_parameter_initial_value, dtype=torch.float)
                except ValueError:
                    raise ValueError(
                        f"The initial value for model parameter '{model_parameter_name}' "
                        "should be a tensor, or a weighted tensor.\nInstead, "
                        f"{model_parameter_initial_value} of type {type(model_parameter_initial_value)} "
                        "was received and cannot be casted to a tensor.\nPlease verify this parameter "
                        "initialization code."
                    )
            self.set_parameter(
                model_parameter_name,
                model_parameter_initial_value.expand(model_parameter_variable.shape),
            )

    @abstractmethod
    def _compute_initial_values_for_model_parameters(
        self,
        dataset: Dataset,
        method: InitializationMethod,
    ) -> VariablesValuesRO:
        """Compute initial values for model parameters."""
        raise NotImplementedError

    def move_to_device(self, device: torch.device) -> None:
        """
        Move a model and its relevant attributes to the specified :class:`torch.device`.

        Parameters
        ----------
        device : :class:`torch.device`
        """
        if self._state is None:
            return
        self._state.to_device(device)
        for hyperparameter_name in self.hyperparameters_names:
            if hyperparameter_name in self._state.dag:
                self._state.dag[hyperparameter_name].to_device(device)


def _is_array_like(v) -> bool:
    # abc.Collection is useless here because set, np.array(scalar) or torch.tensor(scalar)
    # are abc.Collection but are not array_like in numpy/torch sense or have no len()
    try:
        len(v)  # exclude np.array(scalar) or torch.tensor(scalar)
        return hasattr(v, '__getitem__')  # exclude set
    except Exception:
        return False
