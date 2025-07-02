import warnings
from abc import abstractmethod
from typing import Iterable, Optional

import torch

from leaspy.exceptions import LeaspyModelInputError
from leaspy.io.data.dataset import Dataset
from leaspy.utils.typing import DictParamsTorch, KwargsType
from leaspy.utils.weighted_tensor import WeightedTensor
from leaspy.variables.dag import VariablesDAG
from leaspy.variables.specs import (
    Hyperparameter,
    IndividualLatentVariable,
    LatentVariableInitType,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
    VariableName,
    VariableNameToValueMapping,
)
from leaspy.variables.state import State, StateForkType

from .base import BaseModel

__all__ = ["StatefulModel"]


class StatefulModel(BaseModel):
    """Stateful models have an internal :class:`~leaspy.variables.State` to handle parameters and variables.

    Parameters
    ----------

    name : :obj:`str`
        The name of the model.

    Attributes
    ----------
    state : :class:`~leaspy.variables.State`
        The internal state of the model, which contains the variables and their values.
    tracked_variables : :obj:`set` [:obj:`str`]
        Set of variable names that are tracked by the model. These variables are not necessarily part of
        the model's state but are monitored for changes or updates. This can include variables that are
        relevant for the model's operation but not directly stored in the state.

    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._state: Optional[State] = None
        self.tracked_variables: set[str] = set()

    def track_variable(self, variable: VariableName) -> None:
        """Track a variable by its name.

        Parameters
        -------
        variable : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable to track. This variable will be monitored for changes or updates.
        """

        self.tracked_variables.add(variable)

    def track_variables(self, variables: Iterable[VariableName]) -> None:
        """
        Track multiple variables by their names.

        Parameters
        ----------
        variables : :obj:`Iterable` [:class:`~leaspy.variables.specs.VariableName`]
            An iterable containing the names of the variables to track. Each variable will be monitored
            for changes or updates.

        """
        for variable in variables:
            self.track_variable(variable)

    def untrack_variable(self, variable: VariableName) -> None:
        """Untrack a variable by its name.

        Parameters
        -------
        variable : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable to untrack. This variable will no longer be monitored for changes or updates.
        """
        self.tracked_variables.remove(variable)

    def untrack_variables(self, variables: Iterable[VariableName]) -> None:
        """Untrack multiple variables by their names.

        Parameters
        ----------
        variables : :obj:`Iterable` [:class:`~leaspy.variables.specs.VariableName`]
            An iterable containing the names of the variables to untrack. Each variable will no longer be monitored
            for changes or updates.
        """
        for variable in variables:
            self.untrack_variable(variable)

    @property
    def state(self) -> State:
        """Get the internal state of the model.

        Returns
        -------
        State : :class:`~leaspy.variables.State`
            The internal state of the model, which contains the variables and their values.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If the model's state is not initialized yet.
        """
        if self._state is None:
            raise LeaspyModelInputError("Model state is not initialized yet")
        return self._state

    @state.setter
    def state(self, s: State) -> None:
        """Set the internal state of the model.
        This method allows to set the internal state of the model, which is an instance of

        Parameters
        ----------
        s : :class:`~leaspy.variables.State`
            The new state to set for the model.

        Raises
        ------
        LeaspyModelInputError
            If the provided state does not match the previous state in terms of DAG structure.
        """
        assert isinstance(s, State), "Provided state should be a valid State instance"
        if self._state is not None and s.dag is not self._state.dag:
            raise LeaspyModelInputError(
                "DAG of new state does not match with previous one"
            )
        # TODO? perform some clean-up steps for provided state (cf. `_terminate_algo` of MCMC algo)
        self._state = s

    @property
    def dag(self) -> VariablesDAG:
        """Get the underlying DAG of the model's state.

        Returns
        -------
        : :class:`~leaspy.variables.dag.VariablesDAG`
            The directed acyclic graph (DAG) representing the model's variables and their relationships

        """
        return self.state.dag

    @property
    def hyperparameters_names(self) -> tuple[VariableName, ...]:
        """Get the names of the model's hyperparameters.

        Returns
        -------
        : :obj:`tuple` [:class:`~leaspy.variables.specs.VariableName`, others...]
            A tuple containing the names of the model's hyperparameters.
        """
        return tuple(self.dag.sorted_variables_by_type[Hyperparameter])

    @property
    def parameters_names(self) -> tuple[VariableName, ...]:
        """Get the names of the model's parameters.

        Returns
        -------
        : :obj:`tuple` [:class:`~leaspy.variables.specs.VariableName`, others...]
            A tuple containing the names of the model's parameters.
        """
        return tuple(self.dag.sorted_variables_by_type[ModelParameter])

    @property
    def population_variables_names(self) -> tuple[VariableName, ...]:
        """Get the names of the population latent variables.

        Returns
        -------
        : :obj:`tuple` [:class:`~leaspy.variables.specs.VariableName`, ...]
            A tuple containing the names of the population latent variables.
        """
        return tuple(self.dag.sorted_variables_by_type[PopulationLatentVariable])

    @property
    def individual_variables_names(self) -> tuple[VariableName, ...]:
        """Get the names of the individual latent variables.

        Returns
        -------
        : :obj:`tuple` [:class:`~leaspy.variables.specs.VariableName`, ...]
            A tuple containing the names of the individual latent variables.
        """
        return tuple(self.dag.sorted_variables_by_type[IndividualLatentVariable])

    @property
    def parameters(self) -> DictParamsTorch:
        """Dictionary of values for model parameters.

        Returns
        -------
        : :class:`~leaspy.utils.typing.DictParamsTorch`
            A dictionary mapping parameter names to their values (as tensors).
        """
        return {p: self._state[p] for p in self.parameters_names}

    @property
    def hyperparameters(self) -> DictParamsTorch:
        """Dictionary of values for model hyperparameters.

        Returns
        -------
        : :class:`~leaspy.utils.typing.DictParamsTorch`
            A dictionary mapping hyperparameter names to their values (as tensors).
        """
        return {p: self._state[p] for p in self.hyperparameters_names}

    def initialize(self, dataset: Optional[Dataset] = None) -> None:
        """Overloads base model initialization (in particular to handle internal model State).

        <!> We do not put data variables in internal model state at this stage (done in algorithm)

        Parameters
        ----------
        dataset : :class:`~leaspy.io.data.dataset.Dataset`, optional
            Input dataset from which to initialize the model.
        """
        super().initialize(dataset=dataset)
        self._initialize_state()
        if not dataset:
            return
        # WIP: design of this may be better somehow?
        with self._state.auto_fork(None):
            self._initialize_model_parameters(dataset)
            self._state.put_population_latent_variables(
                LatentVariableInitType.PRIOR_MODE
            )

    def _initialize_state(self) -> None:
        """Initialize the internal state of model, as well as the underlying DAG.

        Note that all model hyperparameters (dimension, source_dimension, ...) should be defined
        in order to be able to do so.
        """
        if self._state is not None:
            raise LeaspyModelInputError("Trying to initialize the model's state again")
        self.state = State(
            VariablesDAG.from_dict(self.get_variables_specs()),
            auto_fork_type=StateForkType.REF,
        )
        self.state.track_variables(self.tracked_variables)

    def _initialize_model_parameters(self, dataset: Dataset) -> None:
        """Initialize model parameters (in-place, in `_state`).

        The method also checks that the model parameters whose initial values
        were computed from the dataset match the expected model parameters from
        the specifications (i.e. the nodes of the DAG of type 'ModelParameter').

        If there is a mismatch, the method raises a ValueError because there is
        an inconsistency between the definition of the model and the way it computes
        the initial values of its parameters from a dataset.

        Parameters
        ----------
        dataset : :class:`~leaspy.io.data.dataset.Dataset`
            The dataset to use to compute initial values for the model parameters.
        """
        model_parameters_initialization = (
            self._compute_initial_values_for_model_parameters(dataset)
        )
        model_parameters_spec = self.dag.sorted_variables_by_type[ModelParameter]
        if set(model_parameters_initialization.keys()) != set(model_parameters_spec):
            raise ValueError(
                "Model parameters created at initialization are different "
                "from the expected model parameters from the specs:\n"
                f"- From initialization: {sorted(list(model_parameters_initialization.keys()))}\n"
                f"- From Specs: {sorted(list(model_parameters_spec))}\n"
            )
        for (
            model_parameter_name,
            model_parameter_variable,
        ) in model_parameters_spec.items():
            model_parameter_initial_value = model_parameters_initialization[
                model_parameter_name
            ]
            if not isinstance(
                model_parameter_initial_value, (torch.Tensor, WeightedTensor)
            ):
                try:
                    model_parameter_initial_value = torch.tensor(
                        model_parameter_initial_value, dtype=torch.float
                    )
                except ValueError:
                    raise ValueError(
                        f"The initial value for model parameter '{model_parameter_name}' "
                        "should be a tensor, or a weighted tensor.\nInstead, "
                        f"{model_parameter_initial_value} of type {type(model_parameter_initial_value)} "
                        "was received and cannot be casted to a tensor.\nPlease verify this parameter "
                        "initialization code."
                    )
            self._state[model_parameter_name] = model_parameter_initial_value.expand(
                model_parameter_variable.shape
            )

    def load_parameters(self, parameters: KwargsType) -> None:
        """Instantiate or update the model's parameters.

        It assumes that all model hyperparameters are defined.

        Parameters
        ----------
        parameters : :class:`~leaspy.utils.typing.KwargsType`]
            Contains the model's parameters.
        """
        from .utilities import val_to_tensor

        if self._state is None:
            self._initialize_state()

        # TODO: a bit dirty due to hyperparams / params mix (cf. `.parameters` property note)

        params_names = self.parameters_names
        missing_params = set(params_names).difference(parameters)
        if len(missing_params):
            warnings.warn(f"Missing some model parameters: {missing_params}")
        extra_vars = set(parameters).difference(self.dag)
        if len(extra_vars):
            raise LeaspyModelInputError(f"Unknown model variables: {extra_vars}")
        # TODO: check no DataVariable provided???
        # extra_params = set(parameters).difference(cur_params)
        # if len(extra_params):
        #    # e.g. mixing matrix, which is a derived variable - checking their values only
        #    warnings.warn(f"Ignoring some provided values that are not model parameters: {extra_params}")
        # update parameters first (to be able to check values of derived variables afterwards)
        provided_params = {
            p: val_to_tensor(parameters[p], self.dag[p].shape)
            for p in params_names
            if p in parameters
        }
        for p, val in provided_params.items():
            # TODO: WeightedTensor? (e.g. batched `deltas`)
            self._state[p] = val

        # derive the population latent variables from model parameters
        # e.g. to check value of `mixing_matrix` we need `v0` and `betas` (not just `log_v0` and `betas_mean`)
        self._state.put_population_latent_variables(LatentVariableInitType.PRIOR_MODE)

        # check equality of other values (hyperparameters or linked variables)
        for parameter_name, parameter_value in parameters.items():
            if parameter_name in provided_params:
                continue
            # TODO: a bit dirty due to hyperparams / params mix (cf. `.parameters` property note)
            try:
                current_value = self._state[parameter_name]
            except Exception as e:
                raise LeaspyModelInputError(
                    f"Impossible to compare value of provided value for {parameter_name} "
                    "- not computable given current state"
                ) from e
            parameter_value = val_to_tensor(
                parameter_value, getattr(self.dag[parameter_name], "shape", None)
            )
            assert (
                parameter_value.shape == current_value.shape,
                (parameter_name, parameter_value.shape, current_value.shape),
            )
            # TODO: WeightedTensor? (e.g. batched `deltas``)
            assert (
                torch.allclose(parameter_value, current_value, atol=1e-4),
                (parameter_name, parameter_value, current_value),
            )

    @abstractmethod
    def get_variables_specs(self) -> NamedVariables:
        """Return the specifications of the variables (latent variables,
        derived variables, model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables : :class:`~leaspy.variables.specs.NamedVariables`
            The specifications of the model's variables.
        """
        raise NotImplementedError()

    @abstractmethod
    def _compute_initial_values_for_model_parameters(
        self, dataset: Dataset
    ) -> VariableNameToValueMapping:
        """Compute initial values for model parameters.

        Parameters
        ----------
        dataset : :class:`~leaspy.io.data.dataset.Dataset`
            The dataset to use to compute initial values for the model parameters.

        Returns
        -------
        : :class:`~leaspy.utils.typing.Any`
            A dictionary mapping variable names to their initial values.
        """
        raise NotImplementedError()

    def move_to_device(self, device: torch.device) -> None:
        """Move a model and its relevant attributes to the specified :class:`torch.device`.

        Parameters
        ----------
        device : :obj:`torch.device`
            The device to which the model and its attributes should be moved.
        """
        if self._state is None:
            return
        self._state.to_device(device)
        for hp in self.hyperparameters_names:
            self._state.dag[hp].to_device(device)
