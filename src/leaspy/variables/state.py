"""This module defines the State of stateful models.

A state contains 2 main components:
1. The relationships between the variables through a :class:`~leaspy.variables.dag.VariablesDAG`
2. The values of each variable of the DAG as a mapping between variable names and their values

The State class is crucial for stateful models with its logic for efficiently retrieving variable values. This class relies on a caching mechanism that enables quick queries.
"""

from __future__ import annotations

import copy
import csv
from collections.abc import MutableMapping
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import torch

from leaspy.exceptions import LeaspyInputError
from leaspy.utils.weighted_tensor import WeightedTensor, unsqueeze_right

from .dag import VariablesDAG
from .specs import (
    Hyperparameter,
    IndividualLatentVariable,
    LatentVariableInitType,
    PopulationLatentVariable,
    VariableName,
    VariablesLazyValuesRO,
    VariablesLazyValuesRW,
    VariableValue,
)

__all__ = [
    "State",
    "StateForkType",
]


class StateForkType(Enum):
    """
    The strategy used to cache forked values in :class:`.State`.

    REF : Reference-based caching
        Cached values are stored by reference, meaning no copying occurs. Mutating the original
        variables after caching will affect the cached version.

    COPY : Deep copy-based caching
        Cached values are stored via `copy.deepcopy`, ensuring they are independent of the originals.

    Notes
    -----
    - Use `REF` for efficiency when you're certain the original values will not be mutated after caching.
    - Use `COPY` to ensure isolation between the cached values and any subsequent modifications.
    - If using `REF` beware that values will NOT be copied (it only keeps references of values),
      so do NOT mutate them directly or the behavior will be unexpected.
    """

    REF = auto()
    COPY = auto()

    def to_cache(self, d: VariablesLazyValuesRW) -> VariablesLazyValuesRO:
        """Get the values to cache, depending on forking type."""
        if self is self.REF:
            return d
        return {k: copy.deepcopy(v) for k, v in d.items()}


class State(MutableMapping):
    """
    Dictionary of cached values corresponding to the stateless DAG instance.

    Parameters
    ----------
    dag : :class:`~leaspy.variables.dag.VariablesDAG`
        The stateless DAG which state will hold values for.

    auto_fork_type : :class:`~leaspy.variables.state.StateForkType` or None (default)
        Refer to :class:`~leaspy.variables.state.StateForkType` class and :attr:`auto_fork_type`

    Attributes
    ----------
    dag : :class:`~leaspy.variables.dag.VariablesDAG`
        The stateless DAG which the state instance will hold values for.

    auto_fork_type : :class:`~leaspy.variables.state.StateForkType` or None
        If not `StateForkType.NONE` each dictionary assignment will lead to the partial caching
        of previous value and all its children, so they can be reverted without computation.
        The exact caching strategy depends on flag (caching by reference or by copy)
        Can be manually set or via `auto_fork` context manager.

    _tracked_variables : :obj:`set`[:class:`~leaspy.variables.specs.VariableName`, ...]

    _values : :class:`~leaspy.variables.specs.VariablesLazyValuesRW`
        Private cache for values (computations are lazy thus some values may be None).
        All not None values are always self-consistent with respect to DAG dependencies.

    _last_fork : None or Optional[:class:`~leaspy.variables.specs.VariablesLazyValuesRO`]
        If not None, holds the previous partial state values so they may be `.revert()`.
        Automatically populated on assignment operations as soon as `auto_fork_type` is not `NONE`.
        Example: if you set a new value for `a`, then value of `a` and of all its children just before assignment
        are held until either reversion or a new assignment.
    """

    def __init__(
        self, dag: VariablesDAG, *, auto_fork_type: Optional[StateForkType] = None
    ):
        self.dag = dag
        self.auto_fork_type = auto_fork_type
        self._tracked_variables: set[VariableName, ...] = set()
        self._values: VariablesLazyValuesRW = {}
        self._last_fork: Optional[VariablesLazyValuesRO] = None
        self.clear()

    @property
    def tracked_variables(self) -> set[VariableName, ...]:
        """
        Get the set of variable names currently tracked by the State.

        Returns
        -------
        :obj:`set` [:class:`~leaspy.variables.specs.VariableName`, ...]
            A set containing the names of the tracked variables.
        """
        return self._tracked_variables

    def track_variables(self, variable_names: Iterable[VariableName]) -> None:
        """
        Add some variables to the tracked variables.

        Parameters
        ----------
        variable_names : :obj:`~typing.Iterable` of :class:`~leaspy.variables.specs.VariableName`
            The names of the variables to be added to the tracked variables.
        """
        for variable_name in variable_names:
            self.track_variable(variable_name)

    def track_variable(self, variable_name: VariableName) -> None:
        """
        Add a single variable to the tracked variables.

        Parameters
        ----------
        variable_name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable to be added to the tracked variables.
        """
        if variable_name in self.dag:
            self._tracked_variables.add(variable_name)

    def untrack_variables(self, variable_names: Iterable[VariableName]) -> None:
        """
        Remove some variables from the tracked variables.

        Parameters
        ----------
        variable_names : :obj:`~typing.Iterable` of :class:`~leaspy.variables.specs.VariableName`
            The names of the variables to be removed from the tracked variables.
        """
        for variable_name in variable_names:
            self.untrack_variable(variable_name)

    def untrack_variable(self, variable_name: VariableName) -> None:
        """
        Remove a single variable from the tracked variables.

        Parameters
        ----------
        variable_name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable to be removed from the tracked variables.
        """
        if variable_name in self.dag:
            self._tracked_variables.discard(variable_name)

    def clear(self) -> None:
        """Reset last forked state and reset all values to their canonical values."""
        self._values = {
            n: var.value if isinstance(var, Hyperparameter) else None
            for n, var in self.dag.items()
        }
        self._last_fork = None

    def clone(
        self, *, disable_auto_fork: bool = False, keep_last_fork: bool = False
    ) -> State:
        """
        Clone current state without copying the DAG.

        Parameters
        ----------
        disable_auto_fork : :obj:`bool`, optional
            Whether to allow auto-fork or not.
            Default=False.

        keep_last_fork : :obj:`bool`, optional
            Whether to keep the last fork or not.
            Default=False.

        Returns
        -------
        :class:`~leaspy.variables.state.State` :
            The new cloned state instance.
        """
        cloned = State(
            self.dag, auto_fork_type=None if disable_auto_fork else self.auto_fork_type
        )
        cloned._values = copy.deepcopy(self._values)
        cloned._tracked_variables = self._tracked_variables
        if keep_last_fork:
            cloned._last_fork = copy.deepcopy(self._last_fork)
        return cloned

    @contextmanager
    def auto_fork(self, type: Optional[StateForkType] = StateForkType.REF):
        """
        Provide a context manager interface with temporary `auto_fork_type` set to `type`.
        
        Parameters
        ----------
        type : :class:`~leaspy.variables.state.StateForkType` or None, optional
            The temporary auto-forking strategy to use within the context.
            Defaults to `StateForkType.REF`.

        Yields
        ------
        None
            Control returns to the caller with the temporary forking strategy applied.
        """
        orig_auto_fork_type = self.auto_fork_type
        try:
            self.auto_fork_type = type
            yield
        finally:
            self.auto_fork_type = orig_auto_fork_type

    def __iter__(self):
        """
        Iterates on keys (.keys(), .values() and .items() methods are automatically provided by `MutableMapping`).
        
        Returns
        -------
        iterator
            An iterator over the variable names (keys) stored in the state.
        """
        return iter(self._values)

    def __len__(self) -> int:
        """
        Get number of variables.
        
        Returns
        -------
        :obj:`int`
            The number of variables in the state.
        """
        return len(self._values)

    def _check_key_exists(self, name: VariableName) -> None:
        """
        Verify that a variable name exists in the DAG.

        Parameters
        ----------
        name : :class:`~leaspy.variables.specs.VariableName`
            The variable name to check.

        Raises
        ------
        :exc:`LeaspyInputError`
            If the variable name is not found in the DAG.
        """
        if name not in self.dag:
            raise LeaspyInputError(f"'{name}' is not a valid variable")

    def _get_or_compute_and_cache(
        self,
        name: VariableName,
        *,
        force_computation: bool = False,
        why: str = " to proceed",
    ) -> VariableValue:
        """
        Retrieve the cached value of a variable (unless `force_computation` is True) or compute it, 
        assuming node exists and all its ancestors have cached values.
        
        Parameters
        ----------
        name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable to retrieve or compute.
        force_computation : :obj:`bool`, optional
            If True, forces recomputation of the variable value even if cached.
            Default is False.
        why : :obj:`str,` optional
            Explanation string used in the error message if computation fails. 
            Default is `" to proceed"`.

        Returns
        -------
        value : :class:`~leaspy.variables.specs.VariableValue`
            The cached or newly computed value of the variable.

        Raises
        ------
        :exc:`LeaspyInputError`
            If the variable is independent and its value cannot be computed (i.e., is None),
            indicating it is required.
        """
        if not force_computation:
            if (value := self._values[name]) is not None:
                return value
        value = self.dag[name].compute(self._values)
        if value is None:
            raise LeaspyInputError(
                f"'{name}' is an independent variable which is required{why}"
            )
        self._values[name] = value
        return value

    def __getitem__(self, name: VariableName) -> VariableValue:
        """
        Retrieve the cached value of a variable or compute it and then cache it 
        (as well as all intermediate computations that were needed).
        
        Parameters
        ----------
        name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable to retrieve.

        Returns
        -------
        :class:`~leaspy.variables.specs.VariableValue`
            The cached or newly computed value of the variable.

        Raises
        ------
        :exc:`LeaspyInputError`
            If the variable or any of its dependencies cannot be computed.
        """
        if (value := self._get_value_from_cache(name)) is not None:
            return value
        for parent in self.dag.sorted_ancestors[name]:
            self._get_or_compute_and_cache(parent, why=f" to get '{name}'")
        return self._get_or_compute_and_cache(name, force_computation=True)

    def __contains__(self, name: VariableName) -> bool:
        """
        Check whether a variable exists in the DAG.

        Parameters
        ----------
        name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable to check.

        Returns
        -------
        :obj:`bool`
            True if the variable is part of the DAG, False otherwise.
        """
        return name in self.dag

    def _get_value_from_cache(self, name: VariableName) -> Optional[VariableValue]:
        """
        Get the value for the provided variable name from the cache. 
        Raise an error if the value is not in the DAG.

        Parameters
        ----------
        name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable whose cached value is to be retrieved.

        Returns
        -------
        Optional[:class:`~leaspy.variables.specs.VariableValue`]
            The cached value of the variable, or `None` if it hasn't been computed.
        """
        self._check_key_exists(name)
        return self._values[name]

    def is_variable_set(self, name: VariableName) -> bool:
        """
        Returns True if the variable is in the DAG and if its value is not None.

        Parameters
        ----------
        name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable to check.

        Returns
        -------
        :obj:`bool`
            True if the variable exists in the DAG and its value has been set (i.e., is not None).
            False otherwise.
        """
        return self._get_value_from_cache(name) is not None

    def are_variables_set(self, variable_names: Iterable[VariableName]) -> bool:
        """
        Returns True if all the variables are in the DAG with values different from None.
        
        Parameters
        ----------
        variable_names : :obj:`Iterable`[:class:`~leaspy.variables.specs.VariableName`]
            A collection of variable names to check.

        Returns
        -------
        :obj:`bool`
            True if all variables exist in the DAG and their values are set (i.e., not None).
            False otherwise.
        """
        return all(self.is_variable_set(name) for name in variable_names)

    def __setitem__(self, name: VariableName, value: Optional[VariableValue]) -> None:
        """
        Smart and protected assignment of a variable value.

        Parameters
        ----------
        name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable to be set.
        value : :class:`~leaspy.variables.specs.VariableValue`
            The value of the variable to set.

        Raises
        ------
        :exc:`LeaspyInputError`
            If the variable does not exist in the DAG or is not settable.
        """
        self._check_key_exists(name)
        if not self.dag[name].is_settable:
            raise LeaspyInputError(f"'{name}' is not intended to be set")
        sorted_children = self.dag.sorted_children[name]
        # automatically fork partial state to easily revert it
        if self.auto_fork_type is not None:
            self._last_fork = self.auto_fork_type.to_cache(
                {child: self._values[child] for child in (name,) + sorted_children}
            )
        # TODO? we do not "validate" / "check" input data for now
        #  (it could be a stateless variable method) to remain light
        self._values[name] = value
        # we reset values of all children of the node we just assigned a value to
        # (we postpone the evaluation of their new values when they will really be needed)
        for child in sorted_children:
            self._values[child] = None

    def put(
        self,
        variable_name: VariableName,
        variable_value: VariableValue,
        *,
        indices: tuple[int, ...] = (),
        accumulate: bool = False,
    ) -> None:
        """
        Smart and protected assignment of a variable value, but potentially on a subset of indices, 
        adding (accumulating) values and OUT-OF-PLACE.

        Parameters
        ----------
        variable_name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable.
        variable_value : :class:`~leaspy.variables.specs.VariableValue`
            The new value to put in the variable name.
        indices : :obj:`tuple` of :obj:`int`, optional
            If set, the operation will happen on a subset of indices.
            Default=()
        accumulate : :obj:`bool`, optional
            If set to True, the new variable value will be added
            to the old value. Otherwise, it will be assigned.
            Default=False
        """
        if indices == ():
            # `torch.index_put` is not working in this case.
            if not accumulate:
                self[variable_name] = variable_value
            else:
                self[variable_name] = self[variable_name] + variable_value
            return
        # For now: no optimization for partial indices operations
        torch_indices = tuple(map(torch.tensor, indices))
        self[variable_name] = self[variable_name].index_put(
            indices=torch_indices,
            values=variable_value,
            accumulate=accumulate,
        )

    def __delitem__(self, name: VariableName) -> None:
        """
        Prevent deletion of variables from the state.

        Parameters
        ----------
        name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable to delete (attempted).

        Raises
        ------
        :exc:`NotImplementedError`
            Always raised to indicate that variable deletion is not allowed.
        """
        raise NotImplementedError("Key removal is not allowed")

    def precompute_all(self) -> None:
        """Pre-compute all values of the graph (assuming leaves already have valid values)."""
        for n in self.dag:
            self._get_or_compute_and_cache(n)

    # def reset_to_admissible(self) -> None:
    #    """Reset all standard variables to their frozen or admissible values and pre-compute all other variables (forked state is cleared)."""
    #    # TODO: more generic?
    #    self.clear()
    #    for n, var in self.dag.items():
    #        if isinstance(var, StandardVariable):
    #            self._values[n] = var.admissible_value
    #    self.precompute_all()

    def revert(
        self, subset: Optional[VariableValue] = None, *, right_broadcasting: bool = True
    ) -> None:
        """
        Revert state to previous forked state. Forked state is then reset.

        Parameters
        ----------
        subset : :class:`~leaspy.variables.specs.VariableValue` or None
            If not None, the reversion is only partial:
            * subset = True <=> revert previous state for those indices
            * subset = False <=> keep current state for those indices
            <!> User is responsible for having tensor values that are consistent with
            `subset` shape (i.e. valid broadcasting) for the forked node and all of its children.
           <!> When the current OR forked state is not set (value = None) on a particular node of forked DAG,
           then the reverted result is always None.
        right_broadcasting : :obj:`bool`, optional
            If True and if `subset` is not None, then the subset of indices to revert uses right-broadcasting,
            instead of the standard left-broadcasting.
            Default=True.

        Raises
        ------
        :exc:`LeaspyInputError`
            If no forked state exists to revert from (i.e., `.auto_fork()` context was not used).
        """
        if self._last_fork is None:
            raise LeaspyInputError(
                "No forked state to revert from, please use within `.auto_fork()` context, "
                "or set `.auto_fork_type` to  `StateForkType.REF` or `StateForkType.COPY`."
            )
        if subset is None:
            self._values.update(self._last_fork)
            self._last_fork = None
            return
        to_revert = subset.to(torch.bool)
        to_keep = ~to_revert
        for k, old_v in self._last_fork.items():
            cur_v = self._values[k]
            if old_v is None or cur_v is None:
                self._values[k] = None
            else:
                assert (
                    old_v.shape == cur_v.shape
                ), f"Bad shapes for {k}: {old_v.shape} != {cur_v.shape}"
                if right_broadcasting:
                    add_ndim = max(old_v.ndim - to_revert.ndim, 0)
                    self._values[k] = old_v * unsqueeze_right(
                        to_revert, ndim=add_ndim
                    ) + cur_v * unsqueeze_right(to_keep, ndim=add_ndim)
                else:
                    self._values[k] = old_v * to_revert + cur_v * to_keep
        self._last_fork = None

    def to_device(self, device: torch.device) -> None:
        """
        Move values to the specified device (in-place).

        Parameters
        ----------
        device : :class:`torch.device`
        """
        for k, v in self._values.items():
            if v is not None:
                self._values[k] = v.to(device=device)
        if self._last_fork is not None:
            for k, v in self._last_fork.items():
                if v is not None:
                    self._last_fork[k] = v.to(device=device)

    def put_population_latent_variables(
        self, method: Optional[Union[str, LatentVariableInitType]]
    ) -> None:
        """"
        Initialize all population latent variables in the state with predefined values.

        Parameters
        ----------
        method : obj:`str` or :class:`~leaspy.variables.specs.LatentVariableInitType` or None
            The method used to initialize the variables. If None, all population latent variables
            will be unset (set to None). Otherwise, the corresponding initialization function will
            be called for each variable using the provided method.

        """
        # Nota: fixing order of variables in this loop is pointless since no randomness is involved in init of pop. vars
        for pp, var in self.dag.sorted_variables_by_type[
            PopulationLatentVariable
        ].items():
            var: PopulationLatentVariable  # for type-hint only
            if method is None:
                self[pp] = None
            else:
                self[pp] = var.get_init_func(method).call(self)

    def put_individual_latent_variables(
        self,
        method: Optional[Union[str, LatentVariableInitType]] = None,
        *,
        n_individuals: Optional[int] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Initialize all individual latent variables in the state with predefined values.
        
        Parameters
        ----------
        method : :obj:`str` or  :class:`~leaspy.variables.specs.LatentVariableInitType`, optional
            The method used to initialize the variables. If None, the variables will be unset (set to None).
            If provided, an initialization function will be called per variable.
            When `method` is not None, `n_individuals` must be specified.
        n_individuals : :obj:`int`, optional
            Number of individuals to initialize. Required when `method` is not None and `df` is not provided.
        df : :obj:`pandas.DataFrame`, optional
            A DataFrame from which to directly extract the individual latent variable values.
            It must contain columns named 'tau' and 'xi' for direct assignment of these variables.
            If the "sources" variable is present, the DataFrame should include columns named
            'sources_0', 'sources_1', ..., up to the expected number of source variables.

        Raises
        ------
        :exc:`LeaspyInputError`
            If `method` is specified without `n_individuals`, or if required columns are missing in `df`.
        """
        if method is not None and n_individuals is None:
            raise LeaspyInputError(
                "`n_individuals` should not be None when `method` is not None."
            )
        individual_parameters = sorted(
            list(set(self.dag.sorted_variables_by_type[IndividualLatentVariable]))
        )
        if df is not None:
            for individual_parameter in individual_parameters:
                if individual_parameter in ("tau", "xi"):
                    self[individual_parameter] = torch.tensor(
                        df[[individual_parameter]].values, dtype=torch.double
                    )
                else:
                    nb_sources = len(df.columns) - 2
                    list_sources_name = [f"sources_{i}" for i in range(nb_sources)]
                    if not set(list_sources_name) <= set(df.columns):
                        raise LeaspyInputError(
                            "Please provide only individual parameters columns with sources stored as "
                            f"{list_sources_name}"
                        )
                    self["sources"] = torch.tensor(
                        df[list_sources_name].values, dtype=torch.double
                    ).float()
        else:
            for individual_parameter in individual_parameters:
                var: IndividualLatentVariable = self.dag[
                    individual_parameter
                ]  # for type-hint only
                if method is None:
                    self[individual_parameter] = None
                else:
                    self[individual_parameter] = var.get_init_func(
                        method, n_individuals=n_individuals
                    ).call(self)

    def save(self, output_folder: str, iteration: Optional[int] = None) -> None:
        """
        Save the tracked variable values of the state.

        Parameters
        ----------
        output_folder : :obj:`str`
            The path to the output folder in which the state's tracked variables should be saved.
        iteration : :obj:`int`, optional
            The iteration number when this method is called from an algorithm.
            This iteration number will appear at the beginning of the row.
        """
        output_folder = Path(output_folder)
        for variable in self._tracked_variables:
            dict_value = self._get_value_as_dict_of_lists(variable)
            for variable_name, value in dict_value.items():
                if iteration is not None:
                    value.insert(0, iteration)
                with open(
                    output_folder / f"{variable_name}.csv", "a", newline=""
                ) as filename:
                    writer = csv.writer(filename)
                    writer.writerow(value)

    def _get_value_as_dict_of_lists(
        self, variable_name: VariableName
    ) -> dict[VariableName, list[float]]:
        """
        Return the value of the given variable as a dictionary with list of floats.
        
        Parameters
        ----------
        variable_name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable whose value is to be converted.

        Returns
        -------
        :obj:`dict` of {:class:`~leaspy.variables.specs.VariableName`: :obj:`list` of :obj:`float`}
            A dictionary mapping variable names (or indexed variable names for 2D tensors) to
            lists of floats representing the variable's values.

        Raises
        ------
        :exc:`ValueError`
            If the variable value tensor has more than 2 dimensions, which is unsupported.
        """
        value = self.__getitem__(variable_name)
        if isinstance(value, WeightedTensor):
            value = value.weighted_value
        if value.ndim == 0:
            return {variable_name: [value.tolist()]}
        elif value.ndim == 1:
            return {variable_name: value.tolist()}
        elif value.ndim == 2:
            if value.shape[1] == 1:
                return {variable_name: value[:, 0].tolist()}
            else:
                return {
                    f"{variable_name}_{i}": value[:, i].tolist()
                    for i in range(value.shape[1])
                }
        else:
            raise ValueError(
                f"The value of variable {variable_name} "
                f"is a tensor of dimension {value.ndim} > 2 "
                f"which is not supported. The value is: {value}."
            )

    def get_tensor_value(self, variable_name: VariableName) -> VariableValue:
        """
        Return the value of the provided variable as a torch tensor.

        Parameters
        ----------
        variable_name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable for which to retrieve the value.

        Returns
        -------
        :class:`~leaspy.variables.specs.VariableValue` :
            The value of the variable.
        """
        if isinstance(self[variable_name], WeightedTensor):
            return self[variable_name].weighted_value
        return self[variable_name]

    def get_tensor_values(
        self, variable_names: Iterable[VariableName]
    ) -> tuple[VariableValue, ...]:
        """
        Return the values of the provided variables as torch tensors.

        Parameters
        ----------
        variable_names : :class:`~typing.Iterable` of :class:`~leaspy.variables.specs.VariableName`
            The names of the variables for which to retrieve the values.

        Returns
        -------
        :obj:`tuple` of :class:`~leaspy.variables.specs.VariableValue` :
            The values of the variables.
        """
        return tuple(self.get_tensor_value(name) for name in variable_names)
