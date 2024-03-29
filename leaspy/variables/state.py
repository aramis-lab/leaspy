from __future__ import annotations

from collections.abc import MutableMapping
from typing import Tuple, Optional, Set, Iterable, List, Union
from contextlib import contextmanager
from enum import Enum, auto
import copy
import csv
import pandas as pd
from pathlib import Path
import torch

from leaspy.variables.specs import (
    VarName,
    VarValue,
    VariablesLazyValuesRO,
    VariablesLazyValuesRW,
    Hyperparameter,
    PopulationLatentVariable,
    IndividualLatentVariable,
    LatentVariableInitType,
)
from leaspy.utils.weighted_tensor import unsqueeze_right, WeightedTensor
from leaspy.variables.dag import VariablesDAG
from leaspy.exceptions import LeaspyInputError


class StateForkType(Enum):
    """
    The strategy used to cache forked values in :class:`.State`.

    REF : caching using references
    COPY : caching using deepcopy

    Notes
    -----
    If using `REF` beware that values will NOT be copied (it only keeps references of values),
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
    dag : VariablesDAG
        The stateless DAG which state will hold values for.
    auto_fork_type : :class:`.StateForkType` or None (default)
        Refer to :class:`.StateForkType` class and :attr:`auto_fork_type`

    Attributes
    ----------
    dag : VariablesDAG
        The stateless DAG which the state instance will hold values for.
    auto_fork_type : :class:`.StateForkType` or None
        If not `StateForkType.NONE` each dictionary assignment will lead to the partial caching
        of previous value and all its children, so they can be reverted without computation.
        The exact caching strategy depends on flag (caching by reference or by copy)
        Can be manually set or via `auto_fork` context manager.
    _values : MutableMapping[VarName, Optional[VarValue]]
        Private cache for values (computations are lazy thus some values may be None).
        All not None values are always self-consistent with respect to DAG dependencies.
    _last_fork : None or Mapping[VarName, Optional[VarValue]]
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
        self._tracked_variables: Set[str, ...] = set()
        self._values: VariablesLazyValuesRW = {}
        self._last_fork: Optional[VariablesLazyValuesRO] = None
        self.clear()

    @property
    def tracked_variables(self) -> Set[str, ...]:
        return self._tracked_variables

    def track_variables(self, variable_names: Iterable[str]) -> None:
        for variable_name in variable_names:
            self.track_variable(variable_name)

    def track_variable(self, variable_name: str) -> None:
        if variable_name in self.dag:
            self._tracked_variables.add(variable_name)

    def untrack_variables(self, variable_names: Iterable[str]) -> None:
        for variable_name in variable_names:
            self.untrack_variable(variable_name)

    def untrack_variable(self, variable_name: str) -> None:
        if variable_name in self.dag:
            self._tracked_variables.discard(variable_name)

    def clear(self) -> None:
        """Reset last forked state and reset all values to their canonical values."""
        self._values = {
            n: var.value if isinstance(var, Hyperparameter) else None
            for n, var in self.dag.items()
        }
        self._last_fork = None

    def clone(self, *, disable_auto_fork: bool = False, keep_last_fork: bool = False) -> State:
        """Clone current state (no copy of DAG)."""
        cloned = State(self.dag, auto_fork_type=None if disable_auto_fork else self.auto_fork_type)
        cloned._values = copy.deepcopy(self._values)
        cloned._tracked_variables = self._tracked_variables
        if keep_last_fork:
            cloned._last_fork = copy.deepcopy(self._last_fork)
        return cloned

    @contextmanager
    def auto_fork(self, type: Optional[StateForkType] = StateForkType.REF):
        """Provide a context manager interface with temporary `auto_fork_type` set to `type`."""
        orig_auto_fork_type = self.auto_fork_type
        try:
            self.auto_fork_type = type
            yield
        finally:
            self.auto_fork_type = orig_auto_fork_type

    def __iter__(self):
        """Iterates on keys (.keys(), .values() and .items() methods are automatically provided by `MutableMapping`)."""
        return iter(self._values)

    def __len__(self) -> int:
        """Get number of variables."""
        return len(self._values)

    def _check_key_exists(self, k: VarName) -> None:
        if k not in self.dag:
            raise LeaspyInputError(f"'{k}' is not a valid variable")

    def _get_or_compute_and_cache(
        self,
        k: VarName,
        *,
        force_computation: bool = False,
        why: str = " to proceed",
    ) -> VarValue:
        """
        Retrieve cached value (unless `force_computation`) or compute it,
        assuming node exists and all its ancestors have cached values.
        """
        if not force_computation:
            if (val := self._values[k]) is not None:
                return val

        val = self.dag[k].compute(self._values)
        if val is None:
            raise LeaspyInputError(
                f"'{k}' is an independent variable which is required{why}"
            )
        self._values[k] = val
        return val

    def __getitem__(self, k: VarName) -> VarValue:
        """
        Retrieve cached variable value or compute it and cache it
        (as well as all intermediate computations that were needed).
        """
        if (val := self._get_value_from_cache(k)) is not None:
            return val
        for a in self.dag.sorted_ancestors[k]:
            self._get_or_compute_and_cache(a, why=f" to get '{k}'")
        return self._get_or_compute_and_cache(k, force_computation=True)

    def __contains__(self, k: VarName) -> bool:
        return k in self.dag

    def _get_value_from_cache(self, k: VarName) -> Optional[VarValue]:
        """Get the value for variable named k from the cache. Raise if not in DAG."""
        self._check_key_exists(k)
        return self._values[k]

    def is_variable_set(self, k: VarName) -> bool:
        """Returns True if the variable is in the DAG and if its value is not None."""
        return self._get_value_from_cache(k) is not None

    def are_variables_set(self, variable_names: Iterable[VarName]) -> bool:
        """Returns True if all the variables are in the DAG with values different from None."""
        return all(self.is_variable_set(k) for k in variable_names)

    def __setitem__(self, k: VarName, v: Optional[VarValue]) -> None:
        """Smart and protected assignment of a variable value."""
        self._check_key_exists(k)
        if not self.dag[k].is_settable:
            raise LeaspyInputError(f"'{k}' is not intended to be set")
        sorted_children = self.dag.sorted_children[k]
        # automatically fork partial state to easily revert it
        if self.auto_fork_type is not None:
            self._last_fork = self.auto_fork_type.to_cache(
                {c: self._values[c] for c in (k,) + sorted_children}
            )
        # TODO? we do not "validate" / "check" input data for now
        #  (it could be a stateless variable method) to remain light
        self._values[k] = v
        # we reset values of all children of the node we just assigned a value to
        # (we postpone the evaluation of their new values when they will really be needed)
        for c in sorted_children:
            self._values[c] = None

    def put(
        self,
        variable_name: VarName,
        variable_value: VarValue,
        *,
        indices: Tuple[int, ...] = (),
        accumulate: bool = False,
    ) -> None:
        """
        Smart and protected assignment of a variable value, but potentially on a subset of indices,
        adding (accumulating) values and OUT-OF-PLACE.

        Parameters
        ----------
        variable_name : VarName
            The name of the variable.
        variable_value : torch.Tensor
            The new value to put in the variable name.
        indices : Tuple of int, optional
            If set, the operation will happen on a subset of indices.
            Default=()
        accumulate : bool, optional
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

    def __delitem__(self, k: VarName) -> None:
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
        self, subset: Optional[torch.Tensor] = None, *, right_broadcasting: bool = True
    ) -> None:
        """
        Revert state to previous forked state, efficiently (forked state is then reset).

        Parameters
        ----------
        subset : None (default) or VarValue[bool]
            If not None, the reversion is only partial:
            * subset = True <=> revert previous state for those indices
            * subset = False <=> keep current state for those indices
            <!> User is responsible for having tensor values that are consistent with
            `subset` shape (i.e. valid broadcasting) for the forked node and all of its children.
           <!> When the current OR forked state is not set (value = None) on a particular node of forked DAG,
           then the reverted result is always None.
        right_broadcasting : bool (default True)
            If True and if `subset` is not None, then the subset of indices to revert uses right-broadcasting,
            instead of the standard left-broadcasting.
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
                    self._values[k] = (
                        old_v * unsqueeze_right(to_revert, ndim=add_ndim)
                        + cur_v * unsqueeze_right(to_keep, ndim=add_ndim)
                    )
                else:
                    self._values[k] = old_v * to_revert + cur_v * to_keep
        self._last_fork = None

    def to_device(self, device: torch.device) -> None:
        """
        Move values to the specified device (in-place).

        Parameters
        ----------
        device : torch.device
        """
        for k, v in self._values.items():
            if v is not None:
                self._values[k] = v.to(device=device)
        if self._last_fork is not None:
            for k, v in self._last_fork.items():
                if v is not None:
                    self._last_fork[k] = v.to(device=device)

    def put_population_latent_variables(self, method: Optional[Union[str, LatentVariableInitType]]) -> None:
        """Put some predefined values in state for all population latent variables (in-place)."""
        # Nota: fixing order of variables in this loop is pointless since no randomness is involved in init of pop. vars
        for pp, var in self.dag.sorted_variables_by_type[PopulationLatentVariable].items():
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

        """Put some predefined values in state for all individual latent variables (in-place)."""
        if method is not None and n_individuals is None:
            raise LeaspyInputError("`n_individuals` should not be None when `method` is not None.")

        # TMP --> fix order of random variables as previously to pass functional tests...
        vars_order = set(self.dag.sorted_variables_by_type[IndividualLatentVariable])
        if vars_order == {'tau', 'xi'}:
            vars_order = ['tau', 'xi']
        elif vars_order == {'tau', 'xi', 'sources'}:
            vars_order = ['tau', 'xi', 'sources']
        # END TMP

        if df is not None:
            for ip in vars_order:
                self[ip] = torch.tensor(df[[ip]].values)
        else:
            # for ip, var in self.dag.sorted_variables_by_type[IndividualLatentVariable].items():
            for ip in vars_order:
                var: IndividualLatentVariable = self.dag[ip]  # for type-hint only
                if method is None:
                    self[ip] = None
                else:
                    self[ip] = var.get_init_func(method, n_individuals=n_individuals).call(self)

    def save(self, output_folder: str, iteration: Optional[int] = None) -> None:
        """Save the tracked variable values of the state.

        Parameters
        ----------
        output_folder : str
            The path to the output folder in which the state's tracked variables
            should be saved.
        iteration : int, optional
            The iteration number when this method is called from an
            algorithm. This iteration number will appear at the beginning of the row.
        """
        output_folder = Path(output_folder)
        for variable in self._tracked_variables:
            value = self._get_value_as_list_of_floats(variable)
            if iteration != None:
                value.insert(0, iteration)
            with open(output_folder / f"{variable}.csv", 'a', newline='') as filename:
                writer = csv.writer(filename)
                writer.writerow(value)

    def _get_value_as_list_of_floats(self, variable_name: str) -> List[float, ...]:
        """Return the value of the given variable as a list of floats."""
        value = self.__getitem__(variable_name)
        if isinstance(value, WeightedTensor):
            value = value.weighted_value
        try:
            return [value.item()]
        except ValueError:
            try:
                return [tensor.item() for tensor in value]
            except ValueError:
                raise ValueError(
                    f"Unable to get the value of variable {variable_name} as a list of floats. "
                    f"The value in the state for this variable is : {value}."
                )

    def get_tensor_value(self, variable_name: str) -> torch.Tensor:
        if isinstance(self[variable_name], WeightedTensor):
            return self[variable_name].weighted_value
        return self[variable_name]

    def get_tensor_values(self, variable_names: Iterable[str]) -> Tuple[torch.Tensor, ...]:
        return tuple(self.get_tensor_value(name) for name in variable_names)


