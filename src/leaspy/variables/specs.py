from __future__ import annotations

from abc import abstractmethod
from collections import UserDict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    ClassVar,
    Optional,
    Union,
)
from typing import (
    Mapping as TMapping,
)
from typing import (
    MutableMapping as TMutableMapping,
)

import torch

from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.functional import (
    Identity,
    Mean,
    NamedInputFunction,
    Prod,
    Sqr,
    Std,
    Sum,
    SumDim,
    get_named_parameters,
)
from leaspy.utils.typing import KwargsType
from leaspy.utils.weighted_tensor import (
    TensorOrWeightedTensor,
    WeightedTensor,
    expand_left,
    sum_dim,
)

from .distributions import SymbolicDistribution
from .utilities import (
    compute_correlation_ind,
    compute_correlation_pop,
    compute_individual_parameter_std_from_sufficient_statistics,
)

__all__ = [
    "VariableName",
    "VariableValue",
    "VariableNameToValueMapping",
    "VariablesToFrozenSet",
    "VariablesLazyValuesRW",
    "VariablesLazyValuesRO",
    "SuffStatsRO",
    "SuffStatsRW",
    "VariableInterface",
    "IndepVariable",
    "Hyperparameter",
    "Collect",
    "ModelParameter",
    "DataVariable",
    "LatentVariableInitType",
    "LatentVariable",
    "PopulationLatentVariable",
    "IndividualLatentVariable",
    "LinkedVariable",
    "NamedVariables",
]

VariableName = str
VariableValue = TensorOrWeightedTensor[float]
VariableNameToValueMapping = TMapping[VariableName, VariableValue]
VariablesToFrozenSet = TMapping[VariableName, frozenset[VariableValue]]
VariablesLazyValuesRO = TMapping[VariableName, Optional[VariableValue]]
VariablesLazyValuesRW = TMutableMapping[VariableName, Optional[VariableValue]]
SuffStatsRO = TMapping[VariableName, torch.Tensor]  # VarValue
SuffStatsRW = TMutableMapping[VariableName, torch.Tensor]  # VarValue

LVL_IND = 0
LVL_FT = -1


class VariableInterface:
    """Interface for variable specifications."""

    is_settable: ClassVar[bool]
    """Is True if and only if state of variables is intended to be manually modified by user."""

    fixed_shape: ClassVar[bool]
    """Is True as soon as we guarantee that shape of variable is only dependent on model hyperparameters, not data."""

    @abstractmethod
    def compute(self, state: VariableNameToValueMapping) -> Optional[VariableValue]:
        """Compute variable value from a `state` exposing a dict-like interface: var_name -> values.

        If not relevant for variable type return None.

        Parameters
        ----------
        state : :class:`~leaspy.variables.specs.VariableNameToValueMapping`
            The state to use in order to perform computations.

        Returns
        -------
        :class:`~leaspy.variables.specs.VariableValue` :
            The variable value computed from the state.
        """

    @abstractmethod
    def get_ancestors_names(self) -> frozenset[VariableName]:
        """Get the names of the variables that the current variable directly depends on.

        Returns
        -------
        :obj:`frozenset` [ :class:`~leaspy.variables.specs.VariableName`] :
            The set of ancestors variable names.
        """

    # TODO? add a check or validate(value) method? (to be optionally called by State)
    # <!> should some extra context be passed to this method
    # (e.g. `n_individuals` or `n_timepoints` dimensions are not known during variable definition
    # but their consistency could/should be checked?)


class IndepVariable(VariableInterface):
    """Base class for variable that is not dependent on any other variable."""

    def get_ancestors_names(self) -> frozenset[VariableName]:
        """Get the names of the variables that the current variable directly depends on.

        Returns
        -------
        :obj:`frozenset` [ :class:`~leaspy.variables.specs.VariableName`] :
            The set of ancestors variable names.
        """
        return frozenset()

    def compute(self, state: VariableNameToValueMapping) -> Optional[VariableValue]:
        """Compute variable value from a `state` exposing a dict-like interface: var_name -> values.

        If not relevant for variable type return None.

        Parameters
        ----------
        state : :class:`~leaspy.variables.specs.VariableNameToValueMapping`
            The state to use in order to perform computations.

        Returns
        -------
        :class:`~leaspy.variables.specs.VariableValue` or None:
            The variable value computed from the state.
        """
        return None


@dataclass(frozen=True)
class Hyperparameter(IndepVariable):
    """Hyperparameters that can not be reset."""

    value: VariableValue
    """The hyperparameter value."""

    fixed_shape: ClassVar = True
    """Whether the variable has a fixed shape or not. For hyperparameters this is True."""

    is_settable: ClassVar = False
    """Whether the variable is mutable or not. For hyperparameters this is False."""

    def __post_init__(self):
        if not isinstance(self.value, (torch.Tensor, WeightedTensor)):
            object.__setattr__(self, "value", torch.tensor(self.value))

    def to_device(self, device: torch.device) -> None:
        """Move the value to specified device (other variables never hold values so need for this method).

        Parameters
        ----------
        device : :class:`torch.device`
            The device on which to move the variable value.
        """
        return object.__setattr__(self, "value", self.value.to(device=device))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape


@dataclass(frozen=True, init=False)
class Collect:
    """A convenient class to produce a function to collect sufficient stats that are existing or dedicated variables (to be automatically created)."""

    existing_variables: tuple[VariableName, ...] = ()
    dedicated_variables: Optional[TMapping[VariableName, LinkedVariable]] = None

    def __init__(
        self, *existing_variables: VariableName, **dedicated_variables: LinkedVariable
    ):
        # custom init to allow more convenient variadic form
        object.__setattr__(self, "existing_variables", existing_variables)
        object.__setattr__(self, "dedicated_variables", dedicated_variables or None)

    @property
    def variables(self) -> tuple[VariableName, ...]:
        return self.existing_variables + tuple(self.dedicated_variables or ())

    def __call__(self, state: VariableNameToValueMapping) -> SuffStatsRW:
        return {k: state[k] for k in self.variables}


@dataclass(frozen=True)
class ModelParameter(IndepVariable):
    """Variable for model parameters with a maximization rule. This variable shouldn't be sampled and it shouldn't be data, a hyperparameter or a linked variable."""

    shape: tuple[int, ...]
    suff_stats: Collect  # Callable[[VariablesValuesRO], SuffStatsRW]
    """
    The symbolic update functions will take variadic `suff_stats` values,
    in order to reuse NamedInputFunction logic: e.g. update_rule=Std('xi')

    <!> ISSUE: for `tau_std` and `xi_std` we also need `state` values in addition to
    `suff_stats` values (only after burn-in) since we can NOT use the variadic form
    readily for both `state` and `suff_stats` (names would be conflicting!), we sent
    `state` as a special kw variable (a bit lazy but valid) (and we prevent using this
    name for a variable as a safety)
    """

    update_rule: Callable[..., VariableValue]
    """Update rule for normal phase, and memory-less (burn-in) phase unless `update_rule_burn_in` is not None."""

    update_rule_burn_in: Optional[Callable[..., VariableValue]] = None
    """Specific rule for burn-in (currently implemented for some variables -> e.g. `xi_std`)"""

    # private attributes (computed in __post_init__)
    _update_rule_parameters: frozenset[VariableName] = field(init=False, repr=False)
    _update_rule_burn_in_parameters: Optional[frozenset[VariableName]] = field(
        default=None, init=False, repr=False
    )

    fixed_shape: ClassVar = True
    is_settable: ClassVar = True

    def __post_init__(self):
        self._check_and_store_update_rule_parameters("update_rule")
        self._check_and_store_update_rule_parameters("update_rule_burn_in")

    def _check_and_store_update_rule_parameters(self, update_method: str) -> None:
        method = getattr(self, update_method)
        if method is None:
            return
        allowed_kws = set(self.suff_stats.variables).union({"state"})
        err_msg = (
            f"Function provided in `ModelParameter.{update_method}` should be a function with keyword-only parameters "
            "(using names of this variable sufficient statistics, or the special 'state' keyword): not {}"
        )
        try:
            inferred_params = get_named_parameters(method)
        except ValueError as e:
            raise LeaspyModelInputError(err_msg.format(str(e))) from e
        forbidden_kws = set(inferred_params).difference(allowed_kws)
        if len(forbidden_kws):
            raise LeaspyModelInputError(err_msg.format(forbidden_kws))

        object.__setattr__(
            self, f"_{update_method}_parameters", frozenset(inferred_params)
        )

    def compute_update(
        self,
        *,
        state: VariableNameToValueMapping,
        suff_stats: SuffStatsRO,
        burn_in: bool,
    ) -> VariableValue:
        """Compute the updated value for the model parameter using a maximization step.

        Parameters
        ----------
        state : :class:`~leaspy.variables.specs.VariableNameToValueMapping`
            The state to use for computations.

        suff_stats : :class:`~leaspy.variables.specs.SuffStatsRO`
            The sufficient statistics to use.

        burn_in : :obj:`bool`
            If True, use the update rule in burning phase.

        Returns
        -------
        :class:`~leaspy.variables.specs.VariableValue` :
            The computed variable value.
        """
        update_rule, update_rule_params = self.update_rule, self._update_rule_parameters
        if burn_in and self.update_rule_burn_in is not None:
            update_rule, update_rule_params = (
                self.update_rule_burn_in,
                self._update_rule_burn_in_parameters,
            )
        state_kw = dict(state=state) if "state" in update_rule_params else {}
        # <!> it would not be clean to send all suff_stats (unfiltered) for standard kw-only functions...
        return update_rule(
            **state_kw,
            **{k: suff_stats[k] for k in self._update_rule_parameters if k != "state"},
        )

    @classmethod
    def for_pop_mean(
        cls, population_variable_name: VariableName, shape: tuple[int, ...]
    ):
        """Smart automatic definition of `ModelParameter` when it is the mean of Gaussian prior of a population latent variable."""
        return cls(
            shape,
            suff_stats=Collect(population_variable_name),
            update_rule=Identity(population_variable_name),
        )

    @classmethod
    def for_ind_mean(
        cls, individual_variable_name: VariableName, shape: tuple[int, ...]
    ):
        """Smart automatic definition of `ModelParameter` when it is the mean of Gaussian prior of an individual latent variable."""
        return cls(
            shape,
            suff_stats=Collect(individual_variable_name),
            update_rule=Mean(individual_variable_name, dim=LVL_IND),
        )

    @classmethod
    def for_ind_std(
        cls, individual_variable_name: VariableName, shape: tuple[int, ...], **tol_kw
    ):
        """Smart automatic definition of `ModelParameter` when it is the std-dev of Gaussian prior of an individual latent variable."""
        individual_variance_sqr_name = f"{individual_variable_name}_sqr"
        update_rule_normal = NamedInputFunction(
            compute_individual_parameter_std_from_sufficient_statistics,
            parameters=(
                "state",
                individual_variable_name,
                individual_variance_sqr_name,
            ),
            kws=dict(
                individual_parameter_name=individual_variable_name,
                dim=LVL_IND,
                **tol_kw,
            ),
        )
        return cls(
            shape,
            suff_stats=Collect(
                individual_variable_name,
                **{
                    individual_variance_sqr_name: LinkedVariable(
                        Sqr(individual_variable_name)
                    )
                },
            ),
            update_rule_burn_in=Std(individual_variable_name, dim=LVL_IND),
            update_rule=update_rule_normal,
        )

    @classmethod
    def for_ind_coeff_corr(
        cls,
        variable_name: VariableName,
        shape: tuple[int, ...],
    ):
        """Smart automatic definition of `ModelParameter` when it is a correlation coefficient (rho)
        between two components of an individual latent variable (e.g. phi_tau[:, 0] and phi_tau[:, 1]).
        """
        update_rule = NamedInputFunction(
            compute_correlation_ind,
            parameters=(
                "state",
                variable_name,
            ),
            kws=dict(
                parameters_name=variable_name,
                dim=LVL_IND,
            ),
        )
        return cls(
            shape,
            suff_stats=Collect(variable_name),
            update_rule=update_rule,
        )

    @classmethod
    def for_pop_coeff_corr(
        cls,
        variable_name: VariableName,
        shape: tuple[int, ...],
    ):
        """Smart automatic definition of `ModelParameter` when it is a correlation coefficient (rho)
        between two components of an population latent variable (e.g. phi_tau[:, 0] and phi_tau[:, 1]).
        """
        update_rule = NamedInputFunction(
            compute_correlation_pop,
            parameters=(
                "state",
                variable_name,
            ),
            kws=dict(
                parameters_name=variable_name,
                dim=LVL_IND,
            ),
        )
        return cls(
            shape,
            suff_stats=Collect(variable_name),
            update_rule=update_rule,
        )


@dataclass(frozen=True)
class DataVariable(IndepVariable):
    """Variables for input data, that may be reset."""

    fixed_shape: ClassVar = False
    is_settable: ClassVar = True


class LatentVariableInitType(str, Enum):
    """Type of initialization for latent variables."""

    PRIOR_MODE = "mode"
    PRIOR_MEAN = "mean"
    PRIOR_SAMPLES = "samples"


@dataclass(frozen=True)
class LatentVariable(IndepVariable):
    """Unobserved variable that will be sampled, with symbolic prior distribution [e.g. Normal('xi_mean', 'xi_std')]."""

    # TODO/WIP? optional mask derive from optional masks of prior distribution parameters?
    # or should be fixed & explicit here?
    prior: SymbolicDistribution
    sampling_kws: Optional[KwargsType] = None

    is_settable: ClassVar = True

    def get_prior_shape(
        self, named_vars: TMapping[VariableName, VariableInterface]
    ) -> tuple[int, ...]:
        """Get shape of prior distribution (i.e. without any expansion for `IndividualLatentVariable`)."""
        bad_params = {
            n for n in self.prior.parameters_names if not named_vars[n].fixed_shape
        }
        if len(bad_params):
            raise LeaspyModelInputError(
                f"Shapes of some prior distribution parameters are not fixed: {bad_params}"
            )
        params_shapes = {n: named_vars[n].shape for n in self.prior.parameters_names}
        return self.prior.shape(**params_shapes)

    def _get_init_func_generic(
        self,
        method: Union[str, LatentVariableInitType],
        *,
        sample_shape: tuple[int, ...],
    ) -> NamedInputFunction[torch.Tensor]:
        """Return a `NamedInputFunction`: State -> Tensor, that may be used for initialization."""
        method = LatentVariableInitType(method)
        if method is LatentVariableInitType.PRIOR_SAMPLES:
            return self.prior.get_func_sample(sample_shape)
        if method is LatentVariableInitType.PRIOR_MODE:
            return self.prior.mode.then(expand_left, shape=sample_shape)
        if method is LatentVariableInitType.PRIOR_MEAN:
            return self.prior.mean.then(expand_left, shape=sample_shape)

    @abstractmethod
    def get_regularity_variables(
        self, value_name: VariableName
    ) -> dict[VariableName, LinkedVariable]:
        """Get extra linked variables to compute regularity term for this latent variable."""
        # return {
        #    # Not really useful... directly sum it to be memory efficient...
        #    f"nll_regul_{value_name}_full": LinkedVariable(
        #        self.prior.get_func_regularization(value_name)
        #    ),
        #    # TODO: jacobian as well...
        # }
        pass


class PopulationLatentVariable(LatentVariable):
    """Population latent variable."""

    # not so easy to guarantee the fixed shape property in fact...
    # (it requires that parameters of prior distribution all have fixed shapes)
    fixed_shape: ClassVar = True

    def get_init_func(
        self,
        method: Union[str, LatentVariableInitType],
    ) -> NamedInputFunction[torch.Tensor]:
        """Return a `NamedInputFunction`: State -> Tensor, that may be used for initialization.

        Parameters
        ----------
        method : :class:`~leaspy.variables.specs.LatentVariableInitType` or :obj:`str`
            The method to be used.

        Returns
        -------
        :class:`~leaspy.utils.functional.NamedInputFunction` :
            The initialization function.
        """
        return self._get_init_func_generic(method=method, sample_shape=())

    def get_regularity_variables(
        self,
        variable_name: VariableName,
    ) -> dict[VariableName, LinkedVariable]:
        """
        Return the negative log likelihood regularity for the
        provided variable name.

        Parameters
        ----------
        variable_name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable for which to retrieve regularity.

        Returns
        -------
        :obj:`dict` [ :class:`~leaspy.variables.specs.VariableName`, :class:`~leaspy.variables.specs.LinkedVariable`] :
            The dictionary holding the :class:`~leaspy.variables.specs.LinkedVariable` for the regularity.
        """
        # d = super().get_regularity_variables(value_name)
        d = {}
        d.update(
            {
                f"nll_regul_{variable_name}": LinkedVariable(
                    # SumDim(f"nll_regul_{value_name}_full")
                    self.prior.get_func_regularization(variable_name).then(sum_dim)
                ),
                # TODO: jacobian as well...
            }
        )
        return d


class IndividualLatentVariable(LatentVariable):
    """Individual latent variable."""

    fixed_shape: ClassVar = False

    def get_init_func(
        self,
        method: Union[str, LatentVariableInitType],
        *,
        n_individuals: int,
    ) -> NamedInputFunction[torch.Tensor]:
        """Return a `NamedInputFunction`: State -> Tensor, that may be used for initialization.

        Parameters
        ----------
        method : :class:`~leaspy.variables.specs.LatentVariableInitType` or :obj:`str`
            The method to be used.

        n_individuals : :obj:`int`
            The number of individuals, used to define the shape.

        Returns
        -------
        :class:`~leaspy.utils.functional.NamedInputFunction` :
            The initialization function.
        """
        return self._get_init_func_generic(method=method, sample_shape=(n_individuals,))

    def get_regularity_variables(
        self,
        variable_name: VariableName,
    ) -> dict[VariableName, LinkedVariable]:
        """Return the negative log likelihood regularity for the provided variable name.

        Parameters
        ----------
        variable_name : :class:`~leaspy.variables.specs.VariableName`
            The name of the variable for which to retrieve regularity.

        Returns
        -------
        :obj:`dict` [ :class:`~leaspy.variables.specs.VariableName`, :class:`~leaspy.variables.specs.LinkedVariable`] :
            The dictionary holding the :class:`~leaspy.variables.specs.LinkedVariable` for the regularity.
        """
        # d = super().get_regularity_variables(value_name)
        d = {}
        d.update(
            {
                f"nll_regul_{variable_name}_ind": LinkedVariable(
                    # SumDim(f"nll_regul_{value_name}_full", but_dim=LVL_IND)
                    self.prior.get_func_regularization(variable_name).then(
                        sum_dim, but_dim=LVL_IND
                    )
                ),
                f"nll_regul_{variable_name}": LinkedVariable(
                    SumDim(f"nll_regul_{variable_name}_ind")
                ),
                # TODO: jacobian as well...
            }
        )
        return d


@dataclass(frozen=True)
class LinkedVariable(VariableInterface):
    """Variable which is a deterministic expression of other variables (we directly use variables names instead of mappings: kws <-> vars)."""

    f: Callable[..., VariableValue]
    parameters: frozenset[VariableName] = field(init=False)
    # expected_shape? (<!> some of the shape dimensions might not be known like `n_individuals` or `n_timepoints`...)
    # admissible_value? (<!> same issue than before, cf. remark on `IndividualLatentVariable`)

    is_settable: ClassVar = False
    # shape of linked variable may be fixed in some cases, but complex/boring/useless logic to guarantee it
    fixed_shape: ClassVar = False

    def __post_init__(self):
        try:
            inferred_params = get_named_parameters(self.f)
        except ValueError:
            raise LeaspyModelInputError(
                "Function provided in `LinkedVariable` should be a function with "
                "keyword-only parameters (using variables names)."
            )
        object.__setattr__(self, "parameters", frozenset(inferred_params))

    def get_ancestors_names(self) -> frozenset[VariableName]:
        return self.parameters

    def compute(self, state: VariableNameToValueMapping) -> VariableValue:
        """Compute the variable value from a given State.

        Parameters
        ----------
        state : :class:`~leaspy.variables.specs.VariableNameToValueMapping`
            The state to use for computations.

        Returns
        -------
        :class:`~leaspy.variables.specs.VariableValue` :
            The value of the variable.
        """
        # print("[debug]", {k: state[k].shape for k in self.parameters})
        return self.f(**{k: state[k] for k in self.parameters})


class NamedVariables(UserDict):
    """Convenient dictionary for named variables specifications.

    In particular, it:
        1. forbids the collisions in variable names when assigning/updating the collection
        2. forbids the usage of some reserved names like 'state' or 'suff_stats'
        3. automatically adds implicit variables when variables of certain kind are added
           (e.g. dedicated vars for sufficient stats of ModelParameter)
        4. automatically adds summary variables depending on all contained variables
           (e.g. `nll_regul_ind_sum` that depends on all individual latent variables contained)

    <!> For now, you should NOT update a `NamedVariables` with another one, only update with a regular mapping.
    """

    FORBIDDEN_NAMES: ClassVar = frozenset(
        {
            "all",
            "pop",
            "ind",
            "sum",
            "tot",
            "full",
            "nll",
            "attach",
            "regul",
            "state",
            "suff_stats",
        }
    )

    AUTOMATIC_VARS: ClassVar = (
        # TODO? jacobians as well
        "nll_regul_ind_sum_ind",
        "nll_regul_ind_sum",
        # "nll_regul_pop_sum" & "nll_regul_all_sum" are not really relevant so far
        # (because priors for our population variables are NOT true bayesian priors)
        # "nll_regul_pop_sum",
        # "nll_regul_all_sum",
    )

    def __init__(self, *args, **kws):
        self._latent_pop_vars = set()
        self._latent_ind_vars = set()
        super().__init__(*args, **kws)

    def __len__(self):
        return super().__len__() + len(self.AUTOMATIC_VARS)

    def __iter__(self):
        return iter(tuple(self.data) + self.AUTOMATIC_VARS)

    def __setitem__(self, name: VariableName, var: VariableInterface) -> None:
        if name in self.FORBIDDEN_NAMES or name in self.AUTOMATIC_VARS:
            raise ValueError(f"Can not use the reserved name '{name}'")
        if name in self.data:
            raise ValueError(f"Can not reset the variable '{name}'")
        super().__setitem__(name, var)
        if isinstance(var, ModelParameter):
            self.update(var.suff_stats.dedicated_variables or {})
        if isinstance(var, LatentVariable):
            self.update(var.get_regularity_variables(name))
            if isinstance(var, PopulationLatentVariable):
                self._latent_pop_vars.add(name)
            else:
                self._latent_ind_vars.add(name)

    def __getitem__(self, name: VariableName) -> VariableInterface:
        if name in self.AUTOMATIC_VARS:
            return self._auto_vars[name]
        return super().__getitem__(name)

    @property
    def _auto_vars(self) -> dict[VariableName, LinkedVariable]:
        # TODO? add jacobian as well?
        d = dict(
            # nll_regul_pop_sum=LinkedVariable(
            #     Sum(
            #         *(
            #             f"nll_regul_{pop_var_name}"
            #             for pop_var_name in self._latent_pop_vars
            #         )
            #     )
            # ),
            nll_regul_ind_sum_ind=LinkedVariable(
                Sum(
                    *(
                        f"nll_regul_{ind_var_name}_ind"
                        for ind_var_name in self._latent_ind_vars
                    )
                )
            ),
            nll_regul_ind_sum=LinkedVariable(SumDim("nll_regul_ind_sum_ind")),
            # nll_regul_all_sum=LinkedVariable(
            #     Sum("nll_regul_pop_sum", "nll_regul_ind_sum")
            # ),
        )
        assert d.keys() == set(self.AUTOMATIC_VARS)
        return d
