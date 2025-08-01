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
from leaspy.models.utilities import (
    compute_ind_param_std_from_suff_stats,
    compute_ind_param_mean_from_suff_stats_mixture,
    compute_ind_param_std_from_suff_stats_mixture,
    compute_ind_param_std_from_suff_stats_mixture_burn_in,
    compute_probs_from_state,
)
from leaspy.utils.functional import (
    Identity,
    Mean,
    Prod,
    NamedInputFunction,
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
from .utilities import compute_individual_parameter_std_from_sufficient_statistics

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
        """
        Compute variable value from a `state` exposing a dict-like interface: var_name -> values.

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
        """
        Get the names of the variables that the current variable directly depends on.

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
        """
        Get the names of the variables that the current variable directly depends on.

        Returns
        -------
        :obj:`frozenset` [ :class:`~leaspy.variables.specs.VariableName`] :
            The set of ancestors variable names.
        """
        return frozenset()

    def compute(self, state: VariableNameToValueMapping) -> Optional[VariableValue]:
        """
        Compute variable value from a `state` exposing a dict-like interface: var_name -> values.

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
        """
        Move the value to specified device (other variables never hold values so need for this method).

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
    """
    A convenient class to produce a function to collect sufficient stats that are existing 
    or dedicated variables (to be automatically created).

    Parameters
    ----------
    existing_variables : :obj:`tuple` of :class:`~leaspy.variables.specs.VariableName`, optional
        Names of existing variables that should be included when collecting statistics.
    dedicated_variables : :obj:`dict` [:class:`~leaspy.variables.specs.VariableName`, :class:`~leaspy.variables.specs.LinkedVariable`], optional
        Custom or derived variables that will be included in the collection process.

    """

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
        """
        Get the combined list of all variable names to be collected.

        Returns
        -------
        :obj:`tuple` of :class:`~leaspy.variables.specs.VariableName`
            Tuple containing both existing and dedicated variable names.
        """
        return self.existing_variables + tuple(self.dedicated_variables or ())

    def __call__(self, state: VariableNameToValueMapping) -> SuffStatsRW:
        """
        Collect sufficient statistics from a given state.

        Parameters
        ----------
        state : :class:`~leaspy.variables.specs.VariableNameToValueMapping`
            A mapping from variable names to their current values.

        Returns
        -------
        stats : :class:`~leaspy.variables.specs.SuffStatsRW`
            A dictionary of variable names and their corresponding values, for all variables
            defined in this collector.
        """
        return {k: state[k] for k in self.variables}


@dataclass(frozen=True)
class ModelParameter(IndepVariable):
    """
    Variable for model parameters with a maximization rule. This variable shouldn't 
    be sampled and it shouldn't be data, a hyperparameter or a linked variable.
    
    Parameters
    ----------
    shape : :obj:`tuple` of :obj:`int`
        Shape of the parameter tensor. It must be fixed and known in advance.
    suff_stats : :class:`~leaspy.variables.specs.Collect`
        A callable object that collects sufficient statistics required to compute the update.
    update_rule : :obj:`.typing.Callable` [..., :class:`~leaspy.variables.specs.VariableValue`]
        The symbolic update rule for this parameter, used during both burn-in and standard
        learning phase unless overridden by `update_rule_burn_in`. 
    update_rule_burn_in : :obj:`.typing.Callable` [..., :class:`~leaspy.variables.specs.VariableValue`] or None, optional
        An optional alternative update rule specifically used during the burn-in phase.
        If provided, it overrides `update_rule` during that phase.

    Attributes
    ----------
    _update_rule_parameters : :obj:`frozenset` of :class:`~leaspy.variables.specs.VariableName`
        Internal cache of variable names required by `update_rule`.
    _update_rule_burn_in_parameters : :obj:`frozenset` of :class:`~leaspy.variables.specs.VariableName` or None
        Internal cache of variable names required by `update_rule_burn_in`, if defined.
    fixed_shape : :obj:`bool` (class attribute)
        Indicates that this variable has a fixed shape (True by design).
    is_settable : :obj:`bool` (class attribute)
        Flags this variable as being settable externally (True by design).
    """

    shape: tuple[int, ...]
    suff_stats: Collect  # Callable[[VariablesValuesRO], SuffStatsRW]
    """
    The symbolic update functions will take variadic `suff_stats` values,
    in order to re-use NamedInputFunction logic: e.g. update_rule=Std('xi')

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
        """
        Validates and stores the keyword parameters required by the specified update rule.
        Parameters
        ----------
        update_method : :obj:`str`
            The name of the update method attribute to validate (either `"update_rule"` or
            `"update_rule_burn_in"`).

        Raises
        ------
        :exc:`LeaspyModelInputError`
            If the function associated with the `update_method` has:
            - Positional arguments
            - Unexpected keyword arguments not matching `suff_stats` variables or `'state'`
            - Any signature that cannot be parsed or is otherwise invalid
        """
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
        """
        Compute the updated value for the model parameter using a maximization step.

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
        """
        Smart automatic definition of `ModelParameter` when it is the mean 
        of Gaussian prior of a population latent variable.
        
        Parameters
        ----------
        population_variable_name : :class:`~leaspy.variables.specs.VariableName`
            Name of the population latent variable for which this is the prior mean.
        shape : :obj:`tuple` of :obj:`int`
            The shape of the model parameter (typically matching the variable's dimensionality).

        Returns
        -------
        :class:`~leaspy.variables.specs.ModelParameter`
            A new instance of `ModelParameter` configured as a prior mean.
        """
        return cls(
            shape,
            suff_stats=Collect(population_variable_name),
            update_rule=Identity(population_variable_name),
        )
    
    @classmethod
    def for_ind_mean(
        cls, individual_variable_name: VariableName, shape: tuple[int, ...]
    ):
        """
        Smart automatic definition of `ModelParameter` when it is the mean 
        of Gaussian prior of an individual latent variable.
        
        Parameters
        ----------
        individual_variable_name : :class:`~leaspy.variables.specs.VariableName`
            Name of the individual latent variable for which this is the prior mean.
        shape : :obj:`tuple` of :obj:`int`
            The shape of the model parameter (typically matching the variable's dimensionality).

        Returns
        -------
        :class:`~leaspy.variables.specs.ModelParameter`
            A new instance of `ModelParameter` configured as a prior mean.
        """
        return cls(
            shape,
            suff_stats=Collect(individual_variable_name),
            update_rule=Mean(individual_variable_name, dim=LVL_IND),
        )

    @classmethod
    def for_ind_mean_mixture(cls,
                             ind_var_name: VariableName ,shape: Tuple[int, ...],):
        """
        Smart automatic definition of `ModelParameter` when it is the mean of a mixture of Gaussians
        prior of an individual latent variable.
        Extra handling to keep one mean per cluster
        """
        update_rule_mixture = NamedInputFunction(
            compute_ind_param_mean_from_suff_stats_mixture,
            parameters = ("state",),
            kws=dict(ip_name = ind_var_name)
        )

        return cls(
            shape,
            suff_stats=Collect(ind_var_name),
            update_rule=update_rule_mixture,
        )

   
    @classmethod
    def for_ind_std(cls, ind_var_name: VariableName, shape: Tuple[int, ...], **tol_kw):
        """
        Smart automatic definition of `ModelParameter` when it is the std-dev of Gaussian
        prior of an individual latent variable.
        """
        ind_var_sqr_name = f"{ind_var_name}_sqr"
        update_rule_normal = NamedInputFunction(
            compute_individual_parameter_std_from_sufficient_statistics,
            parameters=(
                "state",
                ind_var_name,
                ind_var_sqr_name,
            ),
            kws=dict(
                individual_parameter_name=ind_var_name,
                dim=LVL_IND,
                **tol_kw,
            ),
        )
        return cls(
            shape,
            suff_stats=Collect(
                ind_var_name,
                **{
                    ind_var_sqr_name: LinkedVariable(
                        Sqr(ind_var_name)
                    )
                },
            ),
            update_rule_burn_in=Std(ind_var_name, dim=LVL_IND),
            update_rule=update_rule_normal,
        )

    @classmethod
    def for_ind_std_mixture(cls, ind_var_name: VariableName, shape: Tuple[int, ...], **tol_kw):
        """
        Smart automatic definition of `ModelParameter` when it is the std-dev of Gaussian
        prior of an individual latent variable.
        """
        ind_var_sqr_name = f"{ind_var_name}_sqr"
        update_rule_mixture = NamedInputFunction(
            compute_ind_param_std_from_suff_stats_mixture,
            parameters=("state", ind_var_name, ind_var_sqr_name),
            kws=dict(ip_name=ind_var_name, dim=LVL_IND, **tol_kw),
        )
        update_rule_mixture_burn_in =  NamedInputFunction(
            compute_ind_param_std_from_suff_stats_mixture_burn_in,
            parameters = ("state",),
            kws=dict(ip_name = ind_var_name)
        )
        return cls(
            shape,
            suff_stats=Collect(
                ind_var_name, **{ind_var_sqr_name: LinkedVariable(Sqr(ind_var_name))}
            ),
            update_rule_burn_in=update_rule_mixture_burn_in,
            update_rule=update_rule_mixture,
        )

    @classmethod
    def for_probs(cls, shape: Tuple[int, ...],):

        update_rule_probs = NamedInputFunction(
            compute_probs_from_state,
            parameters = ("state",),
        )

        return cls(
            shape,
            suff_stats= Collect(),
            update_rule= update_rule_probs,
        )
    
@dataclass(frozen=True)
class DataVariable(IndepVariable):
    """
    Variables for input data, that may be reset.
    
    Attributes
    ----------
    fixed_shape : :obj:`bool`
        Indicates whether the shape of the variable is fixed. For `DataVariable`.
        `False` by design, allowing for more flexible data injection.
    is_settable : :obj:`bool`
        Flag indicating whether this variable can be set/reset directly in the state.
        `True` by desdign, meaning it can be modified externally.
    """

    fixed_shape: ClassVar = False
    is_settable: ClassVar = True


class LatentVariableInitType(str, Enum):
    """
    Type of initialization for latent variables.

    Members
    -------
    PRIOR_MODE : :obj:`str`
        Initialize latent variables using the mode of their prior distribution.
    PRIOR_MEAN : :obj:`str`
        Initialize latent variables using the mean of their prior distribution.
    PRIOR_SAMPLES : :obj:`str`
        Initialize latent variables by sampling from their prior distribution.
    """

    PRIOR_MODE = "mode"
    PRIOR_MEAN = "mean"
    PRIOR_SAMPLES = "samples"


@dataclass(frozen=True)
class LatentVariable(IndepVariable):
    """
    Unobserved variable that will be sampled, with symbolic prior distribution.

    Attributes
    ----------
    prior : :class:`~leaspy.variables.distributions.SymbolicDistribution`
        The symbolic prior distribution for the latent variable (e.g. `Normal('xi_mean', 'xi_std')`).
    sampling_kws : :obj:`dict`, optional
        Optional keyword arguments to customize the sampling process (e.g. number of samples, random seed).
    
    Class Attributes
    ----------------
    is_settable : :obj:`bool`
        Indicates that this variable can be explicitly set in the model (default: True).    
    """

    # TODO/WIP? optional mask derive from optional masks of prior distribution parameters?
    # or should be fixed & explicit here?
    prior: SymbolicDistribution
    sampling_kws: Optional[KwargsType] = None

    is_settable: ClassVar = True

    def get_prior_shape(
        self, named_vars: TMapping[VariableName, VariableInterface]
    ) -> tuple[int, ...]:
        """
        Get shape of prior distribution (i.e. without any expansion for `IndividualLatentVariable`).
        
        Parameters
        ----------
        named_vars : :obj:`Mapping` [:class:`~leaspy.variables.specs.VariableName`, :class:`~leaspy.variables.specs.VariableInterface`]
            A mapping from variable names to their corresponding variable interfaces.
            These should include the parameters of the prior distribution.

        Returns
        -------
        :obj:`tuple` of :obj:`int`
            The shape of the prior distribution (without any replication for individual variables).

        Raises
        ------
        :exc:`LeaspyModelInputError`
            If any of the prior distribution’s parameter variables do not have a fixed shape.
        """
        bad_params = {
            n for n in self.prior.parameters_names if not named_vars[n].fixed_shape
        }
        if len(bad_params):
            raise LeaspyModelInputError(
                f"Shapes of some prior distribution parameters are not fixed: {bad_params}"
            )
        params_shapes = {n: named_vars[n].shape for n in self.prior.parameters_names}
        res = self.prior.shape(**params_shapes) # before it returned only this
        # some changes needed to handle the parameters in mixture,
        # it sampled with shape n_clusters for the individual latent parameters if we leave it as before
        # the correct sample.size is like in the classic model,
        # the latent individual variables do not have an extra dimension
        if 'Mixture' in str(self.prior):
            name = str([self.prior.parameters_names[0]])
            if 'sources' in name:
                shape_to_modif = self.prior.shape(**params_shapes)
                res = torch.Size(shape_to_modif [:1])
            if 'tau' in name:
                shape_to_modif = torch.Size([1])
                res = shape_to_modif
            if 'xi' in name:
                shape_to_modif = torch.Size([1])
                res = shape_to_modif
        return res

    def _get_init_func_generic(
        self,
        method: Union[str, LatentVariableInitType],
        *,
        sample_shape: tuple[int, ...],
    ) -> NamedInputFunction[torch.Tensor]:
        """
        Return a function that may be used for initialization.
        
        Parameters
        ----------
        method : :obj:`str` or :class:`~leaspy.variables.specs.LatentVariableInitType`
            Initialization method. Must be one of `'samples'`, `'mode'`, or `'mean'`.
        sample_shape : :obj:`tuple` of :obj:`int`
            The shape to prepend to the initialized tensor (i.e., left expansion).

        Returns
        -------
        :class:`~leaspy.utils.functional._named_input_function.NamedInputFunction`[:class:`torch.Tensor`]
            A symbolic function to compute the initial value tensor.

        Raises
        ------
        :exc:`ValueError`
            If `method` is not one of the allowed values.
        """
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
    """
    Population latent variable.
    
    Attributes
    ----------
    fixed_shape : `ClassVar`[:obj:`bool`]
        Indicates that the shape is fixed (True).
    """

    # not so easy to guarantee the fixed shape property in fact...
    # (it requires that parameters of prior distribution all have fixed shapes)
    fixed_shape: ClassVar = True

    def get_init_func(
        self,
        method: Union[str, LatentVariableInitType],
    ) -> NamedInputFunction[torch.Tensor]:
        """
        Return a function that may be used for initialization.

        Parameters
        ----------
        method : :class:`~leaspy.variables.specs.LatentVariableInitType` or :obj:`str`
            The method to be used.

        Returns
        -------
        :class:`~leaspy.utils.functional._named_imput_function.NamedInputFunction`[:class:`torch.Tensor`] :
            The initialization function.
        """
        return self._get_init_func_generic(method=method, sample_shape=())

    def get_regularity_variables(
        self,
        variable_name: VariableName,
    ) -> dict[VariableName, LinkedVariable]:
        """
        Return the negative log likelihood regularity for the provided variable name.

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
    """
    Individual latent variable.
    
    Attributes
    ----------
    fixed_shape : `ClassVar`[:obj:`bool`]
        Indicates that the shape is fixed (True).
    """

    fixed_shape: ClassVar = False

    def get_init_func(
        self,
        method: Union[str, LatentVariableInitType],
        *,
        n_individuals: int,
    ) -> NamedInputFunction[torch.Tensor]:
        """
        Return a function that may be used for initialization.

        Parameters
        ----------
        method : :class:`~leaspy.variables.specs.LatentVariableInitType` or :obj:`str`
            The method to be used.
        n_individuals : :obj:`int`
            The number of individuals, used to define the shape.

        Returns
        -------
        :class:`~leaspy.utils.functional._named_imput_function.NamedInputFunction`[:class:`torch.Tensor`] :
            The initialization function.
        """
        return self._get_init_func_generic(method=method, sample_shape=(n_individuals,))

    def get_regularity_variables(
        self,
        variable_name: VariableName,
    ) -> dict[VariableName, LinkedVariable]:
        """
        Return the negative log likelihood regularity for the provided variable name.

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
        if 'Mixture' in str(self.prior): #specification for the mixture model : we don't want to sum all dimensions, we need one regularity per cluster
            if variable_name == 'sources' :
                d.update(
                    {
                        f"nll_regul_{variable_name}_ind": LinkedVariable(
                            self.prior.get_func_regularization(variable_name).then(
                                sum_dim, but_dim=(LVL_IND, 2) # sum per source but omit the cluster dimension as well
                            )
                        ),
                        f"nll_regul_{variable_name}": LinkedVariable(
                            SumDim(f"nll_regul_{variable_name}_ind")
                        ),
                    }
                )
            else :
                d.update(
                    {
                        f"nll_regul_{variable_name}_ind": LinkedVariable(
                            self.prior.get_func_regularization(variable_name)
                        ), # keep it per cluster dont sum all dimensions
                        f"nll_regul_{variable_name}": LinkedVariable(
                            SumDim(f"nll_regul_{variable_name}_ind")
                        ),
                    }
                )
        else:
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
    """
    Variable which is a deterministic expression of other variables 
    (we directly use variables names instead of mappings: kws <-> vars).
    
    Parameters
    ----------
    f : :obj:`Callable`[..., :class:`~leaspy.variables.specs.VariableValue`]
        A deterministic function that computes this variable's value from its input variables.
        The function should accept keyword arguments matching the variable names in `parameters`.

    Attributes
    ----------
    parameters : :obj:`frozenset`[:class:`~leaspy.variables.specs.VariableName`]
        The set of variable names on which this linked variable depends.
        This is inferred internally from the function `f`.
    is_settable : `ClassVar`[:obj:`bool`]
        Indicates that this variable is not settable directly (`False`).
    fixed_shape : `ClassVar`[obj:`bool`]
        Indicates whether the shape of the linked variable is fixed.
        By design it is `False`.
    """

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
        """
        Return the set of variable names that this linked variable depends on.

        Returns
        -------
        :obj:`frozenset`[:class:`~leaspy.variables.specs.VariableName`]
            The names of ancestor variables used as inputs by this linked variable.
        """
        return self.parameters

    def compute(self, state: VariableNameToValueMapping) -> VariableValue:
        """
        Compute the variable value from a given State.

        Parameters
        ----------
        state : :class:`~leaspy.variables.specs.VariableNameToValueMapping`
            The state to use for computations.

        Returns
        -------
        :class:`~leaspy.variables.specs.VariableValue` :
            The value of the variable.
        """
        return self.f(**{k: state[k] for k in self.parameters})


class NamedVariables(UserDict):
    """
    Convenient dictionary for named variables specifications.

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
