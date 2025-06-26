"""`ObservationModel` defines the common interface for observation models in Leaspy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Optional,
)
from typing import (
    Mapping as TMapping,
)

from leaspy.io.data.dataset import Dataset
from leaspy.utils.functional import SumDim
from leaspy.utils.weighted_tensor import WeightedTensor, sum_dim
from leaspy.variables.distributions import SymbolicDistribution
from leaspy.variables.specs import (
    LVL_IND,
    DataVariable,
    LinkedVariable,
    VariableInterface,
    VariableName,
)

__all__ = ["ObservationModel"]


@dataclass(frozen=True)
class ObservationModel:
    """
    Base class for valid observation models that may be used in probabilistic models (stateless).

    In particular, it provides data & linked variables regarding observations and their attachment to the model
    (the negative log-likelihood - nll - to be minimized).

    Parameters
    ----------
    name : :obj:`str`
        The name of observed variable (to name the data variable & attachment term related to this observation).
    getter : function :class:`.Dataset` -> :class:`.WeightedTensor`
        The way to retrieve the observed values from the :class:`.Dataset` (as a :class:`.WeightedTensor`):
        e.g. all values, subset of values - only x, y, z features, one-hot encoded features, ...
    dist : :class:`.SymbolicDistribution`
        The symbolic distribution, parametrized by model variables, for observed values (so to compute attachment).
    extra_vars : None (default) or Mapping[VarName, :class:`.VariableInterface`]
        Some new variables that are needed to fully define the symbolic distribution or the sufficient statistics.
        (e.g. "noise_std", and "y_L2_per_ft" for instance for a Gaussian model)
    """

    name: VariableName
    getter: Callable[[Dataset], WeightedTensor]
    dist: SymbolicDistribution
    extra_vars: Optional[TMapping[VariableName, VariableInterface]] = None

    def get_nll_attach_var_name(self, named_attach_vars: bool = True) -> str:
        """
        Return the name of the negative log likelihood attachement
        variable.
        """
        return f"nll_attach_{self.name}" if named_attach_vars else "nll_attach"

    def get_variables_specs(
        self,
        named_attach_vars: bool = True,
    ) -> dict[VariableName, VariableInterface]:
        """
        Automatic specifications of variables for this observation model.

        Parameters
        ----------
        named_attached_vars : bool, optional

        Returns
        -------
        dict of VariableName to Variable Interface
            A dictionary mapping variable name to their correspondind specifications with
            - the primary DaraVariable
            - any `extra_vars` defined by the model
            - nll attachment variables :
                - nll_attach_var_ind: a `LinkedVariable` representing the individual-level
                negative log-likelihood contributions
                - nll_attach_var: a `LinkedVariable` that sums the individual contributions

        Notes
        -----
        The distribution object `self.dist`should provide a `get_func_nll(name)` method that
        returns a callable for computing the nll
        """
        # TODO change? a bit dirty? possibility of having aliases for variables?

        nll_attach_var = self.get_nll_attach_var_name(named_attach_vars)

        return {
            self.name: DataVariable(),
            # Dependent vars
            **(self.extra_vars or {}),
            # Attachment variables
            # not really memory efficient nor useful...
            # f"{nll_attach_var}_full": LinkedVariable(self.dist.get_func_nll(self.name)),
            f"{nll_attach_var}_ind": LinkedVariable(
                # SumDim(f"{nll_attach_var}_full", but_dim=LVL_IND)
                self.dist.get_func_nll(self.name).then(sum_dim, but_dim=LVL_IND)
            ),
            nll_attach_var: LinkedVariable(SumDim(f"{nll_attach_var}_ind")),
            # TODO jacobian of {nll_attach_var}_ind_jacobian_{self.name} wrt "y" as well? (for scipy minimize)
        }

    def serialized(self) -> Any:
        """
        Returns a JSON-exportable representation of the instance, excluding its name.

        Returns
        -------
        Any
            A representation of the instance, currently based on `repr(self.dist)`, 
            that is intended to be JSON-serializable.
        """
        # TODO: dirty for now to go fast
        return repr(self.dist)

    def to_dict(self) -> dict:
        """To be implemented..."""
        return {}

    def to_string(self) -> str:
        """
        Returns a string representation of the parameter for saving

        Returns
        -------
        str
            A string representation of the parameter, as stored in `self.string_for_json`.
        """
        return self.string_for_json
