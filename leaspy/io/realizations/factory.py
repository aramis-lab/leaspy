from .realization import (
    AbstractRealization,
    IndividualRealization,
    PopulationRealization,
)
from typing import Union
from enum import Enum, auto
from leaspy.exceptions import LeaspyInputError


class VariableType(Enum):
    """
    Possible types for variables.
    """
    INDIVIDUAL = auto()
    POPULATION = auto()

    @classmethod
    def from_string(cls, variable_type: str):
        variable_type = variable_type.lower().strip()
        if variable_type in {"population", "pop"}:
            return cls.POPULATION
        if variable_type in {"individual", "ind"}:
            return cls.INDIVIDUAL
        raise LeaspyInputError(
            f"Invalid variable type {variable_type}"
            f"Possible values are {list(VariableType)}"
        )


RealizationFactoryInput = Union[AbstractRealization, VariableType, str]


def realization_factory(realization_or_variable_type: RealizationFactoryInput, **kws) -> AbstractRealization:
    """
    Factory for Realizations.

    Parameters
    ----------
    realization_or_variable_type : AbstractRealization or VariableType or str
        If an instance of a subclass of AbstractRealization, returns the instance (no copy).
        If a VariableType variant (or a valid string keyword for this variant), then returns a new instance of the
        appropriate class (with optional parameters `kws`).

    **kws
        Optional parameters for initializing the requested Realization (not used if input is a subclass of AbstractRealization).

    Returns
    -------
    AbstractRealization :
        The desired realization.

    Raises
    ------
    LeaspyInputError:
        If the variable type provided is not supported.
    """
    if isinstance(realization_or_variable_type, AbstractRealization):
        return realization_or_variable_type
    if isinstance(realization_or_variable_type, str):
        realization_or_variable_type = VariableType.from_string(realization_or_variable_type)
    return _realization_as_variable_type_factory(realization_or_variable_type, **kws)


def _realization_as_variable_type_factory(variable_type: VariableType, **kws) -> AbstractRealization:
    if variable_type == VariableType.POPULATION:
        return PopulationRealization(**kws)
    if variable_type == VariableType.INDIVIDUAL:
        return IndividualRealization(**kws)
