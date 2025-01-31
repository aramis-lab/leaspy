from __future__ import annotations

import copy
from typing import Dict, List, NoReturn, Optional, Union, overload

import torch

from leaspy.exceptions import LeaspyInputError
from leaspy.utils.typing import DictParamsTorch

from .realization import AbstractRealization

__all__ = ["DictRealizations"]


class DictRealizations:
    """
    Dictionary of abstract realizations providing an easy-to-use interface.

    Parameters
    ----------
    realizations : Dict[str, AbstractRealization], optional
        The dictionary of realizations (empty if None).

    Attributes
    ----------
    realizations_dict : Dict[str, AbstractRealization]
    """

    def __init__(self, realizations: Optional[Dict[str, AbstractRealization]] = None):
        self.realizations_dict = realizations or {}

    @property
    def names(self) -> List[str]:
        """
        Return the list of variable names.
        """
        return list(self.realizations_dict.keys())

    @property
    def tensors(self) -> List[torch.Tensor]:
        return [realization.tensor for realization in self.realizations]

    @property
    def tensors_dict(self) -> DictParamsTorch:
        return {
            name: realization.tensor
            for name, realization in self.realizations_dict.items()
        }

    @property
    def realizations(self) -> List[AbstractRealization]:
        return list(self.realizations_dict.values())

    @overload
    def __getitem__(self, variable_name: str) -> AbstractRealization: ...

    @overload
    def __getitem__(self, variable_name: List[str]) -> DictRealizations: ...

    def __getitem__(
        self,
        variable_name: Union[str, List[str]],
    ) -> Union[AbstractRealization, DictRealizations]:
        if isinstance(variable_name, str):
            return self._get_realization_by_name(variable_name)
        if isinstance(variable_name, list):
            return DictRealizations(
                {name: self._get_realization_by_name(name) for name in variable_name}
            )
        self._raise_access_error(variable_name)

    def _get_realization_by_name(self, variable_name: str) -> AbstractRealization:
        try:
            return self.realizations_dict[variable_name]
        except KeyError:
            self._raise_access_error(variable_name)

    def _raise_access_error(self, variable_name: str) -> NoReturn:
        raise LeaspyInputError(
            "There are no realization "
            f"matching the provided name {variable_name}. "
            f"Available names are {self.names}."
        )

    def __setitem__(self, variable_name: str, realization: AbstractRealization):
        self.realizations_dict[variable_name] = realization

    def clone(self):
        """
        Deep-copy of the CollectionRealization instance.

        In particular the underlying realizations are cloned and detached.

        Returns
        -------
        CollectionRealization :
            The cloned collection of realizations.
        """
        return DictRealizations(
            {name: copy.deepcopy(r) for name, r in self.realizations_dict.items()},
        )
