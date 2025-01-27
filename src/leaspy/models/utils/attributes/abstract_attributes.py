from abc import ABC, abstractmethod

import torch

from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.typing import DictParamsTorch, ParamType, Set, Tuple


class AbstractAttributes(ABC):
    """
    Abstract base class for attributes of models.

    Contains the common attributes & methods of the different attributes classes.
    Such classes are used to update the models' attributes.

    Parameters
    ----------
    name : str
    dimension : int (default None)
    source_dimension : int (default None)

    Attributes
    ----------
    name : str
        Name of the associated leaspy model.
    dimension : int
        Number of features of the model
    source_dimension : int
        Number of sources of the model
        TODO? move to AbstractManifoldModelAttributes?
    univariate : bool
        Whether model is univariate or not (i.e. dimension == 1)
    has_sources : bool
        Whether model has sources or not (not univariate and source_dimension >= 1)
        TODO? move to AbstractManifoldModelAttributes?
    update_possibilities : set[str] (default empty)
        Contains the available parameters to update. Different models have different parameters.

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        if any inconsistent parameter.
    """

    def __init__(self, name: str, dimension: int, source_dimension: int):
        if not (isinstance(name, str) and len(name)):
            raise LeaspyModelInputError(
                "In model attributes, you must provide a non-empty string for the parameter `name`."
            )
        self.name = name

        if not (isinstance(dimension, int) and dimension >= 1):
            raise LeaspyModelInputError(
                "In model attributes, you must provide an integer >= 1 for the parameter `dimension`."
            )
        self.dimension = dimension
        self.univariate = dimension == 1

        self.source_dimension = source_dimension
        self.has_sources = bool(source_dimension)

        if self.univariate and self.has_sources:
            raise LeaspyModelInputError(
                "Inconsistent attributes: presence of sources for a univariate model."
            )

        if not (
            isinstance(source_dimension, int) and (0 <= source_dimension < dimension)
        ):
            raise LeaspyModelInputError(
                "In model attributes, you must provide an integer in [0, dimension - 1] for the parameter `source_dimension`."
            )

        self.update_possibilities: Set[ParamType] = set()

    @abstractmethod
    def get_attributes(
        self,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Returns the attributes of the model, which is a tuple of three torch tensors.

        For the precise definitions of those attributes please refer to the exact
        attributes class associated to your model.

        Returns
        -------
        positions: `torch.Tensor`
        velocities: `torch.Tensor`
        mixing_matrix: `torch.Tensor`
        """

    @abstractmethod
    def update(
        self, names_of_changed_values: Set[ParamType], values: DictParamsTorch
    ) -> None:
        """
        Update model group average parameter(s).

        Parameters
        ----------
        names_of_changed_values : set[str]
           Values to be updated
        values : dict [str, `torch.Tensor`]
           New values used to update the model's group average parameters

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If `names_of_changed_values` contains unknown values to update.
        """

    def move_to_device(self, device: torch.device):
        """
        Move the tensor attributes of this class to the specified device.

        Parameters
        ----------
        device : torch.device
        """
        for attribute_name in dir(self):
            if attribute_name.startswith("__"):
                continue
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, torch.Tensor):
                setattr(self, attribute_name, attribute.to(device))

    def _check_names(self, names_of_changed_values: Set[ParamType]):
        """
        Check if the name of the parameter(s) to update are in the possibilities allowed by the model.

        Parameters
        ----------
        names_of_changed_values : set[str]

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If `names_of_changed_values` contains unknown values to update.
        """
        unknown_update_possibilities = names_of_changed_values.difference(
            self.update_possibilities
        )
        if len(unknown_update_possibilities):
            raise LeaspyModelInputError(
                f"{unknown_update_possibilities} not in the attributes that can be updated"
            )
