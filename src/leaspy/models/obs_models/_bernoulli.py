import torch

from leaspy.io.data.dataset import Dataset
from leaspy.utils.weighted_tensor import WeightedTensor
from leaspy.variables.distributions import Bernoulli
from leaspy.variables.specs import VariableInterface

from ._base import ObservationModel

__all__ = ["BernoulliObservationModel"]


class BernoulliObservationModel(ObservationModel):
    """
    Observation model for binary outcomes using a Bernoulli distribution.

    This model expects binary-valued observations and uses a Bernoulli distribution
    to define the likelihood. It assumes the response variable is named `"y"`.

    Parameters
    ----------
    **extra_vars : VariableInterface
        Optional extra variables required by the model. These are passed to the
        parent `ObservationModel` class and can be used for conditioning the likelihood.

    Attributes
    ----------
    string_for_json : :obj:`str`
        A static string identifier used for serialization.
    """
    string_for_json = "bernoulli"

    def __init__(
        self,
        **extra_vars: VariableInterface,
    ):
        super().__init__(
            name="y",
            getter=self.y_getter,
            dist=Bernoulli("model"),
            extra_vars=extra_vars,
        )

    @staticmethod
    def y_getter(dataset: Dataset) -> WeightedTensor:
        """
        Extracts and validates the observation values and associated mask from a dataset.

        Parameters
        ----------
        dataset : Dataset
            A dataset object containing `values` and `mask` attributes.

        Returns
        -------
        WeightedTensor
            A tensor containing the observed binary values along with a boolean mask
            indicating which entries are valid.

        Raises
        ------
        ValueError
            If either `dataset.values` or `dataset.mask` is `None`, indicating that
            the dataset is improperly initialized.
        """
        if dataset.values is None or dataset.mask is None:
            raise ValueError(
                "Provided dataset is not valid. "
                "Both values and mask should be not None."
            )
        return WeightedTensor(dataset.values, weight=dataset.mask.to(torch.bool))
