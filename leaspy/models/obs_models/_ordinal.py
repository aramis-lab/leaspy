import torch

from leaspy.io.data.dataset import Dataset
from leaspy.utils.weighted_tensor import WeightedTensor
from leaspy.variables.specs import VariableInterface
from leaspy.variables.distributions import Ordinal

from ._base import ObservationModel


class OrdinalObservationModel(ObservationModel):
    max_level: int
    max_levels: dict[str, int]
    _mask: torch.Tensor
    string_for_json = 'ordinal'

    @property
    def ordinal_infos(self) -> dict:
        """Property to return the ordinal info dictionary."""
        # Maybe not put all ordinal infos in the ObservationModel but in the model itself
        return dict(
            max_level= self.max_level,
            max_levels= self.max_levels,
            mask= self._mask,
        )

    def __init__(
        self,
        **extra_vars: VariableInterface,
    ):
        super().__init__(
            name="y",
            getter=self.y_getter,
            dist=Ordinal("model"),
            extra_vars=extra_vars,
        )

    def y_getter(self, dataset: Dataset) -> WeightedTensor:
        if dataset.values is None or dataset.mask is None:
            raise ValueError(
                "Provided dataset is not valid. "
                "Both values and mask should be not None."
            )
        pdf = dataset.get_one_hot_encoding(sf=False, ordinal_infos=self.ordinal_infos)
        mask = torch.ones_like(pdf)
        mask[..., 1:] = self._mask #Add +1 on last dimension for level 0
        return WeightedTensor(pdf, weight=dataset.mask.to(torch.bool).unsqueeze(-1) * mask)
