import torch

from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.typing import DictParamsTorch

from .stateless import StatelessModel

__all__ = ["ConstantModel"]


class ConstantModel(StatelessModel):
    r"""ConstantModel` is a benchmark model that predicts constant values (no matter what the patient's ages are).

    These constant values depend on the algorithm setting and the patient's values
    provided during :term:`calibration`.

    It could predict:
        * ``last``: last value seen during calibration (even if ``NaN``).
        * ``last_known``: last non ``NaN`` value seen during :term:`calibration`.
        * ``max``: maximum (=worst) value seen during :term:`calibration`.
        * ``mean``: average of values seen during :term:`calibration`.

    .. warning::
        Depending on ``features``, the ``last_known`` / ``max`` value
        may correspond to different visits.

    .. warning::
        For a given feature, value will be ``NaN`` if and only if all
        values for this feature were ``NaN``.

    Parameters
    ----------
    name : :obj:`str`
        The model's name.

    See Also
    --------
    :class:`~leaspy.algo.others.constant_prediction_algo.ConstantPredictionAlgorithm`
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.is_initialized = True

    @property
    def hyperparameters(self) -> DictParamsTorch:
        """Dictionary of values for model hyperparameters."""
        return {}

    def compute_individual_trajectory(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
    ) -> torch.Tensor:
        if self.features is None:
            raise LeaspyModelInputError("The model was not properly initialized.")
        values = [individual_parameters[f] for f in self.features]
        return torch.tensor([[values] * len(timepoints)], dtype=torch.float32)
