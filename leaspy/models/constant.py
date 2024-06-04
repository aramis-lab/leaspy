import torch

from leaspy.models.generic import GenericModel
from leaspy.exceptions import LeaspyModelInputError

from leaspy.utils.docs import doc_with_super


@doc_with_super()
class ConstantModel(GenericModel):
    r"""
    `ConstantModel` is a benchmark model that predicts constant values (no matter what the patient's ages are).

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
    **kwargs
        Hyperparameters for the model.
        None supported for now.

    Attributes
    ----------
    name : :obj:`str`
        The model's name.
    is_initialized : :obj:`bool`
        Always ``True`` (no true initialization needed for constant model).
    features : :obj:`list` of :obj:`str`
        List of the model features.
        Unlike most models features will be determined at :term:`personalization`
        only (because it does not needed any `fit`).
    dimension : :obj:`int`
        Number of features (read-only).
    parameters : :obj:`dict`
        The model has no parameters: empty dictionary.
        The ``prediction_type`` parameter should be defined during
        :term:`personalization`.
        Example:
            >>> AlgorithmSettings('constant_prediction', prediction_type='last_known')

    See Also
    --------
    :class:`~leaspy.algo.others.constant_prediction_algo.ConstantPredictionAlgorithm`
    """

    def __init__(self, name: str, **kwargs):

        super().__init__(name, **kwargs)

        # no fit algorithm is needed for constant model; every "personalization" will re-initialize model
        # however, we need to mock that model is personalization-ready by setting self.is_initialized (API requirement)
        self.is_initialized = True

    def compute_individual_trajectory(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
    ) -> torch.Tensor:
        if self.features is None:
            raise LeaspyModelInputError('The model was not properly initialized.')
        values = [individual_parameters[f] for f in self.features]
        return torch.tensor([[values] * len(timepoints)], dtype=torch.float32)
