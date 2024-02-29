import torch
import warnings

from leaspy.models import BaseModel
from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.typing import DictParams, KwargsType, Tuple, DictParamsTorch, Any
from leaspy.variables.specs import VarName
from leaspy.utils.docs import doc_with_super


@doc_with_super()
class ConstantModel(BaseModel):
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

    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._parameters: DictParamsTorch = {}
        self._hyperparameters: DictParamsTorch = {}
        self.is_initialized = True

    @property
    def parameters(self) -> DictParamsTorch:
        return self._parameters

    @property
    def parameters_names(self) -> Tuple[VarName, ...]:
        return tuple(self.parameters.keys())

    @property
    def hyperparameters_names(self) -> Tuple[VarName, ...]:
        return tuple(self.hyperparameters.keys())

    @property
    def hyperparameters(self) -> DictParamsTorch:
        return self._hyperparameters

    def set_parameter(self, name: str, value: Any) -> None:
        if name in self._parameters:
            warnings.warn(
                f"Parameter {name} was already set in model with value {self._parameters[name]}. "
                f"Resetting it with new value {value}."
            )
        self._parameters[name] = value

    def load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        for name, value in hyperparameters.items():
            self.set_hyperparameter(name, value)

    def set_hyperparameter(self, name: str, value: Any) -> None:
        if name in self._hyperparameters:
            warnings.warn(
                f"Hyperparameter {name} was already set in model with value {self._hyperparameters[name]}."
                f"Resetting it with new value {value}."
            )
        self._hyperparameters[name] = value

    def compute_individual_trajectory(
        self,
        timepoints,
        individual_parameters: DictParams,
        *,
        skip_ips_checks: bool = False,
    ) -> torch.Tensor:
        if self.features is None:
            raise LeaspyModelInputError('The model was not properly initialized.')
        values = [individual_parameters[f] for f in self.features]
        return torch.tensor([[values] * len(timepoints)], dtype=torch.float32)

    def __str__(self):
        lines = [f"=== MODEL {self.name} ==="]
        for hp_name, hp_val in self.hyperparameters.items():
            lines.append(f"{hp_name} : {hp_val}")
        lines.append('-'*len(lines[0]))
        for param_name, param_val in self.parameters.items():
            lines.append(f"{param_name} : {param_val}")

        return "\n".join(lines)
