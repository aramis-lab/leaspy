from enum import Enum
from typing import Optional, Union

from .base import BaseModel
from .constant import ConstantModel
from .joint import JointModel
from .linear import LinearModel
from .lme import LMEModel
from .logistic import LogisticModel
from .mixture import LogisticMultivariateMixtureModel
from .shared_speed_logistic import SharedSpeedLogisticModel

__all__ = [
    "ModelName",
    "model_factory",
]


class ModelName(str, Enum):
    """The available models that users can instantiate in Leaspy."""

    JOINT = "joint"
    LOGISTIC = "logistic"
    LINEAR = "linear"
    SHARED_SPEED_LOGISTIC = "shared_speed_logistic"
    LME = "lme"
    CONSTANT = "constant"
    MIXTURE_LOGISTIC = "mixture_logistic"


def model_factory(
    name: Union[str, ModelName], instance_name: Optional[str] = None, **kwargs
) -> BaseModel:
    """
    Return the model object corresponding to ``name`` arg with possible ``kwargs``.

    Parameters
    ----------
    name : :obj:`str` or ModelName
        The name of the model class to be instantiated. Valid options include:
            - ``"joint"``
            - ``"logistic"``
            - ``"linear"``
            - ``"shared_speed_logistic"``
            - ``"lme"``
            - ``"constant"``
            - ``"mixture_logistic"``

    instance_name : :obj:`str`, optional
        A custom name for the model instance. If not provided, the model's name
        will be used as the instance name.

    **kwargs
        Additional keyword arguments corresponding to the model's hyperparameters.
        These must be valid for the specified model, or an error will be raised.

    Returns
    -------
    :class:`~leaspy.models.base.BaseModel`
        A child class object of :class:`~leaspy.models.base.BaseModel` class object determined by ``name``.

    Raises
    ------
    ValueError
        If an invalid model name is provided or the model cannot be instantiated
        with the provided arguments.
    """
    name = ModelName(name)
    instance_name = instance_name or name.value
    if name == ModelName.JOINT:
        return JointModel(instance_name, **kwargs)
    if name == ModelName.LOGISTIC:
        return LogisticModel(instance_name, **kwargs)
    if name == ModelName.LINEAR:
        return LinearModel(instance_name, **kwargs)
    if name == ModelName.SHARED_SPEED_LOGISTIC:
        return SharedSpeedLogisticModel(instance_name, **kwargs)
    if name == ModelName.LME:
        return LMEModel(instance_name, **kwargs)
    if name == ModelName.CONSTANT:
        return ConstantModel(instance_name, **kwargs)
    if name == ModelName.MIXTURE_LOGISTIC:
        return LogisticMultivariateMixtureModel(instance_name, **kwargs)
