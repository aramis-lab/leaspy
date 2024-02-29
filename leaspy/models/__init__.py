from .abstract_model import AbstractModel
from .abstract_multivariate_model import AbstractMultivariateModel
from .base import BaseModel
from .constant import ConstantModel
from .multivariate import MultivariateModel, LogisticMultivariateModel, LinearMultivariateModel
from .multivariate_parallel import MultivariateParallelModel
from .univariate_joint import UnivariateJointModel
from .univariate import LinearUnivariateModel, LogisticUnivariateModel


ALL_MODELS = {
    "univariate_joint": UnivariateJointModel,
    "univariate_logistic": LogisticUnivariateModel,
    "univariate_linear": LinearUnivariateModel,
    "logistic": LogisticMultivariateModel,
    "linear": LinearMultivariateModel,
    "logistic_parallel": MultivariateParallelModel,
    "constant": ConstantModel,
}

from .factory import ModelFactory  # noqa


__all__ = [
    "ALL_MODELS",
    "AbstractModel",
    "AbstractMultivariateModel",
    "BaseModel",
    "ConstantModel",
    "ModelFactory",
    "MultivariateModel",
    "LogisticMultivariateModel",
    "LinearMultivariateModel",
    "MultivariateParallelModel",
    "LinearUnivariateModel",
    "LogisticUnivariateModel",
    "UnivariateJointModel",
]
