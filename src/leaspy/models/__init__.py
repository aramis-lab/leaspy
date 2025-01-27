from .abstract_model import AbstractModel
from .abstract_multivariate_model import AbstractMultivariateModel
from .base import BaseModel
from .constant import ConstantModel
from .generic import GenericModel
from .joint import JointModel
from .lme import LMEModel
from .multivariate import (
    LinearMultivariateModel,
    LogisticMultivariateModel,
    MultivariateModel,
)
from .multivariate_parallel import MultivariateParallelModel
from .ordinal_multivariate_model import OrdinalMultivariateModel
from .univariate import LinearUnivariateModel, LogisticUnivariateModel
from .univariate_joint import UnivariateJointModel

# flexible dictionary to have a simpler and more maintainable ModelFactory
ALL_MODELS = {
    "univariate_joint": UnivariateJointModel,
    "joint": JointModel,
    "univariate_logistic": LogisticUnivariateModel,
    "univariate_linear": LinearUnivariateModel,
    "logistic": LogisticMultivariateModel,
    "logistic_ordinal": OrdinalMultivariateModel,
    "linear": LinearMultivariateModel,
    # 'mixed_linear-logistic': MultivariateModel,
    "logistic_parallel": MultivariateParallelModel,
    # naive models (for benchmarks)
    "lme": LMEModel,
    "constant": ConstantModel,
}


from .factory import ModelFactory  # noqa


__all__ = [
    "ALL_MODELS",
    "AbstractModel",
    "AbstractMultivariateModel",
    "BaseModel",
    "ConstantModel",
    "GenericModel",
    "LMEModel",
    "ModelFactory",
    "MultivariateModel",
    "LogisticMultivariateModel",
    "LinearMultivariateModel",
    "MultivariateParallelModel",
    "LinearUnivariateModel",
    "LogisticUnivariateModel",
    "UnivariateJointModel",
    "JointModel",
    "OrdinalMultivariateModel",
]
