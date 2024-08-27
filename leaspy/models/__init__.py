from .abstract_model import AbstractModel
from .abstract_multivariate_model import AbstractMultivariateModel
from .base import BaseModel
from .constant import ConstantModel
from .generic import GenericModel
from .lme import LMEModel
from .multivariate import MultivariateModel, LogisticMultivariateModel, LinearMultivariateModel
from .multivariate_parallel import MultivariateParallelModel
from .univariate_joint import UnivariateJointModel
from .joint import JointModel
from .univariate import LinearUnivariateModel, LogisticUnivariateModel


# flexible dictionary to have a simpler and more maintainable ModelFactory
ALL_MODELS = {
    "univariate_joint": UnivariateJointModel,
    "joint": JointModel,
    "univariate_logistic": LogisticUnivariateModel,
    "univariate_linear": LinearUnivariateModel,
    "logistic": LogisticMultivariateModel,
    "linear": LinearMultivariateModel,
    # 'mixed_linear-logistic': MultivariateModel,
    "logistic_parallel": MultivariateParallelModel,

    # naive models (for benchmarks)
    'lme': LMEModel,
    'constant': ConstantModel,
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
]
