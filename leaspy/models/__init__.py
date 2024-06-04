from .abstract_model import AbstractModel
from .abstract_multivariate_model import AbstractMultivariateModel
from .base import BaseModel
from .constant import ConstantModel
from .generic import GenericModel
from .lme import LMEModel
from .multivariate import MultivariateModel
from .multivariate_parallel import MultivariateParallelModel
from .univariate import UnivariateModel


# flexible dictionary to have a simpler and more maintainable ModelFactory
ALL_MODELS = {
    # univariate Leaspy models
    'univariate_logistic': UnivariateModel,
    'univariate_linear': UnivariateModel,

    # multivariate Leaspy models
    'logistic': MultivariateModel,
    'linear': MultivariateModel,
    'mixed_linear-logistic': MultivariateModel,
    'logistic_parallel': MultivariateParallelModel,

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
    "MultivariateParallelModel",
    "UnivariateModel",
]
