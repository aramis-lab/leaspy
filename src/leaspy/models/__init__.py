from .base import BaseModel, ModelInterface
from .constant import ConstantModel
from .factory import ModelName, model_factory
from .joint import JointModel
from .linear import LinearModel
from .lme import LMEModel
from .logistic import LogisticModel
from .mcmc_saem_compatible import McmcSaemCompatibleModel
from .riemanian_manifold import RiemanianManifoldModel
from .settings import ModelSettings
from .shared_speed_logistic import SharedSpeedLogisticModel
from .stateful import StatefulModel
from .stateless import StatelessModel
from .time_reparametrized import TimeReparametrizedModel
from .mixture import MixtureModel

__all__ = [
    "ModelInterface",
    "ModelName",
    "McmcSaemCompatibleModel",
    "TimeReparametrizedModel",
    "BaseModel",
    "ConstantModel",
    "StatelessModel",
    "StatefulModel",
    "LMEModel",
    "model_factory",
    "ModelSettings",
    "RiemanianManifoldModel",
    "LogisticModel",
    "LinearModel",
    "SharedSpeedLogisticModel",
    "JointModel",
    "MixtureModel",
]


def __getattr__(name: str):
    # `LogisticMultivariateMixtureModel` was the name of `MixtureModel` up to leaspy 2.1
    if name == "LogisticMultivariateMixtureModel":
        import warnings

        warnings.warn(
            "`LogisticMultivariateMixtureModel` is deprecated and will be removed in a "
            "future release, use `MixtureModel` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return MixtureModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
