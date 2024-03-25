from leaspy.variables.distributions import WeibullRightCensored
from leaspy.utils.weighted_tensor import EventTensor
from leaspy.io.data.dataset import Dataset

from ._base import ObservationModel
from leaspy.variables.specs import (
    VarName,
    VariableInterface,
    LinkedVariable,
)
from typing import Dict


class WeibullRightCensoredObservationModel(ObservationModel):
    string_for_json = "weibull-right-censored"

    def __init__(
        self,
        nu: VarName,
        rho: VarName,
        xi: VarName,
        tau: VarName,
        **extra_vars: VariableInterface,
    ):
        super().__init__(
            name="event",
            getter=self.getter,
            dist=WeibullRightCensored(nu, rho, xi, tau),
            extra_vars=extra_vars,
        )

    @staticmethod
    def getter(dataset: Dataset) -> EventTensor:
        if dataset.event_time is None or dataset.event_bool is None:
            raise ValueError(
                "Provided dataset is not valid. "
                "Both values and mask should be not None."
            )
        return EventTensor(dataset.event_time, dataset.event_bool)

    def get_variables_specs(
        self,
        named_attach_vars: bool = True,
    ) -> Dict[VarName, VariableInterface]:
        """Automatic specifications of variables for this observation model."""
        specs = super().get_variables_specs(named_attach_vars)
        nll_attach_var = self.get_nll_attach_var_name(
            named_attach_vars
        )
        specs[f"{nll_attach_var}_ind"] = LinkedVariable(
            self.dist.get_func_nll(self.name)
        )
        specs[f"survival_{self.name}"] = LinkedVariable(
            self.dist.get_func("compute_log_survival", self.name)
        )
        return specs

    @classmethod
    def default_init(cls, **kwargs):
        return cls(
            nu=kwargs.pop("nu", "nu"),
            rho=kwargs.pop("rho", "rho"),
            xi=kwargs.pop("xi", "xi"),
            tau=kwargs.pop("tau", "tau"),
        )
