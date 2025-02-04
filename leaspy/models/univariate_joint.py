import torch
from lifelines import WeibullFitter

from leaspy.models.univariate import LogisticUnivariateModel
from leaspy.models.joint import JointModel
from leaspy.io.data.dataset import Dataset
from leaspy.utils.docs import doc_with_super  # doc_with_
from leaspy.models.base import InitializationMethod
from leaspy.variables.state import State
from leaspy.variables.specs import (
    NamedVariables,
    ModelParameter,
    PopulationLatentVariable,
    LinkedVariable,
    Hyperparameter,
    VariablesValuesRO,
)
from leaspy.variables.distributions import Normal
from leaspy.utils.functional import Exp, Sum
from leaspy.models.obs_models import observation_model_factory
import pandas as pd
from leaspy.utils.typing import DictParams, Optional
from leaspy.exceptions import LeaspyInputError

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either
# model values or model values + jacobians wrt individual parameters

# TODO refact? subclass or other proper code technique to extract model's concrete
#  formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class UnivariateJointModel(LogisticUnivariateModel, JointModel):
    """
    Manifold model for multiple variables of interest (logistic or linear formulation).

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    **kwargs
        Hyperparameters of the model (including `noise_model`)

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        * If `name` is not one of allowed sub-type: 'univariate_linear' or 'univariate_logistic'
        * If hyperparameters are inconsistent
    """
    init_tolerance = 0.3

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        obs_models_to_string = [o.to_string() for o in self.obs_models]
        if "weibull-right-censored" not in obs_models_to_string:
            self.obs_models += (
                observation_model_factory(
                    "weibull-right-censored",
                    nu='nu',
                    rho='rho',
                    xi='xi',
                    tau='tau',
                ),
            )
        variables_to_track = (
            "n_log_nu_mean",
            "log_rho_mean",
            "nll_attach_y",
            "nll_attach_event",
        )
        self.tracked_variables = self.tracked_variables.union(set(variables_to_track))


    def _validate_compatibility_of_dataset(self, dataset: Optional[Dataset] = None) -> None:
        super()._validate_compatibility_of_dataset(dataset)
        # Check that there is only one event stored
        if not (dataset.event_bool.unique() == torch.tensor([0, 1])).all():
            raise LeaspyInputError(
                "You are using a one event model, your event_bool value should only contain 0 and 1, "
                "with at least one censored event and one observed event"
            )
