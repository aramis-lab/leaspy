from typing import Optional

import torch

from leaspy.exceptions import LeaspyInputError
from leaspy.io.data.dataset import Dataset
from leaspy.utils.docs import doc_with_super  # doc_with_

from .joint import JointModel
from .multivariate import LogisticModel
from .obs_models import observation_model_factory

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either
# model values or model values + jacobians wrt individual parameters

# TODO refact? subclass or other proper code technique to extract model's concrete
#  formulation depending on if linear, logistic, mixed log-lin, ...


__all__ = ["UnivariateJointModel"]


@doc_with_super()
class UnivariateJointModel(JointModel):
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
        if "dimension" not in kwargs:
            super().__init__(name, dimension=1, **kwargs)
        else:
            super().__init__(name, **kwargs)
        obs_models_to_string = [o.to_string() for o in self.obs_models]
        if "gaussian-scalar" not in obs_models_to_string:
            self.obs_models += (
                observation_model_factory("gaussian-scalar", dimension=1),
            )
        if "weibull-right-censored" not in obs_models_to_string:
            self.obs_models += (
                observation_model_factory(
                    "weibull-right-censored",
                    nu="nu",
                    rho="rho",
                    xi="xi",
                    tau="tau",
                ),
            )
        variables_to_track = (
            "n_log_nu_mean",
            "log_rho_mean",
            "nll_attach_y",
            "nll_attach_event",
        )
        self.tracked_variables = self.tracked_variables.union(set(variables_to_track))

    def _validate_compatibility_of_dataset(
        self, dataset: Optional[Dataset] = None
    ) -> None:
        """
        Raise if the given :class:`.Dataset` is not compatible with the current model.

        Parameters
        ----------
        dataset : :class:`.Dataset`, optional

        Raises
        ------
        :exc:`.LeaspyInputError` :
            - If the :class:`.Dataset` has a number of dimensions smaller than 2.
            - If the :class:`.Dataset` does not have the same dimensionality as the model.
            - If the :class:`.Dataset`'s headers do not match the model's.
        """
        super()._validate_compatibility_of_dataset(dataset)
        # Check that there is only one event stored
        if not (dataset.event_bool.unique() == torch.tensor([0, 1])).all():
            raise LeaspyInputError(
                "You are using a one event model, your event_bool value should only contain 0 and 1, "
                "with at least one censored event and one observed event"
            )
