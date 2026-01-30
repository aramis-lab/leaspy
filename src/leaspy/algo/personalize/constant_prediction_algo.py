from enum import Enum

import numpy as np

from leaspy.io.data import Dataset
from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.models import ConstantModel
from leaspy.utils.typing import FeatureType

from ..base import AlgorithmName
from ..settings import AlgorithmSettings
from .base import PersonalizeAlgorithm

__all__ = ["ConstantPredictionAlgorithm"]


class PredictionType(str, Enum):
    LAST = "last"
    LAST_KNOWN = "last-known"
    MAX = "max"
    MEAN = "mean"


class ConstantPredictionAlgorithm(
    PersonalizeAlgorithm[ConstantModel, IndividualParameters]
):
    r"""ConstantPredictionAlgorithm is an algorithm that provides constant predictions.

    It is used with the :class:`~leaspy.models.ConstantModel`.

    Parameters
    ----------
    settings : :class:`~leaspy.algo.settings.AlgorithmSettings`
        The settings of constant prediction algorithm. It supports the following  `prediction_type` values (str)::
            * ``'last'``: last value even if NaN,
            * ``'last_known'``: last non NaN value,
            * ``'max'``: maximum (=worst) value ,
            * ``'mean'``: average of values

        depending on features, the `last_known` / `max` value may correspond to different visits.
        For a given feature, value will be NaN if and only if all values for this feature are NaN.

    Raises
    ------
    :exc:`.LeaspyAlgoInputError`
        If any invalid setting for the algorithm
    """

    name: AlgorithmName = AlgorithmName.PERSONALIZE_CONSTANT
    deterministic: bool = True

    def __init__(self, settings: AlgorithmSettings):
        super().__init__(settings)
        self.prediction_type: PredictionType = PredictionType(
            settings.parameters["prediction_type"]
        )

    def _compute_individual_parameters(
        self, model: ConstantModel, dataset: Dataset, **kwargs
    ) -> IndividualParameters:
        # always overwrite model features (no fit process)
        # TODO? we could fit the model before, only to recover model features,
        #  and then check at personalize that is the same (as in others personalize algos...)
        # Always overwrite model features (no fit for constant model...)
        model.initialize(dataset)
        individual_parameters = IndividualParameters()
        for individual in range(dataset.n_individuals):
            idx = dataset.indices[individual]
            times = dataset.get_times_patient(individual)
            values = dataset.get_values_patient(individual).numpy()
            ind_ip = self._get_individual_last_values(
                times, values, features=model.features
            )
            individual_parameters.add_individual_parameters(str(idx), ind_ip)
        return individual_parameters

    def _get_individual_last_values(
        self, times: np.ndarray, values: np.ndarray, *, features: list[FeatureType]
    ):
        """Get individual last values.

        Parameters
        ----------
        times : :class:`numpy.ndarray` [float]
            shape (n_visits,)

        values : :class:`numpy.ndarray` [float]
            shape (n_visits, n_features)

        features : list[FeatureType]
            Feature names

        Returns
        -------
        dict[ft_name: str, constant_value_to_be_padded]
        """
        # return a dict with parameters names being features names
        return dict(zip(features, self._get_feature_values(times, values)))

    def _get_feature_values(self, times: np.ndarray, values: np.ndarray):
        if self.prediction_type == PredictionType.MAX:
            return np.nanmax(values, axis=0)
        if self.prediction_type == PredictionType.MEAN:
            return np.nanmean(values, axis=0)
        sorted_indices = sorted(range(len(times)), key=times.__getitem__, reverse=True)
        # Sometimes, last value can be a NaN.
        # If this behavior is intended, then return it anyway
        if self.prediction_type == PredictionType.LAST:
            return values[sorted_indices[0]]
        values_sorted_desc = values[sorted_indices]
        # get first index of values being non nan, with visits ordered by more recent
        last_non_nan_ix_per_ft = (~np.isnan(values_sorted_desc)).argmax(axis=0)
        # 1 feature value will be nan iff feature was nan at all visits
        return values_sorted_desc[last_non_nan_ix_per_ft, range(values.shape[1])]
