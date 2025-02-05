import warnings

import numpy as np

from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.io.data import Dataset
from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.models import ConstantModel
from leaspy.utils.typing import FeatureType

from ..base import AbstractAlgo, AlgorithmName, AlgorithmType
from ..settings import AlgorithmSettings

__all__ = ["ConstantPredictionAlgorithm"]


class ConstantPredictionAlgorithm(AbstractAlgo):
    r"""
    ConstantPredictionAlgorithm is the algorithm that outputs a constant prediction

    It is associated to :class:`~.models.constant_model.ConstantModel`

    TODO: it should be a child of `AbstractPersonalizeAlgorithm` (refactoring needed)

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        The settings of constant prediction algorithm.
        It may define `prediction_type` (str):
            * ``'last'``: last value seen during calibration (even if NaN) [default],
            * ``'last_known'``: last non NaN value seen during calibration*§,
            * ``'max'``: maximum (=worst) value seen during calibration*§,
            * ``'mean'``: average of values seen during calibration§.

        | \\* <!> depending on features, the `last_known` / `max` value may correspond to different visits.
        | § <!> for a given feature, value will be NaN if and only if all values for this feature were NaN.

    Raises
    ------
    :exc:`.LeaspyAlgoInputError`
        If any invalid setting for the algorithm
    """

    name: AlgorithmName = AlgorithmName.PERSONALIZE_CONSTANT
    family: AlgorithmType = AlgorithmType.PERSONALIZE
    deterministic: bool = True
    _prediction_types = {"last", "last_known", "max", "mean"}

    def __init__(self, settings: AlgorithmSettings):
        super().__init__(settings)
        if settings.parameters["prediction_type"] not in self._prediction_types:
            raise LeaspyAlgoInputError(
                f"The `prediction_type` of the constant prediction should be in {self._prediction_types}"
            )
        self.prediction_type = settings.parameters["prediction_type"]

    def run_impl(self, model: ConstantModel, dataset: Dataset):
        """
        Main method, refer to abstract definition in :meth:`~.algo.personalize.abstract_personalize_algo.AbstractPersonalizeAlgo.run`.

        Parameters
        ----------
        model : :class:`~.models.constant_model.ConstantModel`
            A subclass object of leaspy `ConstantModel`.
        dataset : :class:`.Dataset`
            Dataset object build with leaspy class objects Data, algo & model

        Returns
        -------
        individual_parameters : :class:`.IndividualParameters`
            Contains individual parameters.
        noise_std : float
            TODO: always 0 for now
        """

        # always overwrite model features (no fit process)
        # TODO? we could fit the model before, only to recover model features, and then check at personalize that is the same (as in others personalize algos...)

        # Always overwrite model features (no fit for constant model...)
        model.initialize(dataset)

        ip = IndividualParameters()
        for it in range(dataset.n_individuals):
            idx = dataset.indices[it]
            times = dataset.get_times_patient(it)
            values = dataset.get_values_patient(it).numpy()
            ind_ip = self._get_individual_last_values(times, values, fts=model.features)

            ip.add_individual_parameters(str(idx), ind_ip)

        return ip

    def _get_individual_last_values(
        self, times: np.ndarray, values: np.ndarray, *, fts: list[FeatureType]
    ):
        """
        Parameters
        ----------
        times : :class:`numpy.ndarray` [float]
            shape (n_visits,)

        values : :class:`numpy.ndarray` [float]
            shape (n_visits, n_features)

        fts : list[str]
            Feature names

        Returns
        -------
        dict[ft_name: str, constant_value_to_be_padded]
        """

        # Return the maximum value
        if self.prediction_type == "max":
            fts_values = np.nanmax(values, axis=0)

        # Return the mean value
        elif self.prediction_type == "mean":
            fts_values = np.nanmean(values, axis=0)

        # Return the last or last-known
        else:
            # sort by more recent visits first (= patient is older)
            sorted_indices = sorted(
                range(len(times)), key=times.__getitem__, reverse=True
            )

            # Sometimes, last value can be a NaN. If this behavior is intended, then return it anyway
            if self.prediction_type == "last":
                fts_values = values[sorted_indices[0]]

            else:  # == 'last_known'
                values_sorted_desc = values[sorted_indices]

                # get first index of values being non nan, with visits ordered by more recent
                last_non_nan_ix_per_ft = (~np.isnan(values_sorted_desc)).argmax(axis=0)
                # 1 feature value will be nan iff feature was nan at all visits
                fts_values = values_sorted_desc[
                    last_non_nan_ix_per_ft, range(values.shape[1])
                ]

        # return a dict with parameters names being features names
        return dict(zip(fts, fts_values))

    def set_output_manager(self, output_settings):
        """
        Not implemented.
        """
        if output_settings is not None:
            warnings.warn(
                "Settings logs in constant prediction algorithm is not supported."
            )
        return
