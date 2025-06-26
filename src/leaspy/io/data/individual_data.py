from bisect import bisect
from typing import Any, Optional

import numpy as np
import pandas as pd

from leaspy.exceptions import LeaspyDataInputError, LeaspyInputError, LeaspyTypeError
from leaspy.utils.typing import FeatureType, IDType

__all__ = ["IndividualData"]


class IndividualData:
    """
    Container for an individual's data

    Parameters
    ----------
    idx : IDType
        Unique ID

    Attributes
    ----------
    idx : :class:`~leaspy.utils.typing.IDType`
        Unique ID
    timepoints : :obj:`np.ndarray` [:obj:`float`]
        Timepoints associated with the observations 1D array
    observations : :obj:`np.ndarray` [:obj:`float`]
        Observed data points,
        Shape is ``(n_timepoints, n_features)``
    cofactors : :obj:`dict` [:class: `~leaspy.utils.typing.FeatureType`,:class: `~leaspy.utils.typing.Any`]
        Cofactors in the form {cofactor_name: cofactor_value}
    event_time : :obj:`float`
        Time of an event, if the event is censored, the time correspond to the last patient observation
    event_bool : :obj:`bool`
        Boolean to indicate if an event is censored or not: 1 observed, 0 censored
    """

    def __init__(self, idx: IDType):
        self.idx: IDType = idx
        self.timepoints: np.ndarray = None
        self.observations: np.ndarray = None
        self.event_time: Optional[np.ndarray] = None
        self.event_bool: Optional[np.ndarray] = None
        self.cofactors: dict[FeatureType, Any] = {}
        self.covariates: Optional[np.ndarray] = None

    def add_observations(
        self, timepoints: list[float], observations: list[list[float]]
    ) -> None:
        """
        Include new observations and associated timepoints

        Parameters
        ----------
        timepoints : :obj:`array-like` [:obj:`float`]
            Timepoints associated with the observations to include, 1D array
        observations : :obj:`array-like` [:obj:`float`]
            Observations to include, 2D array

        Raises
        ------
        :exc:`.LeaspyDataInputError`
        """
        for t, obs in zip(timepoints, observations):
            if self.timepoints is None:
                self.timepoints = np.array([t])
                self.observations = np.array([obs])
            elif t in self.timepoints:
                raise LeaspyDataInputError(
                    f"Trying to overwrite timepoint {t} " f"of individual {self.idx}"
                )
            else:
                index = bisect(self.timepoints, t)
                self.timepoints = np.concatenate(
                    [self.timepoints[:index], [t], self.timepoints[index:]]
                )
                self.observations = np.concatenate(
                    [self.observations[:index], [obs], self.observations[index:]]
                )

    def add_event(self, event_time: list[float], event_bool: list[bool]) -> None:
        """
        Include event time and associated censoring bool

        Parameters
        ----------
        event_time : :obj:`float`
            Time of the event
        event_bool : :obj:`float`
            0 if censored (not observed) and 1 if observed

        """
        self.event_time = np.array(event_time)
        self.event_bool = np.array(event_bool)

    def add_covariates(self, covariates: list[list[int]]) -> None:
        """
        Include covariates

        Parameters
        ----------
        covariates : :obj:`array-like` [:obj:`float`]
            Covariates to include, 2D array
        """
        self.covariates = np.array(covariates)

    def add_cofactors(self, cofactors: dict[FeatureType, Any]) -> None:
        """
        Include new cofactors

        Parameters
        ----------
        cofactors : :obj:`dict` [:class:`~leaspy.utils.typing.FeatureType`, :class:`~leaspy.utils.typing.Any`]
            Cofactors to include, in the form `{name: value}`

        Raises
        ------
        :exc:`.LeaspyDataInputError`
        :exc:`.LeaspyTypeError`
        """
        if not (
            isinstance(cofactors, dict)
            and all(
                isinstance(cofactor_name, str) for cofactor_name in cofactors.keys()
            )
        ):
            raise LeaspyTypeError("Invalid argument type for `cofactors`")

        for cofactor_name, cofactor_value in cofactors.items():
            if (
                cofactor_name in self.cofactors
                and cofactor_value != self.cofactors[cofactor_name]
            ):
                raise LeaspyDataInputError(
                    f"Cofactor {cofactor_name} is already present for patient {self.idx} "
                    f"with a value of {self.cofactors[cofactor_name]} different from the value "
                    f"{cofactor_value} that you are trying to set."
                )
            self.cofactors[cofactor_name] = cofactor_value

    def to_frame(
        self,
        headers: list,
        event_time_name: str,
        event_bool_name: str,
        covariate_names: list[str],
    ) -> pd.DataFrame:
        """
        Convert the individual data to a pandas DataFrame

        Parameters
        ----------
        headers : :obj:`list` [:obj:`str`]
            List of feature names for the observations
        event_time_name : :obj:`str`
            Name of the column for the event time
        event_bool_name : :obj:`str`
            Name of the column for the event boolean (0 or 1)
        covariate_names : :obj:`list` [:obj:`str`]
            List of covariate names

        Returns
        -------
        :obj:`pd.DataFrame`
            DataFrame containing the individual's data with the following columns:
            - ID: Unique identifier for the individual
            - TIME: Timepoints associated with the observations
            - Observations: Observed data points for each feature
            - Event Time: Time of the event (if any)
            - Event Boolean: Boolean indicating if the event was observed (1) or censored (0)
            - Covariates: Values of the covariates for the individual

        """
        type_to_concat = []
        if self.observations is not None:
            ix_tpts = pd.MultiIndex.from_product(
                [[self.idx], self.timepoints], names=["ID", "TIME"]
            )
            type_to_concat.append(
                pd.DataFrame(self.observations, columns=headers, index=ix_tpts)
            )
        if self.event_time is not None:
            df_event = self._event_to_frame(event_time_name, event_bool_name)
            type_to_concat.append(df_event)

        if self.covariates is not None:
            df_covariate = self._covariate_to_frame(covariate_names)
            type_to_concat.append(df_covariate)

        if len(type_to_concat) == 1:
            return type_to_concat[0]
        else:
            return type_to_concat[1].join(type_to_concat[0])

    def _event_to_frame(
        self, event_time_name: str, event_bool_name: str
    ) -> pd.DataFrame:
        """
        Convert the event data to a pandas DataFrame

        Parameters
        ----------
        event_time_name : :obj:`str`
            Name of the column for the event time
        event_bool_name : :obj:`str`
            Name of the column for the event boolean (0 or 1)

        Returns
        -------
        :obj:`pd.DataFrame`
            DataFrame containing the event data with the following columns:
            - ID: Unique identifier for the individual
            - Event Time: Time of the event (if any)
            - Event Boolean: Boolean indicating if the event was observed (1) or censored (0)
        """
        ix_tpts = pd.Index([self.idx], name="ID")
        if len(np.unique(self.event_time)) != 1:
            raise LeaspyInputError(
                f"Individual {self.idx} has multiple time at event only one is accepted"
            )

        if self.event_bool.sum() == 1:
            event_coded = np.where(self.event_bool)[0][0]
            event_bool = event_coded + 1
        elif self.event_bool.sum() == 0:
            event_bool = 0
        else:
            raise LeaspyInputError(
                f"Individual {self.idx} should contain maximum one observed event"
            )

        df_event = pd.DataFrame(
            data=[[self.event_time[0], event_bool]],
            index=ix_tpts,
            columns=[event_time_name, event_bool_name],
        )
        df_event[event_time_name] = df_event[event_time_name].astype(float)
        df_event[event_bool_name] = df_event[event_bool_name].astype(int)
        return df_event

    def _covariate_to_frame(self, covariate_names: list[str]) -> pd.DataFrame:
        """
        Convert the covariates to a pandas DataFrame

        Parameters
        ----------
        covariate_names : :obj:`list`[:obj:`str`]
            List of covariate names

        Returns
        -------
        :obj:`pd.DataFrame`
            DataFrame containing the covariates with the following columns:
            - ID: Unique identifier for the individual
            - Covariates: Values of the covariates for the individual
        """
        ix_tpts = pd.Index([self.idx], name="ID")
        df_covariates = pd.DataFrame(
            data=[self.covariates],
            index=ix_tpts,
            columns=[covariate_names],
        )
        return df_covariates
