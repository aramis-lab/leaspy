from bisect import bisect

import numpy as np

from leaspy.exceptions import LeaspyDataInputError, LeaspyTypeError
from leaspy.utils.typing import Any, Dict, FeatureType, IDType, List


class IndividualData:
    """
    Container for an individual's data

    Parameters
    ----------
    idx : IDType
        Unique ID
        
    Attributes
    ----------
    idx : IDType
        Unique ID
    timepoints : np.ndarray[float, 1D]
        Timepoints associated with the observations
    observations : np.ndarray[float, 2D]
        Observed data points.
        Shape is ``(n_timepoints, n_features)``
    cofactors : Dict[FeatureType, Any]
        Cofactors in the form {cofactor_name: cofactor_value}
    event_time: Float
        Time of an event, if the event is censored, the time correspond to the last patient observation
    event_bool: bool
        Boolean to indicate if an event is censored or not: 1 observed, 0 censored
    """

    def __init__(self, idx: IDType):
        self.idx: IDType = idx
        self.timepoints: np.ndarray = None
        self.observations: np.ndarray = None
        self.event_time: float = None
        self.event_bool: Optional[int] = None
        self.cofactors: Dict[FeatureType, Any] = {}

    def add_observations(self, timepoints: List[float], observations: List[List[float]]) -> None:
        """
        Include new observations and associated timepoints

        Parameters
        ----------
        timepoints : array-like[float, 1D]
            Timepoints associated with the observations to include
        observations : array-like[float, 2D]
            Observations to include

        Raises
        ------
        :exc:`.LeaspyDataInputError`
        """
        for t, obs in zip(timepoints, observations):
            if self.timepoints is None:
                self.timepoints = np.array([t])
                self.observations = np.array([obs])
            elif t in self.timepoints:
                raise LeaspyDataInputError(f"Trying to overwrite timepoint {t} "
                                           f"of individual {self.idx}")
            else:
                index = bisect(self.timepoints, t)
                self.timepoints = np.concatenate([
                    self.timepoints[:index],
                    [t],
                    self.timepoints[index:]
                ])
                self.observations = np.concatenate([
                    self.observations[:index],
                    [obs],
                    self.observations[index:]
                ])

    def add_event(self, event_time: float, event_bool: bool) -> None:
        """
        Include event time and associated censoring bool

        Parameters
        ----------
        event_time : float
            Time of the event
        event_bool : float
            0 if censored (not observed) and 1 if observed

        """
        self.event_time = event_time
        self.event_bool = event_bool

    def add_cofactors(self, cofactors: Dict[FeatureType, Any]) -> None:
        """
        Include new cofactors

        Parameters
        ----------
        cofactors : Dict[FeatureType, Any]
            Cofactors to include, in the form `{name: value}`

        Raises
        ------
        :exc:`.LeaspyDataInputError`
        :exc:`.LeaspyTypeError`
        """
        if not (
            isinstance(cofactors, dict)
            and all(isinstance(cofactor_name, str) for cofactor_name in cofactors.keys())
        ):
            raise LeaspyTypeError("Invalid argument type for `cofactors`")

        for cofactor_name, cofactor_value in cofactors.items():
            if cofactor_name in self.cofactors and cofactor_value != self.cofactors[cofactor_name]:
                raise LeaspyDataInputError(
                    f"Cofactor {cofactor_name} is already present for patient {self.idx} "
                    f"with a value of {self.cofactors[cofactor_name]} different from the value "
                    f"{cofactor_value} that you are trying to set."
                )
            self.cofactors[cofactor_name] = cofactor_value
