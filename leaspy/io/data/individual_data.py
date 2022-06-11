from dataclasses import dataclass, field

import numpy as np

from leaspy.exceptions import LeaspyDataInputError, LeaspyTypeError
from leaspy.utils.typing import Any, Dict, FeatureType


@dataclass
class IndividualData:
    timepoints : np.ndarray
    observations : np.ndarray
    cofactors : Dict[FeatureType, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.timepoints, np.ndarray):
            raise LeaspyTypeError("Invalid `timepoints` type")

        if not len(self.timepoints.shape) == 1:
            raise LeaspyDataInputError(f"Expected 1D `timepoints`, input "
                                       f"shape was {self.timepoints.shape}")

        if not isinstance(self.observations, np.ndarray):
            raise LeaspyTypeError("Invalid `observations` type")

        if not len(self.observations.shape) == 2:
            raise LeaspyDataInputError(f"Expected 2D `observations`, input "
                                       f"shape was {self.observations.shape}")

        unique_t, occurrences = np.unique(self.timepoints, return_counts=True)
        duplicate_t = unique_t[occurrences > 1]
        if len(duplicate_t):
            raise LeaspyDataInputError(f"Duplicates found in individual "
                                       f"timepoints: {duplicate_t}")
        
        if not (
            isinstance(self.cofactors, dict)
            and all(isinstance(k, FeatureType) for k in self.cofactors.keys())
        ):
            raise LeaspyTypeError("Invalid `cofactors` type")

        # For convenience and to avoid doing so later on, timepoints and 
        # observations are jointly sorted.
        sorted_indices = np.argsort(self.timepoints)
        self.timepoints = self.timepoints[sorted_indices]
        self.observations = self.observations[sorted_indices, :]
