from bisect import bisect


class IndividualData:
    def __init__(self, idx):
        self.idx = idx
        self.timepoints = None
        self.observations = None
        self.individual_parameters = {}
        self.cofactors = {}

    def add_observation(self, timepoint, observation):
        if self.timepoints is None:
            self.timepoints = []
            self.observations = []

        if timepoint in self.timepoints:
            raise ValueError(
                'You are trying to overwrite the observation at time {} of the subject {}'.format(timepoint, self.idx))

        index = bisect(self.timepoints, timepoint)
        self.timepoints.insert(index, timepoint)
        self.observations.insert(index, observation)

    def add_observations(self, timepoints, observations):

        for i, timepoint in enumerate(timepoints):
            self.add_observation(timepoint, observations[i])

    def add_individual_parameters(self, name, value):
        self.individual_parameters[name] = value

    def add_cofactors(self, dict):
        for k, v in dict.items():
            if k in self.cofactors.keys() and v != self.cofactors[k]:
                raise ValueError("The cofactor {} is already present for patient {} ".format(k, self.idx))
            self.cofactors[k] = v

    def add_cofactor(self, name, value):
        self.cofactors[name] = value
