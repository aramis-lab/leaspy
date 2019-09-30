from leaspy.inputs.data.individual_data import IndividualData


class DataframeDataReader:
    def __init__(self, df):
        self.individuals = {}
        self.iter_to_idx = {}
        self.headers = None
        self.dimension = None
        self.n_individuals = 0
        self.n_visits = 0
        self.n_observations = 0

        self._read(df)

    @staticmethod
    def _check_headers(columns):
        columns = [_.lower() for _ in columns]
        for key in ['id', 'time']:
            if key not in columns:
                raise ValueError("Your dataframe must have a {} column".format(key))

    def _check_observation(self, observation):
        if self.dimension is None:
            self.dimension = len(observation)
        assert len(observation) == self.dimension

    def _read(self, df):
        df = df.copy(deep=True)  # No modification on the input dataframe !
        columns = df.columns.values
        self._check_headers(columns)
        df.set_index(['ID', 'TIME'], inplace=True)
        self.headers = df.columns.values.tolist()

        for k, v in df.iterrows():
            idx = k[0]
            timepoint = k[1]
            observation = v.values
            self._check_observation(observation)

            if idx not in self.individuals:
                self.individuals[idx] = IndividualData(idx)
                self.iter_to_idx[self.n_individuals] = idx
                self.n_individuals += 1

            self.individuals[idx].add_observation(timepoint, observation)
            self.n_visits += 1
            self.n_observations += len(observation)
