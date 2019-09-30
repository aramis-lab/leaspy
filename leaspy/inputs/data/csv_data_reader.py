import csv
from leaspy.inputs.data.individual_data import IndividualData


class CSVDataReader:
    def __init__(self, path):
        self.individuals = {}
        self.iter_to_idx = {}
        self.headers = None
        self.dimension = None
        self.n_individuals = 0
        self.n_visits = 0
        self.n_observations = 0

        self._read(path)

    def _check_headers(self, csv_headers):
        if len(csv_headers) < 3:
            raise ValueError("There must be at least three columns in the input dataset")
        if csv_headers[0].lower() != 'id':
            raise ValueError("The first column of the input csv must be \'ID\'")
        if csv_headers[1].lower() != 'time':
            raise ValueError("The second column of the input csv must be \'Time\'")

        self.headers = csv_headers[2:]

    @staticmethod
    def _get_timepoint(idx, timepoint):
        try:
            return float(timepoint)
        except ValueError:
            print('The timepoint {} of individual {} cannot be converted to float'.format(timepoint, idx))

    @staticmethod
    def _get_observation(idx, timepoint, observation):
        try:
            return [float(_) for _ in observation]
        except ValueError:
            print('The observations of individual ' + str(idx) + ' at time ' + str(
                timepoint) + ' cannot be converted to float')

    def _check_observation(self, observation):
        if self.dimension is None:
            self.dimension = len(observation)
        assert len(observation) == self.dimension

    def _read(self, path):
        # Read csv
        with open(path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_headers = next(csv_reader)
            self._check_headers(csv_headers)

            # Add new individuals
            for row in csv_reader:
                idx = row[0]
                timepoint = self._get_timepoint(idx, row[1])
                observation = self._get_observation(idx, timepoint, row[2:])
                if observation is not None:
                    self._check_observation(observation)

                    if idx not in self.individuals:
                        self.individuals[idx] = IndividualData(idx)
                        self.iter_to_idx[self.n_individuals] = idx
                        self.n_individuals += 1

                    self.individuals[idx].add_observation(timepoint, observation)
                    self.n_visits += 1
                    self.n_observations += len(observation)
