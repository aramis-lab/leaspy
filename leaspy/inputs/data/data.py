import numpy as np
import pandas as pd

from leaspy.inputs.data.dataframe_data_reader import DataframeDataReader
from leaspy.inputs.data.csv_data_reader import CSVDataReader
from leaspy.inputs.data.individual_data import IndividualData
from leaspy.inputs.data.dataset import Dataset


# TODO : object data as output ??? or a result object ? Because there could be ambiguetes here
# TODO or find a good way to say thet there are individual parameters here ???

class Data:
    def __init__(self):

        self.individuals = {}
        self.iter_to_idx = {}
        self.headers = None
        self.dimension = None
        self.n_individuals = 0
        self.n_visits = 0
        self.n_observations = 0
        self.iter = 0

    def get_by_idx(self, idx):
        return self.individuals[idx]

    def __getitem__(self, iter):
        return self.individuals[self.iter_to_idx[iter]]

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter >= self.n_individuals:
            self.iter = 0
            raise StopIteration
        else:
            self.iter += 1
            return self.__getitem__(self.iter - 1)

    def load_cofactors(self, df, cofactors):

        df = df.copy(deep=True)

        for iter, idx in self.iter_to_idx.items():

            # Get the cofactors and check that it is unique
            cof = df.loc[[idx]][cofactors].to_dict(orient='list')

            for c in cofactors:
                v = np.unique(cof[c])
                v = [_ for _ in v if _ == _]
                if len(v) > 1:
                    raise ValueError("Multiples values of the cofactor {} for patient {} : {}".format(c, idx, v))
                elif len(v) == 0:
                    cof[c] = None
                else:
                    cof[c] = v[0]

            # Add these cofactor to the individual
            self.individuals[idx].add_cofactors(cof)

    @staticmethod
    def from_csv_file(path):
        reader = CSVDataReader(path)
        return Data._from_reader(reader)

    def to_dataframe(self):

        indices = []
        timepoints = np.zeros((self.n_visits, 1))
        arr = np.zeros((self.n_visits, self.dimension))

        iteration = 0
        for i, indiv in enumerate(self.individuals.values()):
            ages = indiv.timepoints
            for j, age in enumerate(ages):
                indices.append(indiv.idx)
                timepoints[iteration] = age
                arr[iteration] = indiv.observations[j]

                iteration += 1

        arr = np.concatenate((timepoints, arr), axis=1)

        df = pd.DataFrame(data=arr, index=indices, columns=['TIME'] + self.headers)
        df.index.name = 'ID'
        return df.reset_index()

    @staticmethod
    def from_dataframe(df):
        reader = DataframeDataReader(df)
        return Data._from_reader(reader)

    @staticmethod
    def _from_reader(reader):
        data = Data()

        data.individuals = reader.individuals
        data.iter_to_idx = reader.iter_to_idx
        data.headers = reader.headers
        data.dimension = reader.dimension
        data.n_individuals = reader.n_individuals
        data.n_visits = reader.n_visits
        data.n_observations = reader.n_observations

        return data

    @staticmethod
    def from_individuals(indices, timepoints, values, headers):
        """
        :param indices: list of indices
        :param timepoints: list of timepoints (nd array)
        :param values: list of values (nd array
        :return:
        """

        data = Data()
        data.dimension = len(values[0][0])
        data.headers = headers

        for i, idx in enumerate(indices):
            # Create individual
            data.individuals[idx] = IndividualData(idx)
            data.iter_to_idx[data.n_individuals] = idx
            data.n_individuals += 1

            # Add observations / timepoints
            data.individuals[idx].add_observations(timepoints[i], values[i])

            # Update Data metrics
            data.n_visits += len(timepoints[i])
            data.n_observations += len(timepoints) * data.dimension

        return data
