import pytest

import pandas as pd

from leaspy.io.data.data import Data
from leaspy.exceptions import LeaspyDataInputError, LeaspyTypeError
from leaspy.utils.typing import Tuple

from tests import LeaspyTestCase


class DataTest(LeaspyTestCase):

    def load_multivariate_data(self):
        path_to_data = self.get_test_data_path('data_mock', 'multivariate_data.csv')
        return Data.from_csv_file(path_to_data)

    def test_constructor_univariate(self):
        path_to_data = self.get_test_data_path('data_mock', 'univariate_data.csv')
        data = Data.from_csv_file(path_to_data)
        individual = data[2]

        self.assertEqual(data.iter_to_idx[0], '100_S_0006')
        self.assertEqual(data.iter_to_idx[len(data.iter_to_idx)-1], '130_S_0232')
        self.assertEqual(data.headers, ['MMSE'])
        self.assertEqual(data.dimension, 1)
        self.assertEqual(data.n_individuals, 7)
        self.assertEqual(data.n_visits, 33)
        self.assertEqual(data.cofactors, [])

        self.assertEqual(individual.idx, '027_S_0179')
        self.assertEqual(individual.timepoints.tolist(), [80.9, 81.9, 82.4, 82.8])
        self.assertEqual(individual.observations.tolist(), [[0.2], [0.2], [0.3], [0.5]])

    def test_constructor_multivariate(self):
        data = self.load_multivariate_data()
        individual = data[3]

        self.assertEqual(data.iter_to_idx[0], '007_S_0041')
        self.assertEqual(data.iter_to_idx[len(data.iter_to_idx)-1], '128_S_0138')
        self.assertEqual(data.headers, ['ADAS11', 'ADAS13', 'MMSE'])
        self.assertEqual(data.dimension, 3)
        self.assertEqual(data.n_individuals, 5)
        self.assertEqual(data.n_visits, 18)
        self.assertEqual(data.cofactors, [])

        self.assertEqual(individual.idx, '130_S_0102')
        self.assertEqual(individual.timepoints.tolist(), [71.3, 71.8])
        self.assertEqual(individual.observations.tolist(), [[0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])

    def check_sub_data(self, data, sub_data, individual, sub_individual):
        """Helper to check the compliant behaviour of sliced data

        Parameters
        ----------
        data : Data
            The source container to compare against
        sub_data : Data
            The sliced data container to compare against the source
        individual : IndividualData
            An individual from the source data container to compare
            against
        sub_individual : IndividualData
            An individual from the sliced data container to compare
            against the source individual
        """
        self.assertEqual(data.headers, sub_data.headers)
        self.assertEqual(data.dimension, sub_data.dimension)
        self.assertEqual(data.cofactors, sub_data.cofactors)

        self.assertEqual(individual.idx, sub_individual.idx)
        self.assertEqual(individual.timepoints.tolist(), sub_individual.timepoints.tolist())
        self.assertEqual(individual.observations.tolist(), sub_individual.observations.tolist())
        self.assertEqual(individual.cofactors, sub_individual.cofactors)

    def test_data_slicing(self):
        data = self.load_multivariate_data()
        individual_key = 3
        individual = data[individual_key]

        # Slice slicing
        start, stop = 1, 5
        sub_data_slice = data[start:stop]
        sub_individual_slice = sub_data_slice[individual_key - start]
        self.check_sub_data(data, sub_data_slice, individual, sub_individual_slice)

        # list[int] slicing
        l_int = [0, individual_key]
        sub_data_int = data[l_int]
        sub_individual_int = sub_data_int[l_int.index(individual_key)]
        self.check_sub_data(data, sub_data_int, individual, sub_individual_int)

        # list[IDType] slicing
        l_id = [data.iter_to_idx[i] for i in l_int]
        sub_data_id = data[l_id]
        sub_individual_id = sub_data_id[data.iter_to_idx[individual_key]]
        self.check_sub_data(data, sub_data_id, individual, sub_individual_id)

        # Unsupported slicing
        with pytest.raises(LeaspyTypeError):
            _ = data[{}]
        
        # Membership
        assert individual.idx in data
        assert data[0].idx not in sub_data_slice

        # Unsupported membership
        with pytest.raises(LeaspyTypeError):
            _ = (0 in data)

    def test_data_iteration(self):
        data = self.load_multivariate_data()
        for iter, individual in enumerate(data):
            expected_individual = data[iter]
            self.assertEqual(individual.idx, expected_individual.idx)
            self.assertEqual(individual.timepoints.tolist(), expected_individual.timepoints.tolist())
            self.assertEqual(individual.observations.tolist(), expected_individual.observations.tolist())
            if iter > 4:
                break

    def get_data_and_cofactors_df(self) -> Tuple[Data, pd.DataFrame]:
        data = self.load_multivariate_data()
        idx_list = data.individuals.keys()
        cofactors_list = ["Cofactor_1", "Cofactor_2"]
        cofactors_df = pd.DataFrame(
            index=idx_list,
            data=[(idx[0], idx[-1]) for idx in idx_list],
            columns=cofactors_list
        )
        cofactors_df.index.name = "ID"
        data.load_cofactors(cofactors_df, cofactors=None)
        return data, cofactors_df

    def test_data_load_cofactors(self):
        data, _ = self.get_data_and_cofactors_df()
        self.assertEqual(data.cofactors, ["Cofactor_1", "Cofactor_2"])
        self.assertEqual(data[3].cofactors["Cofactor_2"], data[3].idx[-1])

    def test_data_load_cofactors_index_error(self):
        data, cofactors_df = self.get_data_and_cofactors_df()
        cofactors_df.index.name = "Wrong_index_name"
        with pytest.raises(LeaspyDataInputError):
            data.load_cofactors(cofactors_df, cofactors=None)

    def test_data_load_cofactors_value_error(self):
        data, cofactors_df = self.get_data_and_cofactors_df()
        cofactors_df.loc[4] = [0, 0]
        with pytest.raises(LeaspyDataInputError):
            data.load_cofactors(cofactors_df, cofactors=None)

    def test_data_load_cofactors_missing_entry_error(self):
        data, cofactors_df = self.get_data_and_cofactors_df()
        cofactors_df.drop(data[3].idx, inplace=True)
        with pytest.raises(LeaspyDataInputError):
            data.load_cofactors(cofactors_df, cofactors=None)

    def test_data_cofactors_to_dataframe(self):
        data, cofactors_df = self.get_data_and_cofactors_df()
        df = data.to_dataframe(cofactors="all")
        self.assertEqual(
            df.shape,
            (data.n_visits, len(data.headers + data.cofactors) + 2)
        )
        self.assertEqual(
            df.loc[df["ID"] == data[3].idx, "Cofactor_1"].to_list(),
            [data[3].idx[0] for _ in range(len(data[3].timepoints))]
        )

    def test_data_cofactors_to_dataframe_error(self):
        data, _ = self.get_data_and_cofactors_df()
        with pytest.raises(LeaspyDataInputError):
            data.to_dataframe(cofactors="Cofactor_1")

    def test_data_cofactors_to_dataframe_empty_error(self):
        data, _ = self.get_data_and_cofactors_df()
        with pytest.raises(LeaspyTypeError):
            data.to_dataframe(cofactors={})

    def test_data_cofactors_to_dataframe_wrong_cofactor_error(self):
        data, _ = self.get_data_and_cofactors_df()
        with pytest.raises(LeaspyDataInputError):
            data.to_dataframe(cofactors=["Wrong_cofactor"])
