import pandas as pd

from leaspy.io.data.visit_dataframe_data_reader import VisitDataframeDataReader
from tests import LeaspyTestCase


class DataframeDataReaderTest(LeaspyTestCase):
    def setUp(self):
        # example dataframe for tests
        self.df = pd.DataFrame(
            {
                "ID": ["S1", "S1", "S3", "S3", "S2", "S2"],
                "TIME": [75, 75.001, 76, 75, 65, 87],
                "Y0": [0.5, float("nan"), 0.5, 0.5, 0.5, 0.5],
                "Y1": [0.4] * 6,
                "Y2": [0.5] * 6,
            }
        )

    def test_constructor_univariate(self):
        path = self.get_test_data_path("data_mock", "univariate_data.csv")
        df = pd.read_csv(path)

        reader = VisitDataframeDataReader()
        reader.read(df)

        iter_to_idx = {
            0: "100_S_0006",
            1: "018_S_0103",
            2: "027_S_0179",
            3: "035_S_0204",
            4: "068_S_0210",
            5: "005_S_0223",
            6: "130_S_0232",
        }

        self.assertEqual(reader.iter_to_idx, iter_to_idx)
        self.assertEqual(reader.long_outcome_names, ["MMSE"])
        self.assertEqual(reader.dimension, 1)
        self.assertEqual(reader.n_individuals, 7)
        self.assertEqual(reader.n_visits, 33)

    def test_constructor_multivariate(self):
        path = self.get_test_data_path("data_mock", "multivariate_data.csv")
        df = pd.read_csv(path)

        reader = VisitDataframeDataReader()
        reader.read(df)

        iter_to_idx = {
            0: "007_S_0041",
            1: "100_S_0069",
            2: "007_S_0101",
            3: "130_S_0102",
            4: "128_S_0138",
        }

        self.assertEqual(reader.iter_to_idx, iter_to_idx)
        self.assertEqual(reader.long_outcome_names, ["ADAS11", "ADAS13", "MMSE"])
        self.assertEqual(reader.dimension, 3)
        self.assertEqual(reader.n_individuals, 5)
        self.assertEqual(reader.n_visits, 18)

    def test_load_data_with_missing_values(self):
        # only test that it works!
        path = self.get_test_data_path("data_mock", "missing_data", "sparse_data.csv")
        df = pd.read_csv(path)
        reader = VisitDataframeDataReader()
        reader.read(df, drop_full_nan=False)

        self.assertEqual(reader.dimension, 4)
        self.assertEqual(reader.n_individuals, 2)
        self.assertEqual(reader.individuals.keys(), {"S1", "S2"})
        self.assertEqual(reader.n_visits, 14)

        nans_count_S1 = (
            pd.DataFrame(
                reader.individuals["S1"].observations, columns=reader.long_outcome_names
            )
            .isna()
            .sum(axis=0)
        )
        pd.testing.assert_series_equal(
            nans_count_S1, pd.Series({"Y0": 5, "Y1": 5, "Y2": 5, "Y3": 5})
        )

        nans_count_S2 = (
            pd.DataFrame(
                reader.individuals["S2"].observations, columns=reader.long_outcome_names
            )
            .isna()
            .sum(axis=0)
        )
        pd.testing.assert_series_equal(
            nans_count_S2, pd.Series({"Y0": 6, "Y1": 6, "Y2": 6, "Y3": 6})
        )

        # drop_full_nan=True by default
        reader = VisitDataframeDataReader()
        reader.read(df)
        self.assertEqual(reader.dimension, 4)
        self.assertEqual(reader.n_individuals, 2)
        self.assertEqual(reader.individuals.keys(), {"S1", "S2"})
        self.assertEqual(reader.n_visits, 9)

    def test_index_bad_type(self):
        # bad type (series)
        with self.assertRaises(ValueError):
            VisitDataframeDataReader().read(self.df.set_index(["ID", "TIME"])["Y0"])
        with self.assertRaises(ValueError):
            VisitDataframeDataReader().read({"ID": [], "TIME": [], "Y0": []})

    def test_missing_index_level(self):
        # missing indexes
        with self.assertRaises(ValueError):
            VisitDataframeDataReader().read(self.df.drop(columns="ID"))
        with self.assertRaises(ValueError):
            VisitDataframeDataReader().read(self.df.drop(columns="TIME"))
        with self.assertRaises(ValueError):
            VisitDataframeDataReader().read(self.df.drop(columns=["ID", "TIME"]))

    def test_bad_id(self):
        # bad ID
        with self.assertRaises(ValueError):
            # bad type
            VisitDataframeDataReader().read(self.df.assign(ID=3.14))
        with self.assertRaises(ValueError):
            # empty string for 1 individual...
            VisitDataframeDataReader().read(self.df.assign(ID=[""] + ["S1"] * 5))
        with self.assertRaises(ValueError):
            # nan string
            VisitDataframeDataReader().read(self.df.assign(ID=[pd.NA] + ["S1"] * 5))
        with self.assertRaises(ValueError):
            # < 0 index for 1 individual
            VisitDataframeDataReader().read(self.df.assign(ID=[-1] + [0] * 5))

    def test_bad_time(self):
        # bad TIME
        with self.assertRaises(ValueError):
            # bad type
            VisitDataframeDataReader().read(self.df.assign(TIME=["75.12"] * 6))
        with self.assertRaises(ValueError):
            # no nan
            VisitDataframeDataReader().read(
                self.df.assign(TIME=[float("nan")] + [75.12] * 5)
            )
        with self.assertRaises(ValueError):
            # no inf
            VisitDataframeDataReader().read(
                self.df.assign(TIME=[float("-inf")] + [75.12] * 5)
            )

    def test_bad_duplicated_index(self):
        # no duplicates on index
        with self.assertRaises(ValueError):
            # duplicates after rounding
            VisitDataframeDataReader().read(
                pd.DataFrame(
                    {
                        "ID": ["S1", "S1"],
                        "TIME": [75, 75 - 1e-10],
                    }
                )
            )

    def test_bad_dataframe_no_data(self):
        # at least one feature & one row
        with self.assertRaises(ValueError):
            VisitDataframeDataReader().read(self.df[["ID", "TIME"]])
        with self.assertRaises(ValueError):
            VisitDataframeDataReader().read(self.df.iloc[[], :])

    def test_bad_features(self):
        # bad type for a column
        with self.assertRaises(ValueError):
            VisitDataframeDataReader().read(self.df.assign(Y_bug=["0.33"] * 6))
        with self.assertRaises(ValueError):
            VisitDataframeDataReader().read(
                self.df.assign(Y_bug=[float("-inf")] + [0.4] * 5)
            )

        with self.assertWarnsRegex(UserWarning, "only contain nan"):
            VisitDataframeDataReader().read(self.df.assign(Y3=[float("nan")] * 6))

    def test_ok_when_dataframe_is_ok_even_with_a_missing_value(self):
        # check that otherwise it passes
        reader = VisitDataframeDataReader()
        reader.read(self.df, sort_index=True)
        self.assertEqual(reader.n_individuals, 3)
        self.assertEqual(reader.n_visits, 6)
        self.assertEqual(
            list(reader.individuals.keys()), ["S1", "S2", "S3"]
        )  # re-ordered
        self.assertEqual(reader.individuals["S3"].timepoints.tolist(), [75.0, 76.0])
