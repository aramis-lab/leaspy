import json
import os
import warnings

import numpy as np
import pandas as pd
import torch

from leaspy.io.outputs.individual_parameters import IndividualParameters
from tests import LeaspyTestCase


class IndividualParametersTest(LeaspyTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # for tmp handling
        super().setUpClass()

        cls.indices = ["idx1", "idx2", "idx3"]
        cls.p1 = {"xi": [0.1], "tau": [70], "sources": [0.1, -0.3]}
        cls.p2 = {"xi": [0.2], "tau": [73], "sources": [-0.4, 0.1]}
        cls.p3 = {"xi": [0.3], "tau": [58], "sources": [-0.6, 0.2]}
        cls.parameters_shape = {"xi": (1,), "tau": (1,), "sources": (2,)}
        cls.individual_parameters = {"idx1": cls.p1, "idx2": cls.p2, "idx3": cls.p3}

        ip = IndividualParameters()
        ip.add_individual_parameters("idx1", cls.p1)
        ip.add_individual_parameters("idx2", cls.p2)
        ip.add_individual_parameters("idx3", cls.p3)

        cls.ip = ip

        cls.ip_df = pd.DataFrame(
            data=[[0.1, 70, 0.1, -0.3], [0.2, 73, -0.4, 0.1], [0.3, 58, -0.6, 0.2]],
            index=["idx1", "idx2", "idx3"],
            columns=["xi", "tau", "sources_0", "sources_1"],
        )

        cls.ip_pytorch = {
            "xi": torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32),
            "tau": torch.tensor([[70], [73], [58.0]], dtype=torch.float32),
            "sources": torch.tensor(
                [[0.1, -0.3], [-0.4, 0.1], [-0.6, 0.2]], dtype=torch.float32
            ),
        }
        cls.path_json = cls.hardcoded_ip_path("ip_save.json")
        cls.path_csv = cls.hardcoded_ip_path("ip_save.csv")

    def test_constructor(self):
        ip = IndividualParameters()
        self.assertEqual(ip._indices, [])
        self.assertEqual(ip._individual_parameters, {})
        self.assertEqual(ip._parameters_shape, None)  # changed
        self.assertEqual(ip._default_saving_type, "csv")

    def test_individual_parameters(self):
        ip = IndividualParameters()
        p1 = {"xi": [0.1], "tau": [70], "sources": [0.1, -0.3]}
        p2 = {"xi": [0.2], "tau": [73], "sources": [-0.4, 0.1]}
        p3 = {"xi": [0.3], "tau": [58], "sources": [-0.6, 0.2]}

        ip.add_individual_parameters("idx1", p1)
        ip.add_individual_parameters("idx2", p2)
        ip.add_individual_parameters("idx3", p3)

        ## test fail index exist
        with self.assertRaises(ValueError):
            ip.add_individual_parameters("idx1", p3)
        ## test fail index numeric
        with self.assertRaises(ValueError):
            ip.add_individual_parameters(1, p1)
        ## test fail expect scalar
        with self.assertRaises(ValueError):
            ip.add_individual_parameters(
                "tau_not_list", {"tau": 0.2, "xi": [0.1], "sources": [0, 0]}
            )
        ## test fail missing key
        with self.assertRaises(ValueError):
            ip.add_individual_parameters("no_xi", {"tau": [0.2], "sources": [0, 0]})

        self.assertEqual(ip._indices, ["idx1", "idx2", "idx3"])
        self.assertEqual(
            ip._indices, list(ip._individual_parameters.keys())
        )  # TODO: delete indices? as they should be dict keys
        self.assertEqual(
            ip._individual_parameters, {"idx1": p1, "idx2": p2, "idx3": p3}
        )
        self.assertEqual(
            ip._parameters_shape, {"xi": (1,), "tau": (1,), "sources": (2,)}
        )
        self.assertEqual(ip._parameters_size, {"xi": 1, "tau": 1, "sources": 2})

        ### test with 1 source only (previous bug)
        ip1 = IndividualParameters()

        src1 = {"tau": [73], "xi": [0.1], "sources": [0.14]}

        ip1.add_individual_parameters("id1", src1)
        ip1.add_individual_parameters("id2", src1)

        self.assertEqual(
            ip1._parameters_shape, {"xi": (1,), "tau": (1,), "sources": (1,)}
        )
        self.assertEqual(ip1._parameters_size, {"xi": 1, "tau": 1, "sources": 1})

        ## test fail compat nb sources 1 != 2
        with self.assertRaises(ValueError):
            ip1.add_individual_parameters("id_fail", p1)

        ## test columns of dataframe
        # previously would have been "sources" for 1 source which is not generic...
        self.assertEqual({"tau", "xi", "sources_0"}, set(ip1.to_dataframe().columns))
        # previously one would get a column named "sources", with a string encoded list of 1 element at each row...
        self.assertTrue(
            all(map(pd.api.types.is_numeric_dtype, ip1.to_dataframe().dtypes))
        )

    def test_get_item(self):
        ip = IndividualParameters()

        p1 = {"xi": [0.1], "tau": [70], "sources": [0.1, -0.3]}
        p2 = {"xi": [0.2], "tau": [73], "sources": [-0.4, 0.1]}

        ip.add_individual_parameters("idx1", p1)
        ip.add_individual_parameters("idx2", p2)

        self.assertDictEqual(ip["idx1"], p1)
        self.assertDictEqual(ip["idx2"], p2)

    def test_subset(self):
        ip2 = self.ip.subset(["idx1", "idx3"])

        self.assertEqual(ip2._indices, ["idx1", "idx3"])
        self.assertEqual(ip2._individual_parameters, {"idx1": self.p1, "idx3": self.p3})
        self.assertEqual(ip2._parameters_shape, self.parameters_shape)

    def test_get_mean(self):
        ip = self.ip

        self.assertAlmostEqual(ip.get_mean("xi")[0], 0.2, delta=10e-10)
        self.assertAlmostEqual(ip.get_mean("tau")[0], 67.0, delta=10e-10)
        ss = ip.get_mean("sources")
        self.assertEqual(len(ss), 2)
        self.assertEqual(type(ss), list)
        self.assertAlmostEqual(ss[0], -0.3, delta=10e-10)
        self.assertAlmostEqual(ss[1], 0.0, delta=10e-10)

    def test_get_std(self):
        ip = self.ip

        self.assertAlmostEqual(ip.get_std("xi")[0], 0.0816496580927726, delta=10e-10)
        self.assertAlmostEqual(ip.get_std("tau")[0], 6.48074069840786, delta=10e-10)
        ss = ip.get_std("sources")
        self.assertEqual(len(ss), 2)
        self.assertEqual(type(ss), list)
        self.assertAlmostEqual(ss[0], 0.2943920288775949, delta=10e-10)
        self.assertAlmostEqual(ss[1], 0.21602468994692867, delta=10e-10)

    def test_to_dataframe(self):
        ip = self.ip
        df = ip.to_dataframe()

        self.assertTrue((df.values == self.ip_df.values).all())
        self.assertTrue((df.index == self.ip_df.index).all())
        for n1, n2 in zip(df.columns, self.ip_df.columns):
            self.assertEqual(n1, n2)

    def test_from_dataframe(self):
        ip = IndividualParameters.from_dataframe(self.ip_df)

        self.assertEqual(ip._indices, self.indices)
        print(ip._individual_parameters)
        self.assertEqual(ip._individual_parameters, self.individual_parameters)

        self.assertEqual(ip._parameters_shape, self.parameters_shape)

    def test_to_from_dataframe(self):
        ip1 = IndividualParameters.from_dataframe(self.ip_df)
        df2 = ip1.to_dataframe()
        ip2 = IndividualParameters.from_dataframe(df2)

        # Test between individual parameters
        self.assertEqual(ip1._indices, ip2._indices)
        self.assertDictEqual(ip1._individual_parameters, ip2._individual_parameters)
        self.assertDictEqual(ip1._parameters_shape, ip2._parameters_shape)

        # Test between dataframes
        self.assertTrue((self.ip_df.values == df2.values).all())
        self.assertTrue((self.ip_df.index == df2.index).all())
        for n1, n2 in zip(self.ip_df.columns, df2.columns):
            self.assertEqual(n1, n2)

    def test_to_pytorch(self):
        ip = self.ip

        indices, ip_pytorch = ip.to_pytorch()

        self.assertEqual(indices, self.indices)

        self.assertEqual(ip_pytorch.keys(), self.ip_pytorch.keys())
        for k in self.ip_pytorch.keys():
            self.assertTrue((ip_pytorch[k] == self.ip_pytorch[k]).all())

    def test_from_pytorch(self):
        ip = IndividualParameters.from_pytorch(self.indices, self.ip_pytorch)

        self.assertEqual(ip._indices, self.indices)
        self.assertEqual(
            ip._individual_parameters.keys(), self.individual_parameters.keys()
        )
        print(ip._parameters_shape)
        self.assertDictEqual(ip._parameters_shape, self.parameters_shape)
        for k, v in self.individual_parameters.items():
            for kk, vv in self.individual_parameters[k].items():
                self.assertIn(kk, ip._individual_parameters[k].keys())
                if np.ndim(vv) == 0:
                    self.assertAlmostEqual(
                        ip._individual_parameters[k][kk], vv, delta=10e-8
                    )
                else:
                    l2 = ip._individual_parameters[k][kk]
                    for s1, s2 in zip(vv, l2):
                        self.assertAlmostEqual(s1, s2, delta=10e-8)

    def test_from_to_pytorch(self):
        ip = IndividualParameters.from_pytorch(self.indices, self.ip_pytorch)
        ip_indices, ip_pytorch2 = ip.to_pytorch()
        ip2 = IndividualParameters.from_pytorch(ip_indices, ip_pytorch2)

        # Test Individual parameters
        self.assertEqual(ip._indices, ip2._indices)
        self.assertDictEqual(ip._individual_parameters, ip2._individual_parameters)
        self.assertDictEqual(ip._parameters_shape, ip2._parameters_shape)

        # Test Pytorch dictionaries
        self.assertEqual(self.ip_pytorch.keys(), ip_pytorch2.keys())
        for k in self.ip_pytorch.keys():
            for v1, v2 in zip(self.ip_pytorch[k], ip_pytorch2[k]):
                self.assertTrue((v1.numpy() - v2.numpy() == 0).all())

    def test_check_and_get_extension(self):
        tests = [
            ("file.csv", "csv"),
            ("path/to/file.csv", "csv"),
            ("file.json", "json"),
            ("path/to/file.json", "json"),
            ("nopath", None),
            ("bad_path.bad", "bad"),
        ]

        for input_path, expected_ext in tests:
            ext = IndividualParameters._check_and_get_extension(input_path)
            self.assertEqual(ext, expected_ext)

    def test_save_csv(self):
        ip = self.ip

        test_path = self.get_test_tmp_path("ip_save_csv_test.csv")
        ip._save_csv(test_path)

        with open(self.path_csv, "r") as f1, open(test_path, "r") as f2:
            file1 = f1.readlines()
            file2 = f2.readlines()

        for l1, l2 in zip(file1, file2):
            self.assertEqual(l1, l2)

        os.remove(test_path)

    def test_save_json(self):
        ip = self.ip

        test_path = self.get_test_tmp_path("ip_save_json_test.json")
        ip._save_json(test_path)

        with open(self.path_json, "r") as f1, open(test_path, "r") as f2:
            ip1 = json.load(f1)
            ip2 = json.load(f2)

        self.assertEqual(
            self.deep_sort(ip1, sort_seqs=()), self.deep_sort(ip2, sort_seqs=())
        )

        os.remove(test_path)

    def test_load_csv(self):
        ip = IndividualParameters._load_csv(self.path_csv)

        self.assertEqual(ip._indices, self.indices)
        self.assertEqual(ip._individual_parameters, self.individual_parameters)
        self.assertEqual(ip._parameters_shape, self.parameters_shape)

    def test_load_json(self):
        ip = IndividualParameters._load_json(self.path_json)

        self.assertEqual(ip._indices, self.indices)
        self.assertEqual(ip._individual_parameters, self.individual_parameters)
        self.assertEqual(ip._parameters_shape, self.parameters_shape)

    def test_load_individual_parameters(self):
        # Test json
        ip_json = IndividualParameters.load(self.path_json)

        self.assertEqual(ip_json._indices, self.indices)
        self.assertEqual(ip_json._individual_parameters, self.individual_parameters)
        self.assertEqual(ip_json._parameters_shape, self.parameters_shape)

        # Test csv
        ip_csv = IndividualParameters.load(self.path_csv)

        self.assertEqual(ip_csv._indices, self.indices)
        self.assertEqual(ip_csv._individual_parameters, self.individual_parameters)
        self.assertEqual(ip_csv._parameters_shape, self.parameters_shape)

    def test_save_individual_parameters(self):
        # Parameters
        ip = self.ip
        path_json_test = self.get_test_tmp_path("ip_save_json_test.json")
        path_csv_test = self.get_test_tmp_path("ip_save_csv_test.csv")
        path_default = self.get_test_tmp_path("ip_save_default")

        # Test json
        ip.save(path_json_test)

        with open(self.path_json, "r") as f1, open(path_json_test, "r") as f2:
            ip1 = json.load(f1)
            ip2 = json.load(f2)

        self.assertEqual(
            self.deep_sort(ip1, sort_seqs=()), self.deep_sort(ip2, sort_seqs=())
        )

        os.remove(path_json_test)

        # Test csv
        ip.save(path_csv_test)

        with open(self.path_csv, "r") as f1, open(path_csv_test, "r") as f2:
            file1 = f1.readlines()
            file2 = f2.readlines()

        for l1, l2 in zip(file1, file2):
            self.assertEqual(l1, l2)

        os.remove(path_csv_test)

        # Test default (with a warning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ip.save(path_default)

        # warning check
        self.assertEqual(len(w), 1)
        self.assertIn("You did not provide a valid extension", str(w[0].message))

        path_default_with_extension = path_default + "." + ip._default_saving_type

        with (
            open(self.path_csv, "r") as f1,
            open(path_default_with_extension, "r") as f2,
        ):
            file1 = f1.readlines()
            file2 = f2.readlines()

        for l1, l2 in zip(file1, file2):
            self.assertEqual(l1, l2)

        os.remove(path_default_with_extension)
