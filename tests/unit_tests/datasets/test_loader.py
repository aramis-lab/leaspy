from leaspy.datasets.loader import Loader
from leaspy.models.noise_models import GaussianScalarNoiseModel

from tests import LeaspyTestCase

# TODO: regenerate example models + individual parameters

class LoaderTest(LeaspyTestCase):

    def test_load_dataset(self):
        """
        Check ID and dtype of ID, TIME and values.
        """
        self.assertEqual(list(Loader().data_paths.keys()),
                         ['alzheimer-multivariate', 'parkinson-multivariate',
                          'parkinson-putamen', 'parkinson-putamen-train_and_test'])
        for name in Loader().data_paths.keys():
            df = Loader.load_dataset(name)
            if 'train_and_test' in name:
                self.assertEqual(df.index.names, ['ID', 'TIME', 'SPLIT'])
            else:
                self.assertEqual(df.index.names, ['ID', 'TIME'])
            self.assertTrue(all(df.dtypes.values == 'float64'))
            self.assertEqual(df.index.get_level_values('ID').unique().tolist(),
                             ['GS-' + '0'*(3 - len(str(i))) + str(i) for i in range(1, 201)])
            self.assertIn(df.index.get_level_values('TIME').dtype, ('float64', 'float32'))

    def test_load_leaspy_instance(self):
        """
        Check that all models are loadable, and check parameter values for one model.
        """
        self.assertEqual(list(Loader().model_paths.keys()), ['alzheimer-multivariate', 'parkinson-multivariate', 'parkinson-putamen-train'])

        for name in Loader().model_paths.keys():
            leaspy_instance = Loader.load_leaspy_instance(name)
            if 'multivariate' in name:
                self.assertEqual(leaspy_instance.type, 'logistic')
            else:
                self.assertEqual(leaspy_instance.type, 'univariate_logistic')

        leaspy_instance = Loader.load_leaspy_instance('parkinson-putamen-train')
        self.assertEqual(leaspy_instance.model.features, ['PUTAMEN'])
        self.assertIsInstance(leaspy_instance.model.noise_model, GaussianScalarNoiseModel)
        self.assertDictAlmostEqual(leaspy_instance.model.noise_model.parameters, {"scale": 0.02122}, atol=1e-4)

        parameters = {
            "g": [-1.1861],
            "v0": [-4.0517],
            "tau_mean": 68.7493,
            "tau_std": 10.0294,
            "xi_mean": 0.0,
            "xi_std": 0.5542,
            "sources_mean": 0.0,
            "sources_std": 1.0,
            "betas": [],
        }
        self.assertDictAlmostEqual(leaspy_instance.model.parameters, parameters, atol=1e-4)

    def test_load_individual_parameters(self):
        """
        Check that all ips are loadable, and check values for one individual_parameters
        instance.
        """
        self.assertEqual(list(Loader().ip_paths.keys()), ['alzheimer-multivariate', 'parkinson-multivariate', 'parkinson-putamen-train'])

        for name in Loader().ip_paths.keys():
            ip = Loader.load_individual_parameters(name)

        ip = Loader.load_individual_parameters('alzheimer-multivariate')

        self.assertAlmostEqual(ip.get_mean('tau'), 76.9612, delta=1e-4)
        self.assertAlmostEqual(ip.get_mean('xi'), 0.0629, delta=1e-4)
        self.assertAllClose(ip.get_mean('sources'), [0.00315, -0.02109], atol=1e-5, rtol=1e-4, what='sources.mean')
