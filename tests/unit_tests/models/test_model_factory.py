from leaspy.models import ALL_MODELS, UnivariateModel
from leaspy.models.factory import ModelFactory
from leaspy.models.noise_models import NOISE_MODELS

from tests import LeaspyTestCase

class ModelFactoryTest_Mixin(LeaspyTestCase):

    def check_model_factory_constructor(self, model):
        """
        Test initialization of leaspy model.

        Parameters
        ----------
        model : str, optional (default None)
            Name of the model
        """
        # valid name (preconditon)
        self.assertIn(model.name, ALL_MODELS)
        self.assertEqual(type(model), ALL_MODELS[model.name])

class ModelFactoryTest(ModelFactoryTest_Mixin):

    def test_model_factory_constructor(self):
        for name in ALL_MODELS.keys():
            with self.subTest(model_name=name):
                self.check_model_factory_constructor(model=ModelFactory.model(name))

    def test_lower_case(self):
        """Test lower case"""
        name_examples = ['univariate_logistic', 'uNIVariaTE_LogIsTIc', 'UNIVARIATE_LOGISTIC']
        for name in name_examples:
            model = ModelFactory.model(name)
            # Test model type
            self.assertEqual(type(model), UnivariateModel)

    def test_wrong_arg(self):
        """Test if raise error for wrong argument"""
        # Test if raise ValueError if wrong string arg for name
        wrong_arg_examples = ['lgistic', 'blabla']
        for wrong_arg in wrong_arg_examples:
            self.assertRaises(ValueError, ModelFactory.model, wrong_arg)

        # Test if raise AttributeError if wrong object in name (not a string)
        wrong_arg_examples = [3.8, {'truc': .1}]
        for wrong_arg in wrong_arg_examples:
            self.assertRaises(ValueError, ModelFactory.model, wrong_arg)

    def test_load_hyperparameters(self):
        """Test if kwargs are ok"""
        # --- Univariate
        for name in ('univariate_linear', 'univariate_logistic'):
            with self.subTest(model_name=name):
                model = ModelFactory.model(
                    name,
                    features=['t1'],
                )
                self.assertEqual(model.features, ['t1'])
                self.assertIsInstance(model.noise_model, NOISE_MODELS['gaussian-scalar'])
                self.assertEqual(model.dimension, 1)
                self.assertEqual(model.source_dimension, 0)
                # inconsistent features for a univariate model (dimension=1)
                with self.assertRaisesRegex(ValueError, r"(?i)\bdimension\b.+\bfeatures\b"):
                    ModelFactory.model(name, features=['t1', 't2', 't3'])


        # -- Multivariate
        for name in ('linear', 'logistic', 'logistic_parallel'):
            with self.subTest(model_name=name):
                model = ModelFactory.model(
                    name,
                    features=['t1', 't2', 't3'],
                    source_dimension=2,
                    dimension=3,
                )
                self.assertEqual(model.features, ['t1','t2','t3'])
                self.assertIsInstance(model.noise_model, NOISE_MODELS['gaussian-diagonal'])
                self.assertEqual(model.dimension, 3) # TODO: automatic from length of features?
                self.assertEqual(model.source_dimension, 2)
                with self.assertRaisesRegex(ValueError, r"(?i)\bhyperparameters\b.+\bblabla\b"):
                    ModelFactory.model(name, blabla=2)

    def test_bad_noise_model_or_old_loss(self):
        # raise if invalid loss
        with self.assertRaises(ValueError):
            ModelFactory.model('logistic', noise_model='bad_noise_model')

        # NO MORE BACKWARD COMPAT -> raises about old loss kw
        with self.assertRaises(ValueError):
            ModelFactory.model('logistic', loss='MSE_diag_noise')
