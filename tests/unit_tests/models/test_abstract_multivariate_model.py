from unittest.mock import patch
from dataclasses import dataclass
from typing import List

from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.models.noise_models import NOISE_MODELS

# <!> NEVER import real tests classes at top-level (otherwise their tests will be duplicated...), only MIXINS!!
from tests.unit_tests.models.test_univariate_model import ManifoldModelTest_Mixin

@dataclass
class MockDataset:
    headers: List[str]

    def __post_init__(self):
        self.dimension = len(self.headers)

@ManifoldModelTest_Mixin.allow_abstract_class_init(AbstractMultivariateModel)
class AbstractMultivariateModelTest(ManifoldModelTest_Mixin):

    def test_constructor_abstract_multivariate(self):
        """
        Test attribute's initialization of leaspy abstract multivariate model

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`, optional (default None)
            An instance of a subclass of leaspy AbstractModel.
        """

        # Abstract Multivariate Model
        model = AbstractMultivariateModel('dummy')
        self.assertEqual(type(model), AbstractMultivariateModel)
        self.assertEqual(model.name, 'dummy')

        # Test common initialization with univariate / manifold model
        self.check_common_attrs(model)

        # Test specific multivariate initialization
        self.assertEqual(model.dimension, None)
        self.assertEqual(model.source_dimension, None)
        self.assertIsInstance(model.noise_model, NOISE_MODELS['gaussian-diagonal'])

        self.assertEqual(model.parameters['betas'], None)
        self.assertEqual(model.parameters['sources_mean'], None)
        self.assertEqual(model.parameters['sources_std'], None)
        self.assertEqual(model.MCMC_toolbox['priors']['betas_std'], None)

    def test_bad_initialize_features_dimension_inconsistent(self):

        with self.assertRaisesRegex(ValueError, 'does not match'):
            AbstractMultivariateModel('dummy', features=['x', 'y'], dimension=3)

    def test_bad_initialize_source_dim(self):

        with self.assertRaises(ValueError):
            m = AbstractMultivariateModel('dummy', source_dimension=-1)

        with self.assertRaises(ValueError):
            m = AbstractMultivariateModel('dummy', source_dimension=0.5)

        m = AbstractMultivariateModel('dummy', source_dimension=3)

        mock_dataset = MockDataset(['ft_1', 'ft_2', 'ft_3'])

        with self.assertRaisesRegex(ValueError, 'source_dimension'):
            # source_dimension should be < dimension
            m.initialize(mock_dataset)

        m = AbstractMultivariateModel('logistic')  # should be a valid name for Attributes
        with patch('leaspy.models.abstract_multivariate_model.initialize_parameters') as MockInitParams:
            MockInitParams.return_value = {}, {}
            with self.assertWarnsRegex(UserWarning, 'source_dimension'):
                m.initialize(mock_dataset)
        self.assertEqual(m.source_dimension, 1)  # int(sqrt(3))

    def test_get_attributes(self):

        m = AbstractMultivariateModel('d')

        # not supported attributes (only None & 'MCMC' are)
        with self.assertRaises(ValueError):
            m._get_attributes(False)
        with self.assertRaises(ValueError):
            m._get_attributes(True)
        with self.assertRaises(ValueError):
            m._get_attributes('toolbox')
