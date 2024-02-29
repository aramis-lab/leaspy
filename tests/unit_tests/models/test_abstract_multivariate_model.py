import re
from typing import List
from dataclasses import dataclass

from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.exceptions import LeaspyModelInputError
from leaspy.models.obs_models import FullGaussianObservationModel

# <!> NEVER import real tests classes at top-level (otherwise their tests will be duplicated...), only MIXINS!!
from tests.unit_tests.models.test_univariate_model import ManifoldModelTestMixin


@dataclass
class MockDataset:
    headers: List[str]

    def __post_init__(self):
        self.dimension = len(self.headers)


@ManifoldModelTestMixin.allow_abstract_class_init(AbstractMultivariateModel)
class AbstractMultivariateModelTest(ManifoldModelTestMixin):

    def test_constructor_abstract_multivariate(self):
        """
        Test attribute's initialization of leaspy abstract multivariate model

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`, optional (default None)
            An instance of a subclass of leaspy AbstractModel.
        """
        model = AbstractMultivariateModel("dummy")
        self.assertEqual(type(model), AbstractMultivariateModel)
        self.assertEqual(model.name, "dummy")
        self.assertEqual(model.dimension, None)
        self.assertEqual(model.source_dimension, None)
        self.assertTrue(isinstance(model.obs_models, tuple))
        self.assertEqual(len(model.obs_models), 1)
        self.assertTrue(isinstance(model.obs_models[0], FullGaussianObservationModel))
        self.assertEqual(model.obs_models[0].name, "y")

    def test_bad_initialize_features_dimension_inconsistent(self):
        with self.assertRaisesRegex(
            LeaspyModelInputError,
            re.escape(
                "Cannot set the model's features to ['x', 'y'], because "
                "the model has been configured with a dimension of 3."
            ),
        ):
            AbstractMultivariateModel("dummy", features=["x", "y"], dimension=3)

    def test_bad_initialize_source_dim_negative(self):
        with self.assertRaisesRegex(
            LeaspyModelInputError,
            re.escape("Source dimension should be an integer in [0, dimension - 1], not -1"),
        ):
            AbstractMultivariateModel("dummy", source_dimension=-1)

    def test_bad_initialize_source_dim_float(self):
        with self.assertRaisesRegex(
            LeaspyModelInputError,
            re.escape("Source dimension should be an integer in [0, dimension - 1], not 0.5"),
        ):
            AbstractMultivariateModel("dummy", source_dimension=0.5)  # noqa

    def test_bad_initialize_source_dim_with_dataset(self):
        """source_dimension should be < dimension."""
        model = AbstractMultivariateModel("dummy", source_dimension=3)
        mock_dataset = MockDataset(["ft_1", "ft_2", "ft_3"])

        with self.assertRaisesRegex(
            LeaspyModelInputError,
            re.escape(
                "Sources dimension should be an integer in [0, dimension - 1[ but "
                "you provided `source_dimension` = 3 whereas `dimension` = 3."
            ),
        ):
            model.initialize(mock_dataset)  # noqa

    def test_initialize_source_dim_with_validate_compatibility_of_dataset(self):
        model = AbstractMultivariateModel("dummy")
        mock_dataset = MockDataset(["ft_1", "ft_2", "ft_3"])
        model._validate_compatibility_of_dataset(mock_dataset)  # noqa
        self.assertEqual(model.source_dimension, 1)  # int(sqrt(3))
