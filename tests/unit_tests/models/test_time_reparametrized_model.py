from dataclasses import dataclass

from leaspy.exceptions import LeaspyModelInputError
from leaspy.models import TimeReparametrizedModel

# <!> NEVER import real tests classes at top-level (otherwise their tests will be duplicated...), only MIXINS!!
from tests.unit_tests.models.test_univariate_model import ManifoldModelTestMixin


@dataclass
class MockDataset:
    headers: list[str]

    def __post_init__(self):
        self.dimension = len(self.headers)


@ManifoldModelTestMixin.allow_abstract_class_init(TimeReparametrizedModel)
class TimeReparametrizedModelTest(ManifoldModelTestMixin):
    def test_constructor_abstract_multivariate(self):
        """
        Test attribute's initialization of leaspy abstract multivariate model

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`, optional (default None)
            An instance of a subclass of leaspy AbstractModel.
        """

        model = TimeReparametrizedModel("dummy")
        self.assertEqual(type(model), TimeReparametrizedModel)
        self.assertEqual(model.name, "dummy")

        # Test specific multivariate initialization
        self.assertEqual(model.dimension, None)
        self.assertEqual(model.source_dimension, None)

    def test_bad_initialize_features_dimension_inconsistent(self):
        with self.assertRaises(LeaspyModelInputError):
            TimeReparametrizedModel("dummy", features=["x", "y"], dimension=3)

    def test_bad_initialize_source_dim(self):
        with self.assertRaises(LeaspyModelInputError):
            TimeReparametrizedModel("dummy", source_dimension=-1)

        with self.assertRaises(LeaspyModelInputError):
            TimeReparametrizedModel("dummy", source_dimension=0.5)

        m = TimeReparametrizedModel("dummy", source_dimension=3)

        mock_dataset = MockDataset(["ft_1", "ft_2", "ft_3"])

        with self.assertRaisesRegex(ValueError, "source_dimension"):
            # source_dimension should be < dimension
            m.initialize(mock_dataset)

        m = TimeReparametrizedModel("logistic")
        m._validate_compatibility_of_dataset(mock_dataset)
        self.assertEqual(m.source_dimension, 1)  # int(sqrt(3))
