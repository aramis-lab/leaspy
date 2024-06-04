import torch

from leaspy.models.noise_models import (
    NOISE_MODELS,
    NO_NOISE,
    BaseNoiseModel,
    BernoulliNoiseModel,
    noise_model_factory,
    GaussianScalarNoiseModel,
    GaussianDiagonalNoiseModel,
    DistributionFamily,
    OrdinalNoiseModel,
    OrdinalRankingNoiseModel,
)
from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.distributions import MultinomialDistribution

from tests import LeaspyTestCase


class NoiseModelFactoryTest(LeaspyTestCase):
    def test_return_type_from_string_input(self):
        """
        Test that the noise model factory returns noise model
        instances when passing a string.
        """
        for name in NOISE_MODELS.keys():
            with self.subTest(model_name=name):
                model = noise_model_factory(name)
                self.assertIsInstance(model, NOISE_MODELS[name])

    def test_return_type_from_dict_input(self):
        """
        Test that the noise model factory returns noise model
        instances when passing a dictionary.
        """
        for name in NOISE_MODELS.keys():
            with self.subTest(model_name=name):
                model = noise_model_factory({"name": name})
                self.assertIsInstance(model, NOISE_MODELS[name])

    def test_return_type_from_class_input(self):
        """
        Test that the noise model factory returns noise model instances
        when passing a noise model instance directly.
        """
        for model_name, model_class in NOISE_MODELS.items():
            with self.subTest(model_name=model_name):
                model = noise_model_factory(model_class())
                self.assertIsInstance(model, model_class)

    def test_lower_case_input(self):
        """
        Test that the noise model factory is case-insensitive
        and handles "-" vs. "_" properly.
        """
        names = (
            "gaussian-scalar",
            "GauSsiaN_scAlaR",
            "gaussian-SCALAR",
            "GAUSSIAN_scalar",
            "GAUSSIAN-SCALAR",
        )
        for name in names:
            model = noise_model_factory(name)
            self.assertIsInstance(model, GaussianScalarNoiseModel)

    def test_wrong_string_argument(self):
        """
        Test that an error is raised when providing a bad
        string argument to the factory.
        """
        bad_inputs = ("gausian-scalar", "foo", "")  # noqa
        for bad_input in bad_inputs:
            self.assertRaises(
                LeaspyModelInputError, noise_model_factory, bad_input
            )

    def test_wrong_non_string_argument(self):
        """
        Test that an error is raised when providing
        a bad non string argument to the factory.
        """
        bad_inputs = (3.8, {"foo": .1})
        for bad_input in bad_inputs:
            self.assertRaises(
                LeaspyModelInputError, noise_model_factory, bad_input
            )


class TestNoNoise(LeaspyTestCase):

    model = NO_NOISE

    def test_constructor(self):
        self.assertIsInstance(self.model, DistributionFamily)
        self.assertIs(self.model.factory, None)
        self.assertIs(self.model.parameters, None)
        self.assertEqual(len(self.model.free_parameters), 0)

    def test_to_dict(self):
        self.assertEqual(self.model.to_dict(), {})

    def test_sample_around(self):
        for shape in [(), (1,), (2, 3), (5, 2)]:
            t = torch.randn(shape)
            self.assertIs(self.model.sampler_around(t)(), t)
            self.assertIs(self.model.sample_around(t), t)

    def test_rv_around_error(self):
        with self.assertRaises(Exception):
            self.model.rv_around(torch.randn((1,)))


class TestBernoulliNoiseModel(LeaspyTestCase):

    model = BernoulliNoiseModel()

    def test_constructor(self):
        self.assertIsInstance(self.model, BaseNoiseModel)
        self.assertIsInstance(self.model, DistributionFamily)
        self.assertEqual(self.model.factory, torch.distributions.Bernoulli)
        self.assertEqual(len(self.model.free_parameters), 0)
        self.assertIs(self.model.parameters, None)

    def test_to_dict(self):
        self.assertEqual(self.model.to_dict(), {})

    def test_sample_around_shapes(self):
        for shape in [(), (1,), (2, 3), (5, 2)]:
            probs = torch.randn(shape).clamp(min=1e-3, max=1 - 1e-3)
            self.assertEqual(self.model.sample_around(probs).shape, probs.shape)
            self.assertEqual(self.model.sampler_around(probs)().shape, probs.shape)
            self.assertEqual(self.model.rv_around(probs).sample().shape, probs.shape)

    def test_sample_around_zero_and_one(self):
        for loc in (torch.tensor([0., 0.]), torch.tensor([[1.]]), torch.tensor([[0.], [1.]])):
            self.assertTrue(
                torch.equal(self.model.sample_around(loc), loc)
            )

    def test_sample_around_exception_bad_range(self):
        with self.assertRaises(Exception):
            self.model.sample_around(torch.tensor(-1.))

    def test_sample_around_exception_bad_range_for_one_prob(self):
        with self.assertRaises(Exception):
            self.model.sample_around(torch.tensor([[.5, 1.05]]))

    def test_sample_around_exception_bad_type(self):
        with self.assertRaises(Exception):
            self.model.sample_around('0.5')  # noqa


class TestGaussianScalarNoiseModel(LeaspyTestCase):

    model = GaussianScalarNoiseModel()
    scale = torch.tensor([.05])

    def test_constructor(self):
        self.assertIsInstance(self.model, BaseNoiseModel)
        self.assertIsInstance(self.model, DistributionFamily)
        self.assertIs(self.model.scale_dimension, None)
        self.assertEqual(self.model.factory, torch.distributions.Normal)
        self.assertEqual(len(self.model.free_parameters), 1)
        self.assertIs(self.model.parameters, None)

    def test_to_dict(self):
        self.assertEqual(self.model.to_dict(), {})

    def test_constructor_error_bad_param_name(self):
        with self.assertRaises(Exception):
            GaussianScalarNoiseModel(Scale=5.)  # noqa
        with self.assertRaises(Exception):
            GaussianScalarNoiseModel(foo=5.)  # noqa

    def test_constructor_error_bad_extra_param_name(self):
        with self.assertRaises(Exception):
            m = GaussianScalarNoiseModel()
            m.update_parameters(
                validate=True,
                scale=self.scale,
                foo=5.0,  # noqa
            )

    def test_sample_around_errors_in_scale(self):
        with self.assertRaises(Exception):
            m = GaussianScalarNoiseModel()
            m.update_parameters(scale=0.)  # noqa
            m.sample_around(torch.tensor([1.]))

    def test_sample_around_errors_type(self):
        with self.assertRaises(Exception):
            self.model.sample_around('0.5')  # noqa

    def test_non_univariate_error(self):
        """Test that an error is raise when providing a non-univariate scale."""
        with self.assertRaises(Exception):
            self.model.update_parameters(
                validate=True,
                scale=torch.tensor([.1, .5]),
            )


class TestGaussianDiagonalNoiseModel(LeaspyTestCase):

    model = GaussianDiagonalNoiseModel()
    scale = torch.tensor([.03, .07, .02, .081293])

    def test_constructor(self):
        self.assertIsInstance(self.model, BaseNoiseModel)
        self.assertIsInstance(self.model, DistributionFamily)
        self.assertIs(self.model.scale_dimension, None)
        self.assertEqual(self.model.factory, torch.distributions.Normal)
        self.assertEqual(len(self.model.free_parameters), 1)
        self.assertIs(self.model.parameters, None)

    def test_to_dict(self):
        self.assertEqual(self.model.to_dict(), {})

    def test_constructor_error_bad_param_name(self):
        with self.assertRaises(Exception):
            GaussianDiagonalNoiseModel(Scale=5.)  # noqa
        with self.assertRaises(Exception):
            GaussianDiagonalNoiseModel(foo=5.)  # noqa

    def test_constructor_error_bad_extra_param_name(self):
        with self.assertRaises(Exception):
            m = GaussianDiagonalNoiseModel()
            m.update_parameters(
                validate=True,
                scale=self.scale,
                foo=5.0,  # noqa
            )

    def test_sample_around_errors_in_scale(self):
        with self.assertRaises(Exception):
            m = GaussianDiagonalNoiseModel()
            m.update_parameters(scale=0.)  # noqa
            m.sample_around(torch.tensor([1.]))

    def test_sample_around_errors_type(self):
        with self.assertRaises(Exception):
            self.model.sample_around('0.5')  # noqa

    def test_sample_around_incompatibility_loc_and_shape(self):
        with self.assertRaises(Exception):
            self.model.sample_around(torch.zeros((len(self.scale) + 1,)))
        with self.assertRaises(Exception):
            self.model.sample_around(torch.zeros((len(self.scale), 42)))


class TestOrdinalNoiseModel(LeaspyTestCase):

    model = OrdinalNoiseModel()

    def test_constructor(self):
        self.assertIsInstance(self.model, BaseNoiseModel)
        self.assertIsInstance(self.model, DistributionFamily)
        self.assertIs(self.model.max_levels, None)
        self.assertEqual(self.model.factory, MultinomialDistribution.from_pdf)
        self.assertEqual(len(self.model.free_parameters), 0)
        self.assertIs(self.model.parameters, None)

    def test_to_dict(self):
        self.assertEqual(self.model.to_dict(), {'max_levels': None})

    def test_sample_around_shapes(self):
        """Check that shapes are correct."""
        for shape in [(1, 2), (2, 3), (5, 4)]:
            probs = torch.rand(shape)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            rv = self.model.rv_around(probs)
            self.assertIsInstance(rv, MultinomialDistribution)
            sampler = self.model.sampler_around(probs)
            self.assertEqual(
                self.model.sample_around(probs).shape,
                probs.shape[:-1]
            )
            self.assertEqual(sampler().shape, probs.shape[:-1])
            self.assertEqual(rv.sample().shape, probs.shape[:-1])

    def test_sample_around(self):
        for loc, expected in zip(
            [
                torch.tensor([0., 1.]),
                torch.tensor([[1., 0., 0.]]),
                torch.tensor([[0., 0., 1.]]),
                torch.tensor([[0., 1., 0.], [1., 0., 0.]]),
            ],
            [
                torch.tensor(1),
                torch.tensor([0]),
                torch.tensor([2]),
                torch.tensor([1, 0]),
            ]
        ):
            self.assertTrue(
                torch.equal(self.model.sample_around(loc), expected)
            )


class TestOrdinalRankingNoiseModel(LeaspyTestCase):

    model = OrdinalRankingNoiseModel()

    def test_constructor(self):
        self.assertIsInstance(self.model, BaseNoiseModel)
        self.assertIsInstance(self.model, DistributionFamily)
        self.assertIs(self.model.max_levels, None)
        self.assertEqual(self.model.factory, MultinomialDistribution)
        self.assertEqual(len(self.model.free_parameters), 0)
        self.assertIs(self.model.parameters, None)

    def test_to_dict(self):
        self.assertEqual(self.model.to_dict(), {'max_levels': None})

    def test_sample_around_shapes(self):
        """Check that shapes are correct."""
        for shape in [(1, 2), (2, 3), (5, 4)]:
            probs = torch.rand(shape)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            sf = (1. - probs.cumsum(dim=-1)).clamp(min=0, max=1)
            rv = self.model.rv_around(sf)
            self.assertIsInstance(rv, MultinomialDistribution)
            sampler = self.model.sampler_around(sf)
            self.assertEqual(
                self.model.sample_around(sf).shape,
                probs.shape[:-1]
            )
            self.assertEqual(sampler().shape, probs.shape[:-1])
            self.assertEqual(rv.sample().shape, probs.shape[:-1])

    def test_sample_around(self):
        for loc, expected in zip(
            [
                torch.tensor([1., 0.]),
                torch.tensor([[0., 0., 0.]]),
                torch.tensor([[1., 1., 0.]]),
                torch.tensor([[1., 1., 1.], [1., 0., 0.]])
            ],
            [
                torch.tensor(1),
                torch.tensor([0]),
                torch.tensor([2]),
                torch.tensor([3, 1])
            ],
        ):
            self.assertTrue(
                torch.equal(self.model.sample_around(loc), expected)
            )
