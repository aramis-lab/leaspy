import re

from leaspy.utils.functional import NamedInputFunction
from tests import LeaspyTestCase


def custom_mult(x, y):
    """Custom mult."""
    return x * y


def multi_output_function(x, y):
    """Function returning two outputs."""
    return x * y, x + y


def custom_kws_only_mult(*, x, y):
    """Custom kws only mult."""
    return x * y


def model(x, y, *, foo, bar=10, baz=16):
    """Fake dummy model."""
    return x * y + foo * (bar + baz)


class NamedInputFunctionTest(LeaspyTestCase):
    func = NamedInputFunction(custom_mult, ("foo", "bar"))
    multi_output_func = NamedInputFunction(multi_output_function, ("foo", "bar"))
    model = NamedInputFunction(model, ("x", "y"), {"foo": 100, "baz": 200})

    def test_instantiation(self):
        self.assertEqual(self.func.f.__name__, "custom_mult")
        self.assertNotEqual(self.func.__doc__, "Custom mult.")
        self.assertEqual(self.func.f.__doc__, "Custom mult.")
        self.assertEqual(self.func.parameters, ("foo", "bar"))
        self.assertIsNone(self.func.kws)
        self.assertEqual(self.model.f.__name__, "model")
        self.assertEqual(self.model.parameters, ("x", "y"))
        self.assertEqual(self.model.kws, {"foo": 100, "baz": 200})

    def test_call(self):
        self.assertEqual(self.func(foo=2, bar=3), 6)
        self.assertEqual(self.func(foo=2, bar=7, baz=10), 14)
        self.assertEqual(self.model(x=2, y=3), 21006)
        self.assertEqual(self.model(x=2, y=3, bar=10000), 21006)
        self.assertEqual(self.model(x=2, y=3, foo=-20), 21006)
        self.assertEqual(self.model(x=2, y=3, baz=1000000), 21006)

    def test_key_error_in_calls(self):
        with self.assertRaisesRegex(KeyError, "foo"):
            self.func(foobar=2)

        with self.assertRaisesRegex(KeyError, "bar"):
            self.func(foo=3)

    def test_bound_to(self):
        custom_mult_named = NamedInputFunction.bound_to(custom_mult)
        custom1 = custom_mult_named("x", "y")
        custom2 = custom_mult_named("foo", "baz")

        self.assertEqual(custom1(x=2, y=3), 6)
        self.assertEqual(custom2(foo=2, baz=3), 6)

        model_named = NamedInputFunction.bound_to(model)
        model1 = model_named("x", "y", foo=100, baz=200)
        model2 = model_named("first", "second", foo=100, baz=200)
        self.assertEqual(model1(x=2, y=3), 21006)
        self.assertEqual(model2(first=2, second=3), 21006)

        custom_kws_only_mult_named = NamedInputFunction.bound_to(custom_kws_only_mult)
        custom_kws_only_mult_1 = custom_kws_only_mult_named(x=10, y=6)
        self.assertEqual(custom_kws_only_mult_1(), 60)
        self.assertEqual(
            custom_kws_only_mult_named.__name__,
            "symbolic_custom_kws_only_mult_factory",
        )
        self.assertIn(
            "Factory of a `NamedInputFunction`, bounded to `custom_kws_only_mult`.",
            custom_kws_only_mult_named.__doc__,
        )

    def test_bound_to_with_argument_checker(self):
        from leaspy.utils.functional._utils import _arguments_checker

        custom_mult_named = NamedInputFunction.bound_to(
            custom_mult,
            _arguments_checker(
                nb_arguments=1,
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "custom_mult: Single name expected for positional parameters",
        ):
            custom_mult_named("x", "y")

    def test_composition(self):
        def h(x, *, b):
            return x * 2 + b

        h_o_func = self.func.then(h, b=10)

        self.assertEqual(h_o_func.parameters, ("foo", "bar"))
        self.assertIsNone(h_o_func.kws)
        self.assertEqual(h_o_func.f.__name__, "h@custom_mult")
        self.assertEqual(h_o_func(foo=2, bar=3), 22)

    def test_composition_outer_function_expects_more_values_than_what_inner_function_produces_error(
        self,
    ):
        """
        func produces a single scalar value while h expects two args.
        """

        def h(x, y, *, b):
            return x * 2 + y * 3 + b

        h_o_func = self.func.then(h, b=10)

        self.assertEqual(h_o_func.parameters, ("foo", "bar"))
        self.assertIsNone(h_o_func.kws)
        self.assertEqual(h_o_func.f.__name__, "h@custom_mult")

        with self.assertRaisesRegex(
            TypeError,
            re.escape("h() missing 1 required positional argument: 'y'"),
        ):
            h_o_func(foo=2, bar=3)

    def test_composition_inner_function_returns_more_values_than_what_outer_function_expects_error(
        self,
    ):
        """
        multi_output_func returns two values which are passed as a tuple to h.
        Here, the composition will fail because h assumes x is not a tuple.
        """

        def h(x, *, b):
            return x * 2 + b

        h_o_func = self.multi_output_func.then(h, b=10)

        self.assertEqual(h_o_func.parameters, ("foo", "bar"))
        self.assertIsNone(h_o_func.kws)
        self.assertEqual(h_o_func.f.__name__, "h@multi_output_function")

        with self.assertRaisesRegex(TypeError, "can only concatenate tuple"):
            h_o_func(foo=2, bar=3)

    def test_composition_inner_function_returns_multiple_values_outer_handles_them_as_tuple(
        self,
    ):
        """
        multi_output_func returns two values which are passed as a tuple to h.
        Here, the composition will succeed because h assumes x is a tuple.
        """

        def h(x, *, b):
            return x[0] * 2 + x[1] * 3 + b

        h_o_func = self.multi_output_func.then(h, b=10)

        self.assertEqual(h_o_func.parameters, ("foo", "bar"))
        self.assertIsNone(h_o_func.kws)
        self.assertEqual(h_o_func.f.__name__, "h@multi_output_function")
        self.assertEqual(h_o_func(foo=2, bar=3), 37)
