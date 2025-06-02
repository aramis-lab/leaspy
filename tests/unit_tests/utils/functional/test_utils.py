import re
from typing import Callable, Optional, Set

from leaspy.utils.functional import MatMul, get_named_parameters
from tests import LeaspyTestCase


class ArgumentCheckerTest(LeaspyTestCase):
    """Unit tests for function _arguments_checker."""

    @staticmethod
    def get_checker(
        nb_arguments: Optional[int] = None,
        mandatory_kws: Optional[Set[str]] = None,
        possible_kws: Optional[Set[str]] = None,
    ) -> Callable:
        from leaspy.utils.functional._utils import _arguments_checker

        return _arguments_checker(
            nb_arguments=nb_arguments,
            mandatory_kws=mandatory_kws,
            possible_kws=possible_kws,
        )

    @property
    def checker_without_kws(self) -> Callable:
        return self.get_checker(nb_arguments=1, possible_kws=set())

    @property
    def checker_with_kws(self) -> Callable:
        return self.get_checker(
            nb_arguments=1,
            possible_kws={"foo", "bar", "baz"},
            mandatory_kws={"foo", "bar"},
        )

    def test_bad_input_value_for_nb_arguments(self):
        with self.assertRaisesRegex(
            ValueError,
            "Number of arguments should be a positive or null integer or None. You provided a <class 'int'>.",
        ):
            self.get_checker(nb_arguments=-1)

    def test_bad_input_type_for_nb_arguments(self):
        with self.assertRaisesRegex(
            ValueError,
            "Number of arguments should be a positive or null integer or None. You provided a <class 'str'>.",
        ):
            self.get_checker(nb_arguments="foo")  # noqa

    def test_missing_mandatory_kws(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Some mandatory kws are not allowed: ['foo']."),
        ):
            self.get_checker(
                nb_arguments=2, mandatory_kws={"foo"}, possible_kws={"bar", "baz"}
            )

    def test_mandatory_kws_not_in_possible_kws(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Some mandatory kws are not allowed: ['baz', 'foo']."),
        ):
            self.get_checker(
                nb_arguments=2, mandatory_kws={"foo", "baz"}, possible_kws={"bar"}
            )

    def test_single_name_expected_for_positional_parameters_empty_input(self):
        with self.assertRaisesRegex(
            ValueError,
            "Single name expected for positional parameters",
        ):
            self.checker_without_kws((), {})

    def test_single_name_expected_for_positional_parameters(self):
        with self.assertRaisesRegex(
            ValueError,
            "Single name expected for positional parameters",
        ):
            self.checker_without_kws((10, 20), {})

    def test_unknown_single_keyword_argument(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Unknown keyword-arguments: ['foo']."),
        ):
            self.checker_without_kws((10,), {"foo": 18})

    def test_unknown_multiple_keyword_argument(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Unknown keyword-arguments: ['bar', 'foo']."),
        ):
            self.checker_without_kws((10,), {"foo": 18, "bar": "baz"})

    def test_missing_single_mandatory_keyword_argument(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Missing mandatory keyword-arguments: ['foo']."),
        ):
            self.checker_with_kws((10,), {"baz": 1, "bar": "foo"})

    def test_missing_multiple_mandatory_keyword_argument(self):
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Missing mandatory keyword-arguments: ['bar', 'foo']."),
        ):
            self.checker_with_kws((10,), {"baz": 1})

    def test_empty(self):
        self.assertIsNone(self.get_checker()((), {}))

    def test_no_kws(self):
        self.assertIsNone(
            self.get_checker(
                nb_arguments=1,
            )(("foo",), {})
        )
        self.assertIsNone(
            self.get_checker(
                nb_arguments=2,
            )(("foo", 42), {})
        )

    def test_with_possible_kws(self):
        self.assertIsNone(
            self.get_checker(
                nb_arguments=2,
                possible_kws={"foo"},
            )(("foo", 42), {})
        )
        self.assertIsNone(
            self.get_checker(
                nb_arguments=2,
                possible_kws={"foo"},
            )(("foo", 42), {"foo": 0.01})
        )
        self.assertIsNone(
            self.get_checker(
                nb_arguments=2,
                possible_kws={"foo", "bar"},
            )(("foo", 42), {"bar": 0.01})
        )

    def test_with_possible_and_mandatory_kws(self):
        self.assertIsNone(
            self.get_checker(
                nb_arguments=2,
                possible_kws={"foo", "bar"},
                mandatory_kws={"bar"},
            )(("foo", 42), {"bar": 0.01})
        )
        self.assertIsNone(
            self.get_checker(
                nb_arguments=2,
                possible_kws={"foo", "bar"},
                mandatory_kws={"bar", "foo"},
            )(("foo", 42), {"foo": [1, 2, 3], "bar": 0.01})
        )


class GetNamedParametersTest(LeaspyTestCase):
    def test_get_named_parameters_error(self):
        def func(x, y, foo=3):
            return x + y + foo

        with self.assertRaises(ValueError):
            get_named_parameters(func)

        with self.assertRaises(ValueError):
            get_named_parameters(lambda x, y: x + y)

    def test_get_named_parameters_custom_function(self):
        def func(*, x, y):
            return x + y

        self.assertEqual(get_named_parameters(func), ("x", "y"))

    def test_get_named_parameters_named_input_function(self):
        self.assertEqual(get_named_parameters(MatMul("foo", "bar")), ("foo", "bar"))
