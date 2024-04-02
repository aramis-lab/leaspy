import torch
import numpy as np
from functools import cached_property
from leaspy.utils.weighted_tensor import WeightedTensor
from tests import LeaspyTestCase
from typing import Optional


class WeightedTensorTest(LeaspyTestCase):
    """Unit tests for WeightedTensor."""
    shape = (3, 6, 9)

    value_1_x_3 = [-1.0, 0.0, 1.0]

    weight_1_x_3 = [0, 1, 1]

    value_2_x_3 = [
        [-1.0, 0.0, 1.0],
        [3.14, 42.1, 16.6],
    ]

    weight_2_x_3 = [
        [1, 0, 0],
        [1, 0, 1],
    ]

    value_3_x_3 = [
        [-1.0, 0.0, 1.0],
        [3.14, 42.1, 16.6],
        [102.2, -23.1, -0.2],
    ]

    weight_3_x_3 = [
        [1, 0, 0],
        [1, 0, 2],
        [0, 0, 0],
    ]

    expected_weighted_tensor_operators = (
        "lt", "le", "eq", "ne", "gt", "ge", "add", "sub", "mul", "truediv", "neg", "abs",
    )

    @cached_property
    def random_data(self) -> np.ndarray:
        return np.random.random(self.shape)

    @cached_property
    def random_weights(self) -> np.ndarray:
        return np.random.random(self.shape)

    @property
    def random_weighted_tensor(self) -> WeightedTensor:
        return WeightedTensor(self.random_data, unsafe=False)

    @property
    def random_weighted_tensor_with_weights(self) -> WeightedTensor:
        return WeightedTensor(self.random_data, self.random_weights, unsafe=False)

    @property
    def weighted_tensor_1_x_3(self) -> WeightedTensor:
        return WeightedTensor(self.value_1_x_3, self.weight_1_x_3, unsafe=False)

    @property
    def weighted_tensor_1_x_3_no_weight(self) -> WeightedTensor:
        return WeightedTensor(self.value_1_x_3, unsafe=False)

    @property
    def weighted_tensor_2_x_3(self) -> WeightedTensor:
        return WeightedTensor(self.value_2_x_3, self.weight_2_x_3, unsafe=False)

    @property
    def weighted_tensor_2_x_3_no_weight(self) -> WeightedTensor:
        return WeightedTensor(self.value_2_x_3, unsafe=False)

    @property
    def weighted_tensor_3_x_3(self) -> WeightedTensor:
        return WeightedTensor(self.value_3_x_3, self.weight_3_x_3, unsafe=False)

    @property
    def tensor_3_x_3(self) -> torch.Tensor:
        return torch.tensor(self.value_3_x_3)

    @staticmethod
    def func(x, y):
        """Toy function to test map and map_both methods."""
        return x * 6 + y

    def assert_value_and_weight(
        self,
        result: WeightedTensor,
        expected_value: list,
        expected_weight: Optional[list] = None,
        exact_values_equality: Optional[bool] = True,
        expected_none_weights: Optional[bool] = False,
    ):
        """
        Assert that provided result WeightedTensor has the expected value and weights.

        Parameters
        ----------
        result : WeightedTensor
            The WeightedTensor resulting from some computation.
        expected_value : list
            The expected value of the result, specified as a list.
        expected_weight : list, optional
            The expected weight of the result, specified as a list.
        exact_values_equality : bool, optional
            If True, the value of 'result' should be exactly equal to the 'expected_value'.
            If False, the two should only be sufficiently close.
        expected_none_weights : bool, optional
            If True, the function will check that the result's weight is None.
            If False, it will compare the result's weight against 'expected_weight'.
        """
        compare_func = torch.equal if exact_values_equality else torch.allclose
        self.assertTrue(
            compare_func(
                result.value,
                torch.tensor(expected_value),
            )
        )
        if expected_none_weights:
            self.assertIsNone(result.weight)
        else:
            self.assertTrue(
                torch.equal(
                    result.weight,
                    torch.tensor(expected_weight),
                )
            )

    def generic_wsum_tester(
        self,
        sum_kwargs: dict,
        expected_wsum_value: torch.Tensor,
        expected_wsum_weight: torch.Tensor,
    ):
        """
        Compute the wsum and sum of a 3x3 tensor and check that the results are as expected.

        Parameters
        ----------
        sum_kwargs : dict
            The kwargs to pass to `WeightedTensor.wsum()` and `WeightedTensor.sum()`.
        expected_wsum_value : torch.Tensor
            The expected value of the wsum's value.
        expected_wsum_weight : torch.Tensor
            The expected value of the wsum's weight.
        """
        wsum_value, wsum_weight = self.weighted_tensor_3_x_3.wsum(**sum_kwargs)
        sum_value = self.weighted_tensor_3_x_3.sum(**sum_kwargs)

        self.assertTrue(torch.equal(wsum_value, expected_wsum_value))
        self.assertTrue(torch.equal(wsum_value, sum_value))
        self.assertTrue(torch.equal(wsum_weight, expected_wsum_weight))

    def assert_has_all_operator_defined(self, tensor: WeightedTensor):
        for operator in self.expected_weighted_tensor_operators:
            self.assertTrue(hasattr(tensor, f"__{operator}__"))

    def test_weighted_tensor_instantiation_without_weights(self):
        self.assertTrue(
            torch.equal(
                self.weighted_tensor_1_x_3_no_weight.value,
                torch.Tensor(self.value_1_x_3),
            )
        )
        self.assertIsNone(self.weighted_tensor_1_x_3_no_weight.weight)
        self.assertEqual(
            self.weighted_tensor_1_x_3_no_weight.shape,
            torch.Size([3]),
        )
        self.assertEqual(self.weighted_tensor_1_x_3_no_weight.ndim, 1)
        self.assertEqual(self.weighted_tensor_1_x_3_no_weight.dtype, torch.float32)
        self.assertEqual(self.weighted_tensor_1_x_3_no_weight.device.type, "cpu")
        self.assertFalse(self.weighted_tensor_1_x_3_no_weight.requires_grad)
        self.assert_has_all_operator_defined(self.weighted_tensor_1_x_3_no_weight)

    def test_weighted_tensor_instantiation_with_weights(self):
        self.assertTrue(
            torch.equal(
                self.weighted_tensor_1_x_3.value,
                torch.Tensor(self.value_1_x_3),
            )
        )
        self.assertTrue(
            torch.equal(
                self.weighted_tensor_1_x_3.weight,
                torch.tensor(self.weight_1_x_3, dtype=torch.int64),
            )
        )
        self.assertEqual(self.weighted_tensor_1_x_3.shape, torch.Size([3]))
        self.assertEqual(self.weighted_tensor_1_x_3.ndim, 1)
        self.assertEqual(self.weighted_tensor_1_x_3.dtype, torch.float32)
        self.assertEqual(self.weighted_tensor_1_x_3.device.type, "cpu")
        self.assertFalse(self.weighted_tensor_1_x_3.requires_grad)
        self.assert_has_all_operator_defined(self.weighted_tensor_1_x_3)

    def test_weighted_tensor_instantiation_from_numpy_array_without_weights(self):
        self.assertTrue(
            torch.equal(
                self.random_weighted_tensor.value,
                torch.from_numpy(self.random_data),
            )
        )
        self.assertIsNone(self.random_weighted_tensor.weight)
        self.assertEqual(
            self.random_weighted_tensor.shape,
            torch.Size(self.shape),
        )
        self.assertEqual(
            self.random_weighted_tensor.ndim,
            len(self.shape),
        )
        self.assertEqual(
            self.random_weighted_tensor.dtype,
            torch.float64,
        )
        self.assertEqual(
            self.random_weighted_tensor.device.type,
            "cpu",
        )
        self.assertFalse(self.random_weighted_tensor.requires_grad)
        self.assert_has_all_operator_defined(self.random_weighted_tensor)

    def test_weighted_tensor_instantiation_from_numpy_array_with_weights(self):
        self.assertTrue(
            torch.equal(
                self.random_weighted_tensor_with_weights.value,
                torch.from_numpy(self.random_data),
            )
        )
        self.assertTrue(
            torch.equal(
                self.random_weighted_tensor_with_weights.weight,
                torch.from_numpy(self.random_weights),
            )
        )
        self.assertEqual(
            self.random_weighted_tensor_with_weights.shape,
            torch.Size(self.shape),
        )
        self.assertEqual(
            self.random_weighted_tensor_with_weights.ndim,
            len(self.shape),
        )
        self.assertEqual(
            self.random_weighted_tensor_with_weights.dtype,
            torch.float64,
        )
        self.assertEqual(
            self.random_weighted_tensor_with_weights.device.type,
            "cpu",
        )
        self.assertFalse(self.random_weighted_tensor_with_weights.requires_grad)
        self.assert_has_all_operator_defined(self.random_weighted_tensor_with_weights)

    def test_weighted_tensor_operation(self):
        expected = WeightedTensor([0.0, 1.0, 2.0], unsafe=False)

        self.assertEqual(
            self.weighted_tensor_1_x_3_no_weight + 1,
            expected,
        )
        self.assertEqual(
            self.weighted_tensor_1_x_3_no_weight - (-torch.ones(())),
            expected,
        )
        self.assertEqual(
            (
                1 + (-torch.ones(self.weighted_tensor_1_x_3_no_weight.shape)) *
                (-self.weighted_tensor_1_x_3_no_weight)
            ),
            expected,
        )
        self.assertEqual(
            (
                torch.ones(self.weighted_tensor_1_x_3_no_weight.shape) -
                self.weighted_tensor_1_x_3_no_weight * (-1)
            ),
            expected,
        )

    def test_weighted_tensor_operation_between_weighted_tensors(self):
        t1 = WeightedTensor(
            self.weighted_tensor_1_x_3_no_weight.value,
            abs(self.weighted_tensor_1_x_3_no_weight).value.to(int),
            unsafe=False,
        )
        t2 = WeightedTensor(self.weighted_tensor_1_x_3_no_weight.value, t1.weight, unsafe=False)

        self.assertEqual(t1 - t2, 0)
        self.assertEqual(t1 + t2, 2 * t1)
        self.assertEqual(-t1 + 2 * t2, t2)
        self.assertEqual(t1 * t2, t1**2)
        for t in (t1, t2):
            self.assert_has_all_operator_defined(t)

    def test_weighted_tensor_identities(self):
        t1 = WeightedTensor(
            self.weighted_tensor_1_x_3_no_weight.value,
            abs(self.weighted_tensor_1_x_3_no_weight).value.to(int),
            unsafe=False,
        )
        t2 = WeightedTensor(
            self.weighted_tensor_1_x_3_no_weight.value,
            t1.weight,
            unsafe=False,
        )
        old_id1 = id(t1)

        self.assertIsNot(t1, t2)
        self.assertIsNot(t1.value, t2.value)

        t1 -= t2

        self.assertNotEqual(id(t1), old_id1)
        self.assertIsNot(t1.value, t2.value)
        for t in (t1, t2):
            self.assert_has_all_operator_defined(t)

    def test_weighted_tensor_comparisons(self):
        t1 = WeightedTensor(
            self.weighted_tensor_1_x_3_no_weight.value,
            abs(self.weighted_tensor_1_x_3_no_weight).value.to(int),
            unsafe=False,
        )
        t2 = WeightedTensor(self.weighted_tensor_1_x_3_no_weight.value, t1.weight, unsafe=False)

        self.assertEqual(t1, t2)
        self.assertLessEqual(t1 - 0.2, t2)
        self.assertLess(t1 - 0.2, t2)
        self.assertGreater(t1 + t2, t1)
        t1 -= t2
        self.assertGreaterEqual(t2, t1)
        for t in (t1, t2):
            self.assert_has_all_operator_defined(t)

    def test_weighted_tensor_filled(self):
        self.assertTrue(
            torch.equal(
                self.weighted_tensor_2_x_3.filled(None),
                self.weighted_tensor_2_x_3.value,
            )
        )
        self.assertTrue(
            torch.equal(
                self.weighted_tensor_2_x_3_no_weight.filled(0.0),
                self.weighted_tensor_2_x_3_no_weight.value,
            )
        )
        self.assertTrue(
            torch.equal(
                self.weighted_tensor_2_x_3.filled(0.0),
                torch.tensor([[-1.0, 0.0, 0.0], [3.14, 0.0, 16.6]]),
            )
        )
        self.assert_has_all_operator_defined(self.weighted_tensor_2_x_3)

    def test_weighted_tensor_valued(self):
        self.assert_value_and_weight(
            self.weighted_tensor_2_x_3.valued(
                torch.ones_like(self.weighted_tensor_2_x_3_no_weight.value)
            ),
            [[1., 1., 1.], [1., 1., 1.]],
            expected_weight=[[1, 0, 0], [1, 0, 1]],
        )

    def test_weighted_tensor_map(self):
        self.assert_value_and_weight(
            self.weighted_tensor_2_x_3.map(self.func, torch.ones((2, 3))),
            [[-5.0, 1.0, 7.0], [19.84, 253.6, 100.6]],
            expected_weight=[[1, 0, 0], [1, 0, 1]],
            exact_values_equality=False,
        )

    def test_weighted_tensor_map_with_fill_value(self):
        self.assert_value_and_weight(
            self.weighted_tensor_2_x_3.map(self.func, torch.ones((2, 3)), fill_value=100.0),
            [[-5.0, 601.0, 601.0], [19.84, 601.0, 100.6]],
            expected_weight=[[1, 0, 0], [1, 0, 1]],
            exact_values_equality=False,
        )

    def test_weighted_tensor_map_no_weight_with_fill_value(self):
        self.assert_value_and_weight(
            self.weighted_tensor_2_x_3_no_weight.map(self.func, torch.ones((2, 3)), fill_value=200),
            [[-5.0, 1.0, 7.0], [19.84, 253.6, 100.6]],
            exact_values_equality=False,
            expected_none_weights=True,
        )

    def test_weighted_tensor_map_both(self):
        self.assert_value_and_weight(
            self.weighted_tensor_2_x_3.map_both(self.func, torch.ones((2, 3))),
            [[-5.0, 1.0, 7.0], [19.84, 253.6, 100.6]],
            expected_weight=[[7., 1., 1.], [7., 1., 7.]],
            exact_values_equality=False,
        )

    def test_weighted_tensor_map_both_with_fill_value(self):
        self.assert_value_and_weight(
            self.weighted_tensor_2_x_3.map_both(
                self.func, torch.ones((2, 3)),
                fill_value=100.0,
            ),
            [[-5.0, 601.0, 601.0], [19.84, 601.0, 100.6]],
            expected_weight=[[7., 1., 1.], [7., 1., 7.]],
            exact_values_equality=False,
        )

    def test_weighted_tensor_map_both_no_weight_with_fill_value(self):
        self.assert_value_and_weight(
            self.weighted_tensor_2_x_3_no_weight.map_both(
                self.func, torch.ones((2, 3)),
                fill_value=200,
            ),
            [[-5.0, 1.0, 7.0], [19.84, 253.6, 100.6]],
            exact_values_equality=False,
            expected_none_weights=True,
        )

    def test_weighted_tensor_wsum_no_args(self):
        self.generic_wsum_tester({}, torch.tensor(35.3400), torch.tensor(4))

    def test_weighted_tensor_wsum_axis_0(self):
        self.generic_wsum_tester({"axis": 0}, torch.tensor([2.14, 0.0, 33.2]), torch.tensor([2, 0, 2]))

    def test_weighted_tensor_wsum_axis_1(self):
        self.generic_wsum_tester({"axis": 1}, torch.tensor([-1.0, 36.34, 0.0]), torch.tensor([1, 3, 0]))

    def test_weighted_tensor_wsum_fill_value(self):
        self.generic_wsum_tester({"fill_value": 1.5}, torch.tensor(35.3400), torch.tensor(4))

    def test_weighted_tensor_wsum_fill_value_and_axis_0(self):
        self.generic_wsum_tester(
            {"fill_value": 1.5, "axis": 0},
            torch.tensor([2.14, 1.5, 33.2]),
            torch.tensor([2, 0, 2]),
        )

    def test_weighted_tensor_wsum_fill_value_and_axis_1(self):
        self.generic_wsum_tester(
            {"fill_value": 1.5, "axis": 1},
            torch.tensor([-1.0, 36.34, 1.5]),
            torch.tensor([1, 3, 0]),
        )

    def test_weighted_tensor_view_1_by_9(self):
        self.assert_value_and_weight(
            self.weighted_tensor_3_x_3.view((1, 9)),
            [[-1.0000, 0.0000, 1.0000, 3.1400, 42.1000, 16.6000, 102.2000, -23.1000, -0.2000]],
            expected_weight=[[1, 0, 0, 1, 0, 2, 0, 0, 0]],
        )

    def test_weighted_tensor_view_9_by_1(self):
        self.assert_value_and_weight(
            self.weighted_tensor_3_x_3.view((9, 1)),
            [[-1.0], [0.0],[1.0], [3.14], [42.1], [16.6], [102.2], [-23.1], [-0.2]],
            expected_weight=[[1], [0], [0], [1], [0], [2], [0], [0], [0]],
        )

    def test_weighted_tensor_expand(self):
        input_tensor = WeightedTensor(
            [[-1.0, 0.0, 1.0]],
            [[1, 0, 0]],
            unsafe=False,
        )

        self.assert_value_and_weight(
            input_tensor.expand((3, -1)),
            [[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]],
            expected_weight=[[1, 0, 0], [1, 0, 0], [1, 0, 0]],
        )

    def test_get_filled_value_and_weight_tensor_without_weights(self):
        values, weights = WeightedTensor.get_filled_value_and_weight(self.tensor_3_x_3)

        self.assertTrue(torch.equal(values, self.tensor_3_x_3))
        self.assertIsNone(weights)

    def test_get_filled_value_and_weight_tensor_without_weights_and_fill_value(self):
        values, weights = WeightedTensor.get_filled_value_and_weight(self.tensor_3_x_3, fill_value=1000.0)

        self.assertTrue(torch.equal(values, self.tensor_3_x_3))
        self.assertIsNone(weights)

    def test_get_filled_value_and_weight_tensor_with_weights(self):
        values, weights = WeightedTensor.get_filled_value_and_weight(self.weighted_tensor_3_x_3)

        self.assertTrue(torch.equal(values, self.tensor_3_x_3))
        self.assertTrue(
            torch.equal(
                weights,
                torch.tensor([[1, 0, 0], [1, 0, 2], [0, 0, 0]])
            )
        )

    def test_get_filled_value_and_weight_tensor_with_weights_and_fill_value(self):
        values, weights = WeightedTensor.get_filled_value_and_weight(
            self.weighted_tensor_3_x_3,
            fill_value=1000.0,
        )

        self.assertTrue(
            torch.equal(
                values,
                torch.tensor(
                    [
                        [-1.0, 1000.0, 1000.0],
                        [3.14, 1000.0, 16.6],
                        [1000.0, 1000.0, 1000.0],
                    ]
                )
            )
        )
        self.assertTrue(
            torch.equal(
                weights,
                torch.tensor([[1, 0, 0], [1, 0, 2], [0, 0, 0]])
            )
        )
