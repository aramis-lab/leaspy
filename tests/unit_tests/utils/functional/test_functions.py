import re

import torch

from leaspy.utils.functional import Exp, Identity, MatMul, Mean, Std
from tests import LeaspyTestCase


class FunctionsTest(LeaspyTestCase):
    input_tensor = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    id_tensor = torch.eye(3, 3)

    def test_identity(self):
        func = Identity("foo")

        self.assertEqual(func(foo=0), 0)
        self.assertEqual(func(foo="bar"), "bar")

    def test_identity_errors(self):
        func = Identity("foo")

        with self.assertRaisesRegex(KeyError, "foo"):
            func(bar=2)

        with self.assertRaisesRegex(
            ValueError, "Single name expected for positional parameters"
        ):
            Identity("foo", "bar")

    def test_exponential(self):
        func = Exp("x")
        input_tensor = torch.tensor([[0, 1, 3], [2, 5, 6]])

        self.assertEqual(
            func(x=torch.tensor(2)),
            torch.exp(torch.tensor(2)),
        )
        self.assertTrue(
            torch.equal(
                func(x=input_tensor),
                torch.exp(input_tensor),
            )
        )

    def test_exponential_errors(self):
        func = Exp("x")

        with self.assertRaisesRegex(
            TypeError,
            "must be Tensor, not int",
        ):
            func(x=0)

    def test_matmult(self):
        func = MatMul("x", "y")

        self.assertTrue(
            torch.equal(
                func(x=self.id_tensor, y=self.input_tensor),
                self.input_tensor,
            )
        )

    def test_matmult_errors(self):
        func = MatMul("x", "y")

        with self.assertRaisesRegex(
            RuntimeError,
            "both arguments to matmul need to be at least 1D, but they are 0D and 0D",
        ):
            func(x=torch.tensor(2), y=torch.tensor(3))

    def test_mean_first_dim(self):
        func = Mean("tau", dim=0)

        self.assertTrue(
            torch.equal(
                func(tau=self.id_tensor),
                torch.tensor(3 * [1.0 / 3.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                func(tau=self.input_tensor),
                torch.tensor([4.0, 5.0, 6.0]),
            )
        )

    def test_mean_second_dim(self):
        func = Mean("tau", dim=1)

        self.assertTrue(
            torch.equal(
                func(tau=self.id_tensor),
                torch.tensor(3 * [1.0 / 3.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                func(tau=self.input_tensor),
                torch.tensor([2.0, 5.0, 8.0]),
            )
        )

    def test_mean_last_dim(self):
        func = Mean("tau", dim=-1)

        self.assertTrue(
            torch.equal(
                func(tau=self.id_tensor),
                torch.tensor(3 * [1.0 / 3.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                func(tau=self.input_tensor),
                torch.tensor([2.0, 5.0, 8.0]),
            )
        )

    def test_mean_other_dim(self):
        func = Mean("tau", dim=-2)

        self.assertTrue(
            torch.equal(
                func(tau=self.id_tensor),
                torch.tensor(3 * [1.0 / 3.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                func(tau=self.input_tensor),
                torch.tensor([4.0, 5.0, 6.0]),
            )
        )

    def test_mean_errors(self):
        func = Mean("tau", dim=2)

        with self.assertRaisesRegex(
            IndexError,
            re.escape(
                "Dimension out of range (expected to be in "
                "range of [-2, 1], but got 2)"
            ),
        ):
            func(tau=self.id_tensor)

    def test_std(self):
        func = Std("foo_bar_42", dim=0, unbiased=True)

        self.assertTrue(
            torch.allclose(
                func(foo_bar_42=self.id_tensor),
                torch.tensor([0.5774, 0.5774, 0.5774]),
                rtol=1e-04,
            )
        )
        self.assertTrue(
            torch.equal(
                func(foo_bar_42=self.input_tensor),
                torch.tensor([3.0, 3.0, 3.0]),
            )
        )
