import torch

from tests import LeaspyTestCase


class UtilityTest(LeaspyTestCase):
    """Unit tests for function _arguments_checker."""

    def test_get_dim_error(self):
        from leaspy.utils.weighted_tensor._utils import _get_dim

        with self.assertRaisesRegex(
            ValueError,
            "`dim` and `but_dim` should not be both defined.",
        ):
            _get_dim(torch.ones(2, 2), dim=1, but_dim=3)

    def test_get_dim(self):
        from leaspy.utils.weighted_tensor._utils import _get_dim

        self.assertEqual(_get_dim(torch.ones(2, 3)), ())
        self.assertEqual(_get_dim(torch.ones(2, 3), dim=1, but_dim=None), 1)
        self.assertEqual(_get_dim(torch.ones(2, 3), dim=None, but_dim=0), (1,))
        self.assertEqual(_get_dim(torch.ones(2, 3), dim=None, but_dim=1), (0,))
        self.assertEqual(_get_dim(torch.ones(2, 3), dim=None, but_dim=2), (0, 1))
        self.assertEqual(_get_dim(torch.ones(2, 3), dim=None, but_dim=6), (0, 1))
        self.assertEqual(_get_dim(torch.ones(2, 3), dim=None, but_dim=1000), (0, 1))

        self.assertEqual(_get_dim(torch.ones(2, 3, 6)), ())
        self.assertEqual(_get_dim(torch.ones(2, 3, 6), dim=1, but_dim=None), 1)
        self.assertEqual(_get_dim(torch.ones(2, 3, 6), dim=None, but_dim=0), (1, 2))
        self.assertEqual(_get_dim(torch.ones(2, 3, 6), dim=None, but_dim=1), (0, 2))
        self.assertEqual(_get_dim(torch.ones(2, 3, 6), dim=None, but_dim=2), (0, 1))
        self.assertEqual(_get_dim(torch.ones(2, 3, 6), dim=None, but_dim=6), (0, 1, 2))
        self.assertEqual(
            _get_dim(torch.ones(2, 3, 6), dim=None, but_dim=1000), (0, 1, 2)
        )

        self.assertEqual(_get_dim(torch.ones(2, 3), dim=(1, 2), but_dim=None), (1, 2))
        self.assertEqual(_get_dim(torch.ones(2, 3), dim=None, but_dim=(1, 2)), (0,))
        self.assertEqual(_get_dim(torch.ones(2, 3), dim=(1, 9), but_dim=None), (1, 9))
        self.assertEqual(_get_dim(torch.ones(2, 3), dim=None, but_dim=(1, 9)), (0,))

    def test_expand_right(self):
        from leaspy.utils.weighted_tensor import expand_right

        x = torch.tensor([[1], [2], [3]])

        self.assertEqual(
            expand_right(x, shape=(3, 4)).shape,
            torch.Size([3, 1, 3, 4]),
        )

    def test_expand_left(self):
        from leaspy.utils.weighted_tensor import expand_left

        x = torch.tensor([[1], [2], [3]])

        self.assertEqual(
            expand_left(x, shape=(3, 4)).shape,
            torch.Size([3, 4, 3, 1]),
        )
