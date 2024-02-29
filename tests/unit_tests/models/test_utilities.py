import torch

from tests import LeaspyTestCase


class UtilitiesTest(LeaspyTestCase):
    def test_cast_value_to_2d_tensor(self):
        from leaspy.models.utilities import cast_value_to_2d_tensor

        test_data = torch.tensor([[5]], dtype=torch.float32)

        for x, unsqueeze_dim, expected_out in zip([
            [1, 2], [1, 2], 5, 5, [5], [5]
        ], [0, -1, 0, -1, 0, -1], [
            torch.tensor([[1, 2]], dtype=torch.float32),
            torch.tensor([[1], [2]], dtype=torch.float32),
            test_data, test_data, test_data, test_data
        ]):
            self.assertTrue(
                torch.equal(
                    cast_value_to_2d_tensor(x, unsqueeze_dim=unsqueeze_dim),
                    expected_out
                )
            )
