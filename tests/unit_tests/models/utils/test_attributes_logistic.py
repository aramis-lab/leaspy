import torch

from leaspy.models.utils.attributes.logistic_attributes import LogisticAttributes
from tests import LeaspyTestCase


class AttributesLogisticTest(LeaspyTestCase):
    def test_constructor(self):
        attributes = LogisticAttributes("logistic", 6, 2)
        self.assertEqual(attributes.dimension, 6)
        self.assertEqual(attributes.source_dimension, 2)
        for attr_name in ("positions", "velocities", "mixing_matrix"):
            self.assertTrue(
                isinstance(getattr(attributes, attr_name), torch.FloatTensor)
            )
            self.assertEqual(len(getattr(attributes, attr_name)), 0)
        self.assertEqual(attributes.orthonormal_basis, None)
        self.assertEqual(attributes.name, "logistic")
        self.assertEqual(
            attributes.update_possibilities, {"all", "g", "v0", "v0_collinear", "betas"}
        )
        self.assertRaises(
            ValueError, LogisticAttributes, "name", "4", 3.2
        )  # with bad type arguments
        self.assertRaises(TypeError, LogisticAttributes)  # without argument

    def check_values_and_get_dimensions(self, values):
        dimension = len(values["g"])
        self.assertEqual(len(values["v0"]), dimension)
        self.assertEqual(values["betas"].shape[0], dimension - 1)
        source_dimension = values["betas"].shape[1]
        self.assertTrue(0 <= source_dimension <= dimension - 1)
        return dimension, source_dimension

    def test_bad_source_dim(self):
        with self.assertRaisesRegex(ValueError, "`source_dimension`"):
            LogisticAttributes(name="logistic", dimension=2, source_dimension=None)

        with self.assertRaisesRegex(ValueError, "`source_dimension`"):
            LogisticAttributes(name="logistic", dimension=2, source_dimension=2)

        with self.assertRaisesRegex(ValueError, "`source_dimension`"):
            LogisticAttributes(name="logistic", dimension=2, source_dimension=0.5)

        with self.assertRaisesRegex(ValueError, "`source_dimension`"):
            LogisticAttributes(name="logistic", dimension=2, source_dimension=-1)

    def test_compute_orthonormal_basis(self, tol=5e-5):
        names = {"all"}
        values = {
            "g": torch.tensor([-3, 2, 0, 3], dtype=torch.float32),
            "betas": torch.tensor(
                [[1, 2, 3], [-0.1, 0.2, 0.3], [-1, 2, -3]], dtype=torch.float32
            ),
            "v0": torch.tensor([-3, 1, 0, -1], dtype=torch.float32),
        }
        dimension, source_dimension = self.check_values_and_get_dimensions(values)
        attributes = LogisticAttributes("logistic", dimension, source_dimension)
        attributes.update(names, values)

        # Test the orthogonality condition
        gamma_t0 = 1 / (1 + attributes.positions)
        dgamma_t0 = attributes.velocities
        # sqrt_metric_norm = attributes.positions / (1 + attributes.positions).pow(2)
        sqrt_metric_norm = gamma_t0 * (1 - gamma_t0)

        orthonormal_basis = attributes.orthonormal_basis
        for i in range(dimension - 1):
            orthonormal_vector = orthonormal_basis[:, i]  # column vector
            # Test orthogonality to dgamma_t0 (metric inner-product)
            self.assertAlmostEqual(
                torch.dot(orthonormal_vector, dgamma_t0 / sqrt_metric_norm**2).item(),
                0,
                delta=tol,
            )
            # Test normality (canonical inner-product)
            self.assertAlmostEqual(
                torch.norm(orthonormal_vector).item(), 1, delta=tol
            )  # /sqrt_metric_norm
            # Test orthogonality to other vectors (canonical inner-product)
            for j in range(i + 1, dimension - 1):
                self.assertAlmostEqual(
                    torch.dot(orthonormal_vector, orthonormal_basis[:, j]).item(),
                    0,
                    delta=tol,
                )  # / sqrt_metric_norm

    def test_mixing_matrix_utils(self, tol=5e-5):
        names = {"all"}
        values = {
            "g": torch.tensor([-3.0, 2.0, 0.0, 1.0], dtype=torch.float32),
            "betas": torch.tensor(
                [[1, 2, 3], [-0.1, 0.2, 0.3], [-1, 2, -3]], dtype=torch.float32
            ),
            "v0": torch.tensor([-3, 1, 0, -1], dtype=torch.float32),
        }
        dimension, source_dimension = self.check_values_and_get_dimensions(values)
        attributes = LogisticAttributes("logistic", dimension, source_dimension)
        attributes.update(names, values)

        gamma_t0 = 1 / (1 + attributes.positions)
        dgamma_t0 = attributes.velocities
        # sqrt_metric_norm = attributes.positions / (1 + attributes.positions).pow(2)
        sqrt_metric_norm = gamma_t0 * (1 - gamma_t0)
        self.assertAlmostEqual(
            torch.norm(
                sqrt_metric_norm
                - attributes.positions / (1 + attributes.positions).pow(2)
            ),
            0,
            delta=tol,
        )

        mixing_matrix = attributes.mixing_matrix
        for mixing_column in mixing_matrix.T:
            self.assertAlmostEqual(
                torch.dot(mixing_column, dgamma_t0 / sqrt_metric_norm**2).item(),
                0,
                delta=tol,
            )

    def test_orthonormal_basis_consistency(self):
        values = {
            "g": torch.tensor([-1.1, 2.2, 0.0, 3.3], dtype=torch.float32),
            "betas": torch.tensor(
                [[0.1, 0.2, 0.3], [-0.1, 0.2, 0.3], [-0.1, 0.2, -0.3]],
                dtype=torch.float32,
            ),
            "v0": torch.tensor([-4.0, -2.8, -4.5, -3.5], dtype=torch.float32),
        }
        dimension, source_dimension = self.check_values_and_get_dimensions(values)

        attributes = LogisticAttributes("logistic", dimension, source_dimension)
        attributes.update({"all"}, values)
        old_velocities = attributes.velocities
        old_BON = attributes.orthonormal_basis
        old_A = attributes.mixing_matrix

        # check shape of BON & A
        self.assertEqual(old_BON.shape, (dimension, dimension - 1))
        self.assertEqual(old_A.shape, (dimension, source_dimension))

        # shift v0 (log of velocities), so the resulting v0 should be collinear to previous one
        # and so the orthonormal basis should be the same!
        new_v0 = values["v0"] - 0.2
        attributes.update({"v0"}, {"v0": new_v0})
        new_velocities = attributes.velocities
        new_BON = attributes.orthonormal_basis
        new_A = attributes.mixing_matrix

        # velocities are different
        self.assertFalse(torch.allclose(old_velocities, new_velocities))
        # but they are collinear
        self.assertTrue(
            attributes._check_collinearity_vectors(old_velocities, new_velocities)
        )
        # and this is not just a random result
        self.assertFalse(
            attributes._check_collinearity_vectors(old_velocities, old_velocities + 0.1)
        )
        # and the orthonormal basis (and mixing matrix) are the same!
        self.assertAllClose(old_BON, new_BON, what="ortho_basis")
        self.assertAllClose(old_A, new_A, what="mixing_matrix")

        # but orthonormal basis & mixing matrix were re-computed!
        self.assertNotEqual(id(old_BON), id(new_BON))
        self.assertNotEqual(id(old_A), id(new_A))

    def test_no_update_of_orthonormal_basis_when_using_v0_collinear_update(self):
        values = {
            "g": torch.tensor([-1.1, 2.2, 0.0, 3.3], dtype=torch.float32),
            "betas": torch.tensor(
                [[0.1, 0.2, 0.3], [-0.1, 0.2, 0.3], [-0.1, 0.2, -0.3]],
                dtype=torch.float32,
            ),
            "v0": torch.tensor([-4.0, -2.8, -4.5, -3.5], dtype=torch.float32),
        }
        dimension, source_dimension = self.check_values_and_get_dimensions(values)

        attributes = LogisticAttributes("logistic", dimension, source_dimension)
        attributes.update({"all"}, values)
        old_velocities = attributes.velocities
        old_BON = attributes.orthonormal_basis
        old_A = attributes.mixing_matrix

        # shift v0 (log of velocities), so the resulting v0 should be collinear to previous one
        # and so the orthonormal basis should be the same!
        new_v0 = values["v0"] - 0.2
        attributes.update({"v0_collinear"}, {"v0": new_v0})
        new_velocities = attributes.velocities
        new_BON = attributes.orthonormal_basis
        new_A = attributes.mixing_matrix

        # velocities are different
        self.assertFalse(torch.allclose(old_velocities, new_velocities))
        # but they are collinear
        self.assertTrue(
            attributes._check_collinearity_vectors(old_velocities, new_velocities)
        )
        # and the orthonormal basis (and mixing matrix) was not re-computed!
        self.assertEqual(id(old_BON), id(new_BON))
        self.assertEqual(id(old_A), id(new_A))

        # consistency when changing betas
        attributes.update({"betas"}, {"betas": values["betas"] + 0.1})
        self.assertEqual(
            id(old_BON), id(attributes.orthonormal_basis)
        )  # not recomputed BON
        self.assertFalse(torch.allclose(old_A, attributes.mixing_matrix))
