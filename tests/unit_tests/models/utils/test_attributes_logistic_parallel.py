import torch

from leaspy.models.utils.attributes.logistic_parallel_attributes import LogisticParallelAttributes

from tests import LeaspyTestCase


class AttributesLogisticParallelTest(LeaspyTestCase):

    def test_constructor(self):
        attributes = LogisticParallelAttributes('logistic_parallel', 6, 2)
        self.assertEqual(attributes.dimension, 6)
        self.assertEqual(attributes.source_dimension, 2)
        self.assertTrue(hasattr(attributes, 'velocities'))
        for attr_name in ('positions', 'velocities', 'mixing_matrix'):
            self.assertTrue(isinstance(getattr(attributes, attr_name), torch.FloatTensor))
            self.assertEqual(len(getattr(attributes, attr_name)), 0)
        self.assertEqual(attributes.deltas, None)
        self.assertEqual(attributes.orthonormal_basis, None)
        self.assertEqual(attributes.name, 'logistic_parallel')
        self.assertEqual(attributes.update_possibilities, {'all', 'g', 'betas', 'deltas'})
        self.assertRaises(ValueError, LogisticParallelAttributes, 'name', '4', 3.2)  # with bad type arguments
        self.assertRaises(TypeError, LogisticParallelAttributes)  # without argument

        self.assertRaises(ValueError, attributes._check_names, {'xi_mean'}) # was USELESS so removed
        self.assertRaises(ValueError, attributes._check_names, {'v0'}) # only for multivariate not parallel
        self.assertRaises(ValueError, attributes._check_names, {'v0_collinear'}) # only for multivariate not parallel

    def check_values_and_get_dimensions(self, values):
        dimension = 1 + len(values['deltas'])
        # positions is univariate in this model!
        self.assertEqual(len(values['g']), 1)
        self.assertEqual(values['betas'].shape[0], dimension-1)
        source_dimension = values['betas'].shape[1]
        self.assertTrue(0 <= source_dimension <= dimension-1)
        return dimension, source_dimension

    def compute_instance_and_variables(self):
        names = {'all'}
        values = {
            'g': torch.tensor([0.]),
            'deltas': torch.tensor([-1., 0., 2.]),
            'betas': torch.tensor([[.1, .2, .3], [-.1, .2, .3], [-.1, .2, -.3]]),
        }

        dimension, source_dimension = self.check_values_and_get_dimensions(values)
        attributes = LogisticParallelAttributes('logistic_parallel', dimension, source_dimension)
        attributes.update(names, values)

        # Test the first value of the derivative of gamma at t0
        p0 = 1. / (1. + torch.exp(values['g']))
        collin_standard_v0 = p0 * (1 - p0)  # do not multiply by scalar constants... esp. the removed torch.exp(values['xi_mean'])
        gamma_t0, collin_dgamma_t0 = attributes._compute_gamma_t0_collin_dgamma_t0()

        # Test the orthogonality condition
        #gamma_t0 = 1. / (1 + attributes.positions * torch.exp(-attributes.deltas))
        sqrt_metric_normalization = gamma_t0 * (1 - gamma_t0) # not squared

        return attributes, collin_dgamma_t0, sqrt_metric_normalization, collin_standard_v0

    def test_compute_orthonormal_basis(self):
        attributes, collin_dgamma_t0, sqrt_metric_norm, collin_standard_v0 = self.compute_instance_and_variables()

        # v0 for delta=0 (at dimension 0 and 2 by construction, cf. deltas above)
        self.assertEqual(collin_dgamma_t0[0], collin_standard_v0)
        self.assertEqual(collin_dgamma_t0[2], collin_standard_v0)

        orthonormal_basis = attributes.orthonormal_basis
        for i in range(attributes.dimension-1):
            orthonormal_vector = orthonormal_basis[:, i] # column vector
            # Test orthogonality to dgamma_t0 (metric inner-product)
            self.assertAlmostEqual(torch.dot(orthonormal_vector,
                                             collin_dgamma_t0 / sqrt_metric_norm**2).item(), 0, delta=1e-6) # / sqrt_metric_norm
            # Test normality (canonical inner-product)
            self.assertAlmostEqual(torch.norm(orthonormal_vector).item(), 1, delta=1e-6) # /sqrt_metric_norm
            # Test orthogonality to other vectors (canonical inner-product)
            for j in range(i+1, attributes.dimension-1):
                self.assertAlmostEqual(torch.dot(orthonormal_vector,
                                                 orthonormal_basis[:, j]).item(), 0, delta=1e-6) # / sqrt_metric_norm


    def test_mixing_matrix_utils(self):
        attributes, dgamma_t0, sqrt_metric_norm, _ = self.compute_instance_and_variables()

        mixing_matrix = attributes.mixing_matrix
        for mixing_column in mixing_matrix.permute(1, 0):
            # Test orthogonality to dgamma_t0 (metric inner-product)
            self.assertAlmostEqual(torch.dot(mixing_column, dgamma_t0 / sqrt_metric_norm**2).item(), 0, delta=1e-6)

    def test_orthonormal_basis_consistency(self):

        values = {
            'g': torch.tensor([0.3], dtype=torch.float32),
            'deltas': torch.tensor([-1., 0., 2.]),
            'betas': torch.tensor([[0.1, 0.2, 0.3], [-0.1, 0.2, 0.3], [-0.1, 0.2, -0.3]], dtype=torch.float32),
        }
        dimension, source_dimension = self.check_values_and_get_dimensions(values)

        attributes = LogisticParallelAttributes('logistic_parallel', dimension, source_dimension)
        attributes.update({'all'}, values)

        # shift all possible parameters and check consistence of impact on orthonormal basis / mixing matrix!
        for param, attr_name in (('g','positions'),
                                 ('deltas','deltas'),
                                 ('betas','betas')):

            old_attr = getattr(attributes, attr_name)
            old_BON = attributes.orthonormal_basis
            old_A = attributes.mixing_matrix

            # check shape of BON & A
            self.assertEqual(old_BON.shape, (dimension, dimension - 1))
            self.assertEqual(old_A.shape, (dimension, source_dimension))

            new_v = values[param] + 0.1
            attributes.update({param}, {param: new_v})
            new_attr = getattr(attributes, attr_name)
            new_BON = attributes.orthonormal_basis
            new_A = attributes.mixing_matrix

            # attr are different
            self.assertFalse(torch.allclose(old_attr, new_attr))
            # and the orthonormal basis (and mixing matrix) also depending on cases!
            if param == 'betas':
                # BON not recomputed!
                self.assertEqual(id(old_BON), id(new_BON))
            else:
                self.assertFalse(torch.allclose(old_BON, new_BON))

            self.assertFalse(torch.allclose(old_A, new_A))
