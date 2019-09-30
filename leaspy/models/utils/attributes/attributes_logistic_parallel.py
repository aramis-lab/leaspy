import torch


## TODO 1 : Have a Abtract Attribute class
## TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class Attributes_LogisticParallel:

    def __init__(self, dimension, source_dimension):
        self.dimension = dimension
        self.source_dimension = source_dimension
        self.g = None  # g = exp(realizations['g']) tel que p0 = 1 / (1+exp(g))
        self.deltas = None  # deltas = [0, delta_2_realization, ..., delta_n_realization]
        self.xi_mean = None  # v0 is a scalar value, which corresponds to the the first dimension of the velocity vector
        self.betas = None
        self.orthonormal_basis = None
        self.mixing_matrix = None  # Matrix A tq w_i = A * s_i

    def get_attributes(self):
        return self.g, self.deltas, self.mixing_matrix

    def update(self, names_of_changed_values, values):
        self._check_names(names_of_changed_values)

        compute_g = False
        compute_v0 = False
        compute_deltas = False
        compute_betas = False

        for name in names_of_changed_values:
            if name == 'g':
                compute_g = True
            elif name == 'deltas':
                compute_deltas = True
            elif name == 'betas':
                compute_betas = True
            elif name == 'xi_mean':
                compute_v0 = True
            elif name == 'all':
                compute_g = True
                compute_deltas = True
                compute_v0 = True
                compute_betas = True

        if compute_g:
            self._compute_g(values)
        if compute_deltas:
            self._compute_deltas(values)
        if compute_v0:
            self._compute_xi_men(values)
        if compute_betas:
            self._compute_betas(values)

        if compute_g or compute_deltas or compute_v0:
            self._compute_orthonormal_basis()
        if compute_g or compute_deltas or compute_v0 or compute_betas:
            self._compute_mixing_matrix()

    def _check_names(self, names_of_changed_values):
        for name in names_of_changed_values:
            if name not in ['g', 'deltas', 'betas', 'xi_mean', 'all']:
                raise ValueError("The name {} is not in the attributes that are used to be updated".format(name))

    def _compute_xi_men(self, values):
        self.xi_mean = torch.exp(torch.tensor([values['xi_mean']], dtype=torch.float32))

    def _compute_g(self, values):
        self.g = torch.exp(values['g'])

    def _compute_deltas(self, values):
        self.deltas = torch.cat((torch.tensor([0], dtype=torch.float32), values['deltas']))

    def _compute_betas(self, values):
        if self.source_dimension == 0:
            return
        self.betas = torch.tensor(values['betas'], dtype=torch.float32).clone()

    def _compute_dgamma_t0(self):
        # Computes the derivative of gamma_0 at time t0
        exp_d = torch.exp(-self.deltas)
        sub = 1. + self.g * exp_d
        dgamma_t0 = self.xi_mean * self.g * exp_d / (sub * sub)
        return dgamma_t0

    def _compute_orthonormal_basis(self):
        # Compute the basis orthogonal to v0 for the inner product implied by the metric
        # It is equivalent to be a base orthogonal to v0 / (p0^2 (1-p0)^2 for the euclidean norm
        if self.source_dimension == 0:
            return

        # Compute the derivative of gamma_0 at t0
        dgamma_t0 = self._compute_dgamma_t0()

        # Compute regularizer to work in the euclidean space
        gamma_t0 = 1. / (1 + self.g * torch.exp(-self.deltas))
        metric_normalization = gamma_t0.pow(2) * (1 - gamma_t0).pow(2)
        dgamma_t0 = dgamma_t0 / metric_normalization

        # Compute Q
        e1 = torch.zeros(self.dimension)
        e1[0] = 1
        alpha = torch.sign(dgamma_t0[0]) * torch.norm(dgamma_t0)
        u_vector = dgamma_t0 - alpha * e1
        v_vector = u_vector / torch.norm(u_vector)
        v_vector = v_vector.reshape(1, -1)

        q_matrix = torch.eye(self.dimension) - 2 * v_vector.permute(1, 0) * v_vector
        self.orthonormal_basis = q_matrix[:, 1:]

    @staticmethod
    def _mixing_matrix_utils(linear_combination_values, matrix):
        return torch.mm(matrix, linear_combination_values)

    def _compute_mixing_matrix(self):
        if self.source_dimension == 0:
            return

        self.mixing_matrix = torch.tensor(self._mixing_matrix_utils(self.betas, self.orthonormal_basis),
                                          dtype=torch.float32)
