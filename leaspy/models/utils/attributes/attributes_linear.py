import torch


class Attributes_Linear():
    def __init__(self, dimension, source_dimension):
        self.dimension = dimension
        self.source_dimension = source_dimension
        self.positions = None
        self.velocities = None
        self.orthonormal_basis = None
        self.mixing_matrix = None

    def get_attributes(self):
        return self.positions, self.velocities, self.mixing_matrix

    def update(self, names_of_changed_values, values):
        self._check_names(names_of_changed_values)

        compute_positions = False
        compute_velocities = False
        compute_betas = False

        for name in names_of_changed_values:
            if name == 'g':
                compute_positions = True
            elif name == 'v0':
                compute_velocities = True
            elif name == 'betas':
                compute_betas = True
            elif name == 'all':
                compute_positions = True
                compute_velocities = True
                compute_betas = True

        if compute_positions: self._compute_positions(values)
        if compute_velocities: self._compute_velocities(values)
        if compute_betas: self._compute_betas(values)

        # TODO : Check if the condition is enough
        if compute_velocities:
            self._compute_orthonormal_basis()
        if compute_velocities or compute_betas:
            self._compute_mixing_matrix(values)

    def _check_names(self, names_of_changed_values):
        def raise_err(name):
            raise ValueError("The name {} is not in the attributes that are used to be updated".format(name))

        possibilities = ['all', 'g', 'v0', 'betas']
        [raise_err(n) for n in names_of_changed_values if n not in possibilities]

    def _compute_positions(self, values):
        self.positions = torch.tensor(values['g'], dtype=torch.float32).clone()

    def _compute_velocities(self, values):
        self.velocities = torch.tensor(values['v0'], dtype=torch.float32).clone()

    def _compute_betas(self, values):
        self.betas = torch.tensor(values['betas'], dtype=torch.float32).clone()

    def _compute_orthonormal_basis(self):
        dgamma_t0 = self.velocities

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

    def _compute_mixing_matrix(self, values):
        if self.source_dimension == 0:
            return
        self.mixing_matrix = torch.tensor(self._mixing_matrix_utils(self.betas, self.orthonormal_basis),
                                          dtype=torch.float32)
