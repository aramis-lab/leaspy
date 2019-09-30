import torch


## TODO 1 : Have a Abtract Attribute class
## TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class Attributes_Univariate:

    def __init__(self):
        self.g = None  # g is a vector such that p0 = 1 / (1+exp(g)) where p0 is the position vector
        self.v0 = None  # v0 is the vector of velocities

    def get_attributes(self):
        return self.g

    def update(self, names_of_changed_values, values):
        self._check_names(names_of_changed_values)

        compute_g = False
        compute_v0 = False

        for name in names_of_changed_values:
            if name == 'g':
                compute_g = True
            elif name == 'v0':
                compute_v0 = True
            elif name == 'all':
                compute_g = True
                compute_v0 = True

        if compute_g:
            self._compute_g(values)
        if compute_v0:
            self._compute_v0(values)

    def _check_names(self, names_of_changed_values):
        for name in names_of_changed_values:
            if name not in ['all', 'g', 'v0']:
                raise ValueError("The name {} is not in the attributes that are used to be updated".format(name))

    def _compute_g(self, values):
        self.g = torch.exp(values['g'])

    def _compute_v0(self, values):
        self.v0 = torch.exp(torch.tensor([values['xi_mean']], dtype=torch.float32))
