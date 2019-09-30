import torch


class Realization:
    def __init__(self, name, shape, variable_type):
        self.name = name
        self.shape = shape
        self.variable_type = variable_type
        self._tensor_realizations = None
        self.is_autograd = None

    @classmethod
    def from_tensor(cls, name, shape, variable_type, tensor_realization):
        """
        Create realization from variable infos and torch tensor object
        :param name:
        :param shape:
        :param variable_type:
        :param tensor_realization:
        :return:
        """
        # TODO : a check of shapes
        realization = cls(name, shape, variable_type)
        realization._tensor_realizations = tensor_realization.clone().detach()
        realization.is_autograd = False
        return realization

    def initialize(self, n_individuals, model, scale_individual=1.0):
        # print("Initialize realizations of {0}".format(self.name))
        if self.variable_type == "population":
            self._tensor_realizations = model.parameters[self.name]

            self.is_autograd = False

        elif self.variable_type == 'individual':

            distribution = torch.distributions.normal.Normal(loc=model.parameters["{0}_mean".format(self.name)],
                                                             scale=scale_individual * model.parameters["{0}_std".format(
                                                                 self.name)])  # TODO change later, to have low variance when initialized
            self._tensor_realizations = distribution.sample(sample_shape=(n_individuals, *self.shape))

            self.is_autograd = False
        else:
            raise ValueError("Variable Type Not Known")

    @property
    def tensor_realizations(self):
        return self._tensor_realizations

    @tensor_realizations.setter
    def tensor_realizations(self, tensor_realizations):
        # TODO, check that it is a torch tensor (not variable for example)
        self._tensor_realizations = tensor_realizations

    def set_tensor_realizations_element(self, element, dim):
        # TODO, check that it is a torch tensor (not variable for example) when assigning
        self._tensor_realizations[dim] = element

    def __str__(self):
        str = "Realization of {0} \n".format(self.name)
        str += "Shape : {0} \n".format(self.shape)
        str += "Variable type : {0} \n".format(self.variable_type)
        return str

    def to_torch_Variable(self):
        if not self.is_autograd:
            self._tensor_realizations = torch.autograd.Variable(self._tensor_realizations, requires_grad=True)
            self.is_autograd = True
        else:
            raise ValueError("Realizations are already variables")

    def to_torch_Tensor(self):
        if self.is_autograd:
            self._tensor_realizations = self._tensor_realizations.detach()
            self.is_autograd = False
        else:
            raise ValueError("Realizations are already tensors")

    def copy(self):
        new_realization = Realization(self.name, self.shape, self.variable_type)
        new_realization.is_autograd = self.is_autograd
        new_realization.tensor_realizations = self.tensor_realizations.clone()
        return new_realization
