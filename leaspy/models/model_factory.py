from . import UnivariateModel, MultivariateModel, MultivariateParallelModel


class ModelFactory():
    """Transition from the model's name to model object"""

    @staticmethod
    def model(name):
        """
        Return the model object corresponding to 'name' arg - check name type and value
        :param name: str - model's name
        :return: model object determined by 'name' - each inherit AbstractModel class
        """
        if type(name) == str:
            name = name.lower()
        else:
            raise AttributeError("The `name` argument must be a string!")

        if name == 'univariate':
            return UnivariateModel(name)
        elif name == 'logistic' or name == 'linear':
            return MultivariateModel(name)
        elif name == 'logistic_parallel':
            return MultivariateParallelModel(name)
        else:
            raise ValueError("The name of the model you are trying to create does not exist! " +
                             "It should be \`logistic\` or \`logistic_parallel\`")
