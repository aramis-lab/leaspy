from . import Attributes_LogisticParallel, Attributes_Logistic, Attributes_Linear, Attributes_Univariate


class AttributesFactory():

    @staticmethod
    def attributes(name, dimension, source_dimension):
        if type(name) == str:
            name = name.lower()
        else:
            raise AttributeError("The `name` argument must be a string!")

        if name == 'univariate':
            return Attributes_Univariate()
        elif name == 'logistic':
            return Attributes_Logistic(dimension, source_dimension)
        elif name == 'logistic_parallel':
            return Attributes_LogisticParallel(dimension, source_dimension)
        elif name == 'linear':
            return Attributes_Linear(dimension, source_dimension)
        else:
            raise ValueError(
                "The name {} you provided for the attributes is not related to an attribute class".format(name))
