from leaspy.utils.realizations.realization import Realization


class CollectionRealization:
    def __init__(self):
        self.realizations = {}

        self.reals_pop_variable_names = []
        self.reals_ind_variable_names = []

    def initialize(self, n_individuals, model):

        # Indices
        infos = model.random_variable_informations()
        for variable, info_variable in infos.items():
            realization = Realization(info_variable['name'], info_variable['shape'], info_variable['type'])
            realization.initialize(n_individuals, model, scale_individual=1.)  ## TODO Check with Raphael
            self.realizations[variable] = realization

        # Name of variables per type
        self.reals_pop_variable_names = [name for name, info_variable in infos.items() if
                                         info_variable['type'] == 'population']
        self.reals_ind_variable_names = [name for name, info_variable in infos.items() if
                                         info_variable['type'] == 'individual']

    def __getitem__(self, variable_name):
        return self.realizations[variable_name]

    def to_dict(self):
        reals_pop = {}
        for pop_var in self.reals_pop_variable_names:
            reals_pop[pop_var] = self.realizations[pop_var].tensor_realizations

        reals_ind = {}
        for i, idx in enumerate(self.indices):
            reals_ind[idx] = {}
            for ind_var in self.reals_ind_variable_names:
                reals_ind[idx][ind_var] = self.realizations[ind_var].tensor_realizations[i]

        return reals_pop, reals_ind

    def keys(self):
        return self.realizations.keys()

    def copy(self):
        new_realizations = CollectionRealization()

        for key in self.keys():
            new_realizations.realizations[key] = self[key].copy()

        return new_realizations

    def initialize_from_values(self, n_individuals, model):

        # Indices
        infos = model.random_variable_informations()
        for variable, info_variable in infos.items():
            realization = Realization(info_variable['name'], info_variable['shape'], info_variable['type'])
            realization.initialize(n_individuals, model, scale_individual=0.01)
            self.realizations[variable] = realization

        # Name of variables per type
        self.reals_pop_variable_names = [name for name, info_variable in infos.items() if
                                         info_variable['type'] == 'population']
        self.reals_ind_variable_names = [name for name, info_variable in infos.items() if
                                         info_variable['type'] == 'individual']
