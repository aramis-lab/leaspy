import numpy as np
import torch


class Result:
    """
    Result object class
    Used as output from personalize & simulate
    """

    def __init__(self, data, individual_parameters, noise_std=None):
        """
        Process the initializer function - called by Leaspy.personalize & Leaspy.simulate

        :param data: leaspy.inputs.data.data object
        :param individual_parameters: dictionary of torch.tensor (or list of float) - containing log-acceleration 'xi', time-shifts 'tau' & 'sources'
        :param noise_std: float - desired noise standard deviation level
        """
        self.data = data
        self.individual_parameters = individual_parameters
        self.noise_std = noise_std

    # def load_covariables(self, covariables, csv):
    #    self.covariables = covariables

    def get_parameter_distribution(self, parameter, cofactor=None):
        """
        Return the wanted parameter distribution (one distribution per covariate state)

        :param parameter: string - corresponding to the wanted parameter's name (ex: 'xi', 'tau' ...)
        :param cofactor: string - corresponding the wanted cofactor's name
        :return: list of floats or dictionary or list of float
            If no cofactor is given & the parameter is univariate => return a list the parameter's distribution
            If no cofactor is given & the parameter is multivariate => return a dictionary =
                {'parameter1': distribution of parameter variable 1, 'parameter2': ...}
            If a cofactor is given & the parameter is univariate => return a dictionary =
                {'cofactor1': parameter distribution such that patient.covariate = covariate1, 'cofactor2': ...}
            If a cofactor is given & the parameter is multivariate => return a dictionary =
                {'cofactor1': {'parameter1': ..., 'parameter2': ...}, 'cofactor2': { ...}, ...}
        """
        # parameter_distribution = self.individual_parameters[parameter].detach().numpy().tolist() ? Why detach ?
        try:
            parameter_distribution = self.individual_parameters[parameter].numpy()
        except AttributeError:
            parameter_distribution = self.individual_parameters[parameter]
        # parameter_distribution is of size (N_subjects, N_dimension_of_parameter)

        # Check the tensor's dimension is <= 2
        if parameter_distribution.ndim > 2:
            raise ValueError('The chosen parameter %s is a tensor of dimension %d - it must be 1 or 2 dimensional!' %
                             (parameter, parameter_distribution.ndim))
        ##############################################
        # If there is no cofactor to take into account
        ##############################################
        if cofactor is None:
            # If parameter is 1-dimensional => use numpy.ravel
            if parameter_distribution.shape[1] == 1:
                # return a list of length = N_subjects
                parameter_distribution = parameter_distribution.ravel().tolist()
            # Else transpose it and split it in a dictionary
            else:
                # return {'parameter1': distribution of parameter variable 1, 'parameter2': ... }
                parameter_distribution = {parameter + str(i): val for i, val in
                                          enumerate(parameter_distribution.T.tolist())}
            return parameter_distribution

        ############################################################
        # If the distribution as asked for different cofactor values
        ############################################################
        # Get possible covariate stats
        cofactor_state = np.unique([_.cofactors[cofactor] for _ in self.data if _.cofactors[cofactor] is not None])

        # Initialize the result
        distributions = {}

        # If parameter 1-dimensional => use numpy.ravel
        if parameter_distribution.shape[1] == 1:
            parameter_distribution = parameter_distribution.ravel() # ex: [1, 2, 3]
            # Create one entry per cofactor state
            for p in cofactor_state:
                if p not in distributions.keys():
                    distributions[p] = []
                # For each covariate state, get parameter distribution
                for i, v in enumerate(parameter_distribution):
                    if self.data[i].cofactors[cofactor] == p:
                        distributions[p].append(v)
                        # return {'cofactor1': ..., 'cofactor2': ...}
        else:
            # Create one dictionary per cofactor state
            for p in cofactor_state:
                if p not in distributions.keys():
                    # Create one dictionary per parameter dimension
                    distributions[p] = {parameter + str(i): [] for i in range(parameter_distribution.shape[1])}
                # Fill these entries by the corresponding values of the corresponding subject
                for i, v in enumerate(parameter_distribution):
                    if self.data[i].cofactors[cofactor] == p:
                        for j, key in enumerate(distributions[p].keys()):
                            distributions[p][key].append(v[j])
                            # return {'cofactor1': {'parameter1': .., 'parameter2': ..}, 'cofactor2': { .. }, .. }
        return distributions

    def get_cofactor_distribution(self, cofactor):
        """
        Get the list of the cofactor's distribution

        :param cofactor: string - cofactor's name
        :return: list of float - cofactor's distribution
        """
        return [d.cofactors[cofactor] for d in self.data]

    def get_patient_individual_parameters(self, idx):
        """
        Get the dictionary of the wanted patient's individual parameters

        :param idx: string - id of the wanted patient
        :return: dictionary of torch.tensor - patient's individual parameters
        """
        indices = list(self.data.individuals.keys())
        idx_number = int(
            np.where(np.array(indices) == idx)[0])  # TODO save somewhere the correspondance : in data probably ???

        patient_dict = dict.fromkeys(self.individual_parameters.keys())

        for variable_ind in list(self.individual_parameters.keys()):
            patient_dict[variable_ind] = self.individual_parameters[variable_ind][idx_number]

        return patient_dict

    def get_error_distribution(self, model, cofactor=None, aggregate_subscores=False, aggregate_visits=False):
        """
        Get error distribution per patient. By default, return one error value per patient & per subscore & per visit.
        Use 'aggregate_subscores' to get one error value per patient & per visit.
        Use 'aggregate_visits' to get one error value per patient & per subscore.
        Use both to have one error value per patient.
        Use `cofactor' to cluster the patients by their corresponding cofactor's state.

        :param model: leaspy model class object
        :param cofactor: string
        :param aggregate_subscores: boolean = False by default  =>  1 error per subscore
        :param aggregate_visits: boolean = False by default  =>  1 error per visit
        :return: if cofactor is None => return a dictionary of torch tensor {'patient1': error1, ...}
            if cofactor is not None => return a dictionary dictionary of torch tensor {'cofactor1': {'patient1': error1, ...}, ...}
        """
        error_distribution = {}
        for i, (key, patient) in enumerate(self.data.individuals.items()):
            param_ind = {'tau': self.individual_parameters['tau'][i],
                         'xi': self.individual_parameters['tau'][i],
                         'sources': self.individual_parameters['sources'][i]}
            if aggregate_subscores:
                if aggregate_visits: # One value per patient
                    error_distribution[key] = torch.sum(model.compute_individual_tensorized(
                        torch.tensor(patient.timepoints), param_ind) - torch.tensor(patient.observations)).tolist()
                else: # One value per patient & per subscore
                    error_distribution[key] = torch.sum(model.compute_individual_tensorized(
                        torch.tensor(patient.timepoints), param_ind) - torch.tensor(patient.observations), 0).tolist()
            elif aggregate_visits: # One value per patient & per visit
                error_distribution[key] = torch.sum(model.compute_individual_tensorized(
                    torch.tensor(patient.timepoints), param_ind) - torch.tensor(patient.observations), 1).tolist()
            else: # One value per patient & per subscore & per visit
                error_distribution[key] = (model.compute_individual_tensorized(
                    torch.tensor(patient.timepoints), param_ind) - torch.tensor(patient.observations)).tolist()

        if cofactor:
            cofactors = self.get_cofactor_distribution(cofactor)
            result = {state: {} for state in np.unique(cofactors)}
            for key in result.keys():
                result[key] = {patient: error_distribution[patient] for i, patient in enumerate(error_distribution.keys())
                               if cofactors[i] == key}
            return result # return {'cofactor1': {'patient1': error1, ...}, ...}
        else:
            return error_distribution # return {'patient1': error1, ...}
