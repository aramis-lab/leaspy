from .abstract_personalize_algo import AbstractPersonalizeAlgo
from scipy.optimize import minimize
import numpy as np
import torch


class ScipyMinimize(AbstractPersonalizeAlgo):

    def _get_model_name(self, name):
        self.model_name = name

    def _initialize_parameters(self, model):
        x = [model.parameters["xi_mean"], model.parameters["tau_mean"]]
        if self.model_name != "univariate":
            x += [0 for _ in range(model.source_dimension)]
        return np.array(x)

    def _get_attachement(self, model, times, values, x):
        xi = torch.tensor([x[0]], dtype=torch.float32).unsqueeze(0)
        tau = torch.tensor([x[1]], dtype=torch.float32).unsqueeze(0)

        if self.model_name == 'univariate':
            individual_parameters = {'xi': xi, 'tau': tau}
            err = model.compute_individual_tensorized(times, individual_parameters) - values
        else:
            sources = torch.tensor(x[2:], dtype=torch.float32).unsqueeze(0)
            individual_parameters = {'xi': xi, 'tau': tau, 'sources': sources}
            err = model.compute_individual_tensorized(times, individual_parameters) - values
        return err

    def _get_regularity(self, model, x):
        xi = torch.tensor(x[0], dtype=torch.float32)
        tau = torch.tensor(x[1], dtype=torch.float32)
        if self.model_name == 'univariate':
            iterates = zip(['xi', 'tau'], (xi, tau))
        else:
            sources = torch.tensor(x[2:], dtype=torch.float32)
            iterates = zip(['xi', 'tau', 'sources'], (xi, tau, sources))

        regularity = 0
        for key, value in iterates:
            mean = model.parameters["{0}_mean".format(key)]
            std = model.parameters["{0}_std".format(key)]
            regularity += torch.sum(model.compute_regularity_variable(value, mean, std))
        return regularity

    def _get_individual_parameters_patient(self, model, times, values):

        timepoints = times.reshape(1, -1)
        self._get_model_name(model.name)

        def obj(x, *args):
            # Parameters
            model, times, values = args

            # Attachement
            xi = torch.tensor([x[0]], dtype=torch.float32).unsqueeze(0)
            tau = torch.tensor([x[1]], dtype=torch.float32).unsqueeze(0)

            if self.model_name == 'univariate':
                individual_parameters = {'xi': xi, 'tau': tau}
                attachement = model.compute_individual_tensorized(times, individual_parameters) - values
                iterates = zip(['xi', 'tau'], (xi, tau))
            else:
                sources = torch.tensor(x[2:], dtype=torch.float32).unsqueeze(0)
                individual_parameters = {'xi': xi, 'tau': tau, 'sources': sources}
                attachement = model.compute_individual_tensorized(times, individual_parameters) - values
                iterates = zip(['xi', 'tau', 'sources'], (xi, tau, sources))

            attachement = torch.sum(attachement ** 2) / (2. * model.parameters['noise_std'] ** 2)

            # Regularity
            regularity = 0
            for key, value in iterates:
                mean = model.parameters["{0}_mean".format(key)]
                std = model.parameters["{0}_std".format(key)]
                regularity += torch.sum(model.compute_regularity_variable(value, mean, std))

            return (regularity + attachement).detach().numpy()

        initial_value = self._initialize_parameters(model)
        res = minimize(obj,
                       x0=initial_value,
                       args=(model, timepoints, values),
                       method="Powell"
                       )

        if res.success is not True:
            print(res.success, res)

        xi_f, tau_f, sources_f = res.x[0], res.x[1], res.x[2:]
        err_f = self._get_attachement(model, times.unsqueeze(0), values, res.x)
        return (tau_f, xi_f, sources_f), err_f  # TODO depends on the order

    def _get_individual_parameters(self, model, data):

        individual_parameters = {}
        for j, name_variable in enumerate(model.get_individual_variable_name()):
            individual_parameters[name_variable] = []

        total_error = []

        for idx in range(data.n_individuals):
            times = data.get_times_patient(idx)
            values = data.get_values_patient(idx)

            ind_patient, err = self._get_individual_parameters_patient(model, times, values)

            for j, name_variable in enumerate(model.get_individual_variable_name()):
                individual_parameters[name_variable].append(torch.tensor([ind_patient[j]], dtype=torch.float32))

            total_error.append(err.squeeze(0).detach().numpy())

        noise_std = np.std(np.vstack(total_error))
        print(noise_std)

        infos = model.random_variable_informations()
        ## TODO change for cleaner shape update

        out = dict.fromkeys(model.get_individual_variable_name())
        for variable_ind in model.get_individual_variable_name():
            out[variable_ind] = torch.stack(individual_parameters[variable_ind]).reshape(
                shape=(data.n_individuals, infos[variable_ind]['shape'][0]))

        return out
