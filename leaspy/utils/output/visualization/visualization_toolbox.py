###### Pas utile ?
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from leaspy.inputs.data.dataset import Dataset
from leaspy.utils.realizations.realization import Realization


class VisualizationToolbox():

    def __init__(self):
        pass

    # Plot model directly
    def plot_mean(self, model, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        n_individuals = 1
        realizations = model.get_realization_object(n_individuals=n_individuals)

        for key, value in model.random_variable_informations().items():
            if value["type"] == "individual":
                realizations.reals_ind_variable_names.append(key)
                realizations.realizations[key] = Realization(key, value["shape"], value["type"])
                realizations.realizations[key].initialize(n_individuals, model, scale_individual=0.0)

        timepoints = np.linspace(model.parameters['tau_mean'] - 2 * np.sqrt(model.parameters['tau_std']),
                                 model.parameters['tau_mean'] + 2 * np.sqrt(model.parameters['tau_std']),
                                 100)
        timepoints = torch.Tensor(timepoints).reshape(1, -1, 1)

        xi = realizations['xi'].tensor_realizations
        tau = realizations['tau'].tensor_realizations
        sources = realizations['sources'].tensor_realizations
        patient_values = model.compute_individual_tensorized(timepoints, (xi, tau, sources))

        model_value = patient_values
        ax.plot(timepoints[0, :, :].detach().numpy(), model_value[0, :, :].detach().numpy(), c='black', alpha=0.4,
                linewidth=5)

        return 0

    def plot_patients(self, model, data, indices, ax=None):

        # Get dataset from data
        dataset = Dataset(data=data, model=model, algo=None)

        # Instanciate realizations
        realizations = data.realizations

        colors = cm.rainbow(np.linspace(0, 1, len(indices) + 2))

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        xi = realizations['xi'].tensor_realizations
        tau = realizations['tau'].tensor_realizations
        sources = realizations['sources'].tensor_realizations
        patient_values = model.compute_individual_tensorized(dataset.timepoints, (xi, tau, sources))

        # TODO only the 10 first, change that to specified indices

        dict_correspondence = {}
        for i, idx in enumerate(data.individuals.keys()):
            dict_correspondence[idx] = i

        for p, idx in enumerate(indices):
            i = dict_correspondence[idx]
            model_value = patient_values[i, 0:dataset.nb_observations_per_individuals[i], :]
            score = dataset.values[i, 0:dataset.nb_observations_per_individuals[i], :]
            ax.plot(dataset.timepoints[i, 0:dataset.nb_observations_per_individuals[i]].detach().numpy(),
                    model_value.detach().numpy(), c=colors[p])
            ax.plot(dataset.timepoints[i, 0:dataset.nb_observations_per_individuals[i]].detach().numpy(),
                    score.detach().numpy(), c=colors[p], linestyle='--',
                    marker='o')

        # Plot average model
        # tensor_timepoints = torch.Tensor(np.linspace(data.time_min, data.time_max, 40).reshape(-1,1))
        # model_average = model.compute_average(tensor_timepoints)
        # ax.plot(tensor_timepoints.detach().numpy(), model_average.detach().numpy(), c='black', linewidth=4, alpha=0.3)

        return 0

    # Plot distributions
    def plot_distributions_individual_parameter(self, data, real_ind_name, covariable=None):
        raise NotImplementedError
