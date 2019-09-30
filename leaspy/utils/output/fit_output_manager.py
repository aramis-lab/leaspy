import numpy as np
import os
import csv
import time
import torch
from leaspy.utils.output.visualization.plotter import Plotter


class FitOutputManager():

    # TODO: add a loading bar for a run

    def __init__(self, outputs):
        self.print_periodicity = outputs.console_print_periodicity
        self.plot_periodicity = outputs.plot_periodicity
        self.save_periodicity = outputs.save_periodicity
        self.path_save_model_parameters_convergence = outputs.parameter_convergence_path
        self.path_output = outputs.root_path
        self.path_plot = outputs.plot_path
        self.path_plot_patients = outputs.patients_plot_path
        self.path_plot_convergence_model_parameters_1 = os.path.join(outputs.plot_path, "convergence_1.pdf")
        self.path_plot_convergence_model_parameters_2 = os.path.join(outputs.plot_path, "convergence_2.pdf")

        # Options
        # TODO : Maybe add to the outputs reader
        self.plot_options = {}
        self.plot_options['maximum_patient_number'] = 5
        self.plotter = Plotter()

        self.time = time.time()

    def iteration(self, algo, data, model, realizations):
        iteration = algo.current_iteration

        if self.print_periodicity is not None:
            if iteration % self.print_periodicity == 0:
                self.print_algo_statistics(algo)
                self.print_model_statistics(model)
                self.print_time()

        if self.path_output is None:
            return

        if self.save_periodicity is not None:
            if iteration % self.save_periodicity == 0:
                self.save_model_parameters_convergence(iteration, model)
                # self.save_model(model)

        if self.plot_periodicity is not None:
            if iteration % self.plot_periodicity == 0:
                self.plot_patient_reconstructions(iteration, data, model, realizations)
                self.plot_convergence_model_parameters(model)

        if (algo.algo_parameters['n_iter'] - iteration) < 100:
            self.save_realizations(iteration, realizations)

    ########
    ## Printing methods
    ########

    def print_time(self):
        current_time = time.time()
        print("Duration since last print : {0}s".format(np.round(current_time - self.time), decimals=4))
        self.time = current_time

    def print_model_statistics(self, model):
        print(model)

    def print_algo_statistics(self, algo):
        print(algo)

    ########
    ## Saving methods
    ########

    def save_model_parameters_convergence(self, iteration, model):
        model_parameters = model.parameters

        # TODO maybe better way ???
        model_parameters_save = model_parameters.copy()

        # TODO I Stopped here, 2d array saves should be fixed.

        # Transform the types
        for key, value in model_parameters.items():
            if type(value) in [torch.Tensor]:
                value = value.numpy()
                model_parameters_save[key] = value
            if type(value) in [float]:
                model_parameters_save[key] = [value]
            elif type(value) in [list]:
                model_parameters_save[key] = np.array(value)
            elif value.shape == ():
                model_parameters_save[key] = [float(value)]
            # TODO, apriori only for beta
            elif type(value) in [np.ndarray]:
                # Beta
                # TODO do something intelligent here
                # if value.shape[0] > 1:
                if key == "betas":
                    model_parameters_save.pop(key)
                    for column in range(value.shape[1]):
                        model_parameters_save["{0}_{1}".format(key, column)] = value[:, column]
                # P0, V0
                elif value.shape[0] == 1 and len(value.shape) > 1:
                    model_parameters_save[key] = value[0]

        # Save the dictionnary
        for key, value in model_parameters_save.items():
            path = os.path.join(self.path_save_model_parameters_convergence, key + ".csv")
            with open(path, 'a', newline='') as filename:
                writer = csv.writer(filename)
                # writer.writerow([iteration]+list(model_parameters.values()))
                writer.writerow([iteration] + list(value))

    def save_realizations(self, iteration, realizations):
        for name in ['xi', 'tau']:
            value = realizations[name].tensor_realizations.squeeze(1).detach().numpy()
            path = os.path.join(self.path_save_model_parameters_convergence, name + ".csv")
            with open(path, 'a', newline='') as filename:
                writer = csv.writer(filename)
                # writer.writerow([iteration]+list(model_parameters.values()))
                writer.writerow([iteration] + list(value))
        if "sources" in realizations.reals_ind_variable_names:
            for i in range(realizations['sources'].tensor_realizations.shape[1]):
                value = realizations['sources'].tensor_realizations[:, i].detach().numpy()
                path = os.path.join(self.path_save_model_parameters_convergence, 'sources' + str(i) + ".csv")
                with open(path, 'a', newline='') as filename:
                    writer = csv.writer(filename)
                    # writer.writerow([iteration]+list(model_parameters.values()))
                    writer.writerow([iteration] + list(value))

    ########
    ## Plotting methods
    ########

    def plot_convergence_model_parameters(self, model):
        self.plotter.plot_convergence_model_parameters(self.path_save_model_parameters_convergence,
                                                       self.path_plot_convergence_model_parameters_1,
                                                       self.path_plot_convergence_model_parameters_2,
                                                       model)

    def plot_model_average_trajectory(self, model):
        raise NotImplementedError

    def plot_patient_reconstructions(self, iteration, data, model, realizations):
        path_iteration = os.path.join(self.path_plot_patients, 'plot_patients_{0}.pdf'.format(iteration))
        param_ind = model.get_param_from_real(realizations)
        self.plotter.plot_patient_reconstructions(path_iteration, data, model, param_ind,
                                                  self.plot_options['maximum_patient_number'])

        """
        colors = cm.rainbow(np.linspace(0, 1, self.plot_options['maximum_patient_number']+2))
        reals_pop, reals_ind = realizations

        fig, ax = plt.subplots(1, 1)

        for i, idx in enumerate(data.indices):
            model_value = model.compute_individual(data[idx], reals_pop, reals_ind[idx])
            score = data[idx].tensor_observations
            ax.plot(data[idx].tensor_timepoints.detach().numpy(), model_value.detach().numpy(), c=colors[i])
            ax.plot(data[idx].tensor_timepoints.detach().numpy(), score.detach().numpy(), c=colors[i], linestyle='--',
                    marker='o')

            if i > self.plot_options['maximum_patient_number']:
                break

        # Plot average model
        tensor_timepoints = torch.Tensor(np.linspace(data.time_min, data.time_max, 40).reshape(-1,1))
        model_average = model.compute_average(tensor_timepoints)
        ax.plot(tensor_timepoints.detach().numpy(), model_average.detach().numpy(), c='black', linewidth=4, alpha=0.3)

        plt.savefig(os.path.join(self.path_plot_patients,'plot_patients_{0}.pdf'.format(iteration)))
        plt.close()"""
