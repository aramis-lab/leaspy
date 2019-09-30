import numpy as np
import pandas as pd
# import scipy
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler

from leaspy.algo.abstract_algo import AbstractAlgo
from leaspy.inputs.data.data import Data
from leaspy.inputs.data.result import Result


class SimulationAlgorithm(AbstractAlgo):
    """
    SimulationAlgorithm object class.
    This algorithm simulate new data given existing one by learning the individual parameters joined distribution
    """

    def __init__(self, settings):
        """
        Process initializer function that is called by Leaspy.simulate
        :param settings: an leaspy.inputs.algorithm_settings class object
        """

        super().__init__()

        self.number_of_subjects = settings.parameters['number_of_subjects']
        self.bandwidth_method = settings.parameters['bandwidth_method']
        self.noise = settings.parameters['noise']
        self.mean_number_of_visits = settings.parameters['mean_number_of_visits']
        self.std_number_of_visits = settings.parameters['std_number_of_visits']
        self.cofactor = settings.parameters['cofactor']
        self.cofactor_state = settings.parameters['cofactor_state']

    def _initialize_kernel(self, results=None):
        return 0

    def _sample_sources(self, xi, tau, bl, source_dimension):
        """
        Simulate individual sources given log-acceleration xi, time-shift tau, baseline time bl & sources dimension

        :param xi: float - log-acceleration
        :param tau: float - time-shift
        :param bl: float - baseline tyme
        :param source_dimension: int - sources dimension
        :return: numpy.array - sources
        """
        ind_1 = [0, 1, 2]
        ind_2 = list(range(3, source_dimension + 3))
        x_1 = np.vstack((xi, tau, bl))

        mu_1 = self.df_mean[ind_1][:, np.newaxis]
        mu_2 = self.df_mean[ind_2][:, np.newaxis]
        sigma_11 = self.df_cov[np.ix_(ind_1, ind_1)]
        sigma_22 = self.df_cov[np.ix_(ind_2, ind_2)]
        sigma_12 = self.df_cov[np.ix_(ind_1, ind_2)]

        mean_cond = (mu_2 + np.dot(np.transpose(sigma_12), np.dot(np.linalg.inv(sigma_11), x_1 - mu_1))).ravel()
        cov_cond = sigma_22 - np.dot(np.dot(np.transpose(sigma_12), np.linalg.inv(sigma_11)), sigma_12)

        return np.random.multivariate_normal(mean_cond, cov_cond)

    def _get_number_of_visits(self):  # ,xi, tau, bl, sources):
        """
        Simulate number of visits for a new simulated patient

        :return: int - number of visits
        """
        # Generate a number of visit around the mean_number_of_visits
        number_of_visits = int(self.mean_number_of_visits)
        if self.mean_number_of_visits != 0:
            number_of_visits += int(np.random.normal(0, self.std_number_of_visits))
        return number_of_visits

    def _get_xi_tau_sources_bl(self, results):
        """
        Get individual parameters

        :param results: leaspy result class object
        :return: xi list[float], tau list[float], sources numpy.ndarray - shape = Number_of_sources x Number_of_subjects,
        bl list[float]
        """
        xi = results.get_parameter_distribution('xi', self.cofactor)
        tau = results.get_parameter_distribution('tau', self.cofactor)
        sources = results.get_parameter_distribution('sources', self.cofactor)
        bl = []
        for idx in results.data.individuals.keys():
            ages = results.data.get_by_idx(idx).timepoints
            bl.append(min(ages))

        if self.cofactor is not None:
            # transform {'state1': xi_list1, 'state2': xi_list2, ...} to xi_list
            xi = xi[self.cofactor_state]
            tau = tau[self.cofactor_state]
            sources = sources[self.cofactor_state]
            bl = [bl[i] for i, state in enumerate(results.get_cofactor_distribution(self.cofactor))
                  if state == self.cofactor_state]

        sources = np.array([sources[key] for key in sources.keys()])
        return xi, tau, sources, bl

    def run(self, model, results):
        """
        Run simulation - learn joined distribution of patients' individual parameters and return a results object
        containing the simulated individual data and parameters.

        :param model: leaspy model class object
        :param results: leaspy result class object
        :return: leaspy result object - contain the simulated individual data and parameters
        """
        # Get individual parameters - for joined density estimation
        xi, tau, sources, bl = self._get_xi_tau_sources_bl(results)

        # Get joined density estimation (sources are not learn in this fashion)
        distribution = np.array([xi, tau, bl]).T
        ss = StandardScaler()
        rescaled_distribution = ss.fit_transform(distribution)
        kernel = stats.gaussian_kde(rescaled_distribution.T, bw_method=self.bandwidth_method)

        # Get mean by variable & covariance matrix
        df = np.concatenate((np.array([xi, tau, bl]), sources), axis=0).T
        df = pd.DataFrame(data=df)
        self.df_mean = df.mean().values
        self.df_cov = df.cov().values

        # Generate individual parameters (except sources)
        samples = np.transpose(kernel.resample(self.number_of_subjects))
        samples = ss.inverse_transform(samples)

        # Initialize simulated scores
        indices, timepoints, values = [], [], []
        # Simulated parameters
        xi, tau, bl = samples.T
        sources = []

        # Generate individual sources, scores, indices & time-points
        for i in range(self.number_of_subjects):
            # Generate sources
            sources.append(self._sample_sources(xi[i], tau[i], bl[i], model.source_dimension).tolist())
            # Generate time-points
            number_of_visits = self._get_number_of_visits()  # xi[i], tau[i], bl[i] - 1, sources[-1])
            if number_of_visits == 1:
                ages = [bl[i]]
            elif number_of_visits == 2:
                ages = [bl[i], bl[i] + 0.5]
            else:
                ages = [bl[i], bl[i] + 0.5] + [bl[i] + i for i in range(1, number_of_visits - 1)]
            timepoints.append(ages)
            # Generate scores
            indiv_param = {'xi': torch.tensor([xi[i]], dtype=torch.float32).unsqueeze(0),
                           'tau': torch.tensor([tau[i]], dtype=torch.float32).unsqueeze(0),
                           'sources': torch.tensor(sources[-1], dtype=torch.float32).unsqueeze(0)}
            observations = model.compute_individual_tensorized(torch.tensor(ages, dtype=torch.float32).unsqueeze(0), indiv_param)
            # Add the desired noise
            if self.noise:
                noise = torch.distributions.Normal(loc=0, scale=model.parameters['noise_std']).sample(
                    observations.shape)
                observations += noise
                observations = observations.clamp(0, 1)
            values.append(observations.squeeze(0).detach().numpy().tolist())
            # Generate indices
            indices.append(i)

        # Return the leaspy.inputs.data.results object
        simulated_parameters = {'xi': torch.tensor(xi, dtype=torch.float32).view(-1, 1),
                                'tau': torch.tensor(tau, dtype=torch.float32).view(-1, 1),
                                'sources': torch.tensor(sources, dtype=torch.float32)}
        simulated_scores = Data.from_individuals(indices, timepoints, values, results.data.headers)
        return Result(data=simulated_scores,
                      individual_parameters=simulated_parameters,
                      noise_std=self.noise)

        # TODO : Check with Raphaël if he needs the following
        '''

        param = []
        for a in param_ind:
            for i in range(a.shape[1]):
                param.append(a[:, i].detach().numpy())
        param = np.array(param)

        kernel = scipy.stats.gaussian_kde(param)

        # Get metrics from Data
        n_points = np.mean(dataset.nb_observations_per_individuals)
        data_sim = pd.DataFrame(columns=['ID', 'TIME'] + dataset.headers)
        indiv_param = {}

        noise = torch.distribution.normal.Normal(0, noise_scale * self.model.parameters['noise_std'])
        t0 = model.parameters['tau_mean'].detach().numpy()
        v0 = model.parameters['xi_mean'].detach().numpy()
        for idx in range(N_indiv):
            this_indiv_param = {}
            this_indiv_param[idx] = {}
            sim = kernel.resample(1)[:, 0]
            this_indiv_param[idx]['xi'] = sim[0]
            this_indiv_param[idx]['tau'] = sim[1]
            if model.name != "univariate":
                this_indiv_param[idx]['sources'] = sim[2:]
            indiv_param.update(this_indiv_param)
            age_diag = (ref_age - t0) * np.exp(v0) / np.exp(sim[0]) + sim[1]
            # Draw the number of visits
            n_visits = np.random.randint(max(2, n_points - 3), dataset.max_observations)
            timepoints = np.linspace(age_diag - n_visits * self.interval / 2, age_diag + n_visits * self.interval / 2, n_visits)
            timepoints = torch.tensor(timepoints, dtype=torch.float32).unsqueeze(0)

            values = model.compute_individual_tensorized(timepoints,
                                                              self.model.param_ind_from_dict(this_indiv_param))
            values = values + noise_scale * noise.sample(values.shape)
            values = torch.clamp(values, 0, 1).squeeze(0)

            ten_idx = torch.tensor([idx] * (n_visits), dtype=torch.float32)
            val = torch.cat([ten_idx.unsqueeze(1), timepoints.squeeze(0).unsqueeze(1), values], dim=1)
            truc = pd.DataFrame(val.detach().numpy(), columns=['ID', 'TIME'] + dataset.headers)
            data_sim = data_sim.append(truc)

        return data_sim

        '''
