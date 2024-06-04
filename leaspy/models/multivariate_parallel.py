import torch

from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.models.utils.attributes.logistic_parallel_attributes import LogisticParallelAttributes
from leaspy.utils.typing import DictParamsTorch, DictParams
from leaspy.utils.docs import doc_with_super
from leaspy.io.data.dataset import Dataset
from leaspy.io.realizations import CollectionRealization


@doc_with_super()
class MultivariateParallelModel(AbstractMultivariateModel):
    """
    Logistic model for multiple variables of interest, imposing same average
    evolution pace for all variables (logistic curves are only time-shifted).

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    **kwargs
        Hyperparameters of the model.
    """
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

        self.parameters["deltas"] = None
        self.MCMC_toolbox['priors']['deltas_std'] = None

    def compute_individual_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        # Population parameters
        g, deltas, mixing_matrix = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # (n_individuals, n_timepoints, -> n_features)

        # Model expected value
        t = reparametrized_time + deltas
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(mixing_matrix.t())  # (n_individuals, n_features)
            g_deltas_exp = g * torch.exp(-deltas)
            b = (1. + g_deltas_exp) ** 2 / g_deltas_exp
            t += (b * wi).unsqueeze(-2)  # to get n_timepoints dimension
        model = 1. / (1. + g * torch.exp(-t))

        return model

    def compute_jacobian_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> DictParamsTorch:
        # TODO: refact highly inefficient (many duplicated code from `compute_individual_tensorized`)

        # Population parameters
        g, deltas, mixing_matrix = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # (n_individuals, n_timepoints, -> n_features)
        alpha = torch.exp(xi).reshape(-1, 1, 1)

        # Model expected value
        t = reparametrized_time + deltas
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(mixing_matrix.t())  # (n_individuals, n_features)
            g_deltas_exp = g * torch.exp(-deltas)
            b = (1. + g_deltas_exp) ** 2 / g_deltas_exp  # (n_features, )
            t += (b * wi).unsqueeze(-2)  # to get n_timepoints dimension
        model = 1. / (1. + g * torch.exp(-t))

        # Jacobian of model expected value w.r.t. individual parameters
        c = model * (1. - model)

        derivatives = {
            'xi': (c * reparametrized_time).unsqueeze(-1),
            'tau': (c * -alpha).unsqueeze(-1),
        }
        if self.source_dimension > 0:
            b = b.reshape((1, 1, -1, 1))  # n_features is third
            derivatives['sources'] = c.unsqueeze(-1) * b * mixing_matrix.expand((1, 1, -1, -1))

        return derivatives

    def compute_individual_ages_from_biomarker_values_tensorized(
        self,
        value: torch.Tensor,
        individual_parameters: dict,
        feature: str,
    ) -> torch.Tensor:
        raise NotImplementedError("Open an issue on Gitlab if needed.")  # pragma: no cover

    ##############################
    ### MCMC-related functions ###
    ##############################

    def initialize_MCMC_toolbox(self) -> None:
        self.MCMC_toolbox = {
            'priors': {
                'g_std': 0.01,
                'deltas_std': 0.01,
                'betas_std': 0.01,
            },  # population parameters
            'attributes': LogisticParallelAttributes(
                self.name, self.dimension, self.source_dimension
            ),
        }
        self.update_MCMC_toolbox({"all"}, self._get_population_realizations())

    def update_MCMC_toolbox(self, vars_to_update: set, realizations: CollectionRealization) -> None:
        values = {}
        update_all = "all" in vars_to_update
        if update_all or "g" in vars_to_update:
            values['g'] = realizations["g"].tensor
        if update_all or "deltas" in vars_to_update:
            values['deltas'] = realizations["deltas"].tensor
        if self.source_dimension != 0 and (update_all or "betas" in vars_to_update):
            values['betas'] = realizations["betas"].tensor

        self.MCMC_toolbox['attributes'].update(vars_to_update, values)

    def compute_model_sufficient_statistics(
        self,
        data: Dataset,
        realizations: CollectionRealization,
    ) -> DictParamsTorch:
        # unlink all sufficient statistics from updates in realizations!
        realizations = realizations.clone()

        sufficient_statistics = realizations[["g", "deltas", "tau", "xi"]].tensors_dict
        if self.source_dimension != 0:
            sufficient_statistics['betas'] = realizations["betas"].tensor
        for param in ("tau", "xi"):
            sufficient_statistics[f"{param}_sqrd"] = torch.pow(realizations[param].tensor, 2)

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data: Dataset, sufficient_statistics: DictParamsTorch) -> None:
        for param in ("g", "deltas"):
            self.parameters[param] = sufficient_statistics[param]
        if self.source_dimension != 0:
            self.parameters['betas'] = sufficient_statistics['betas']
        for param in ("xi", "tau"):
            param_realizations = sufficient_statistics[param]
            self.parameters[f"{param}_mean"] = torch.mean(param_realizations)
            self.parameters[f"{param}_std"] = torch.std(param_realizations)

    def update_model_parameters_normal(self, data: Dataset, sufficient_statistics: DictParamsTorch) -> None:
        # TODO? factorize `update_model_parameters_***` methods?
        from .utilities import compute_std_from_variance

        for param in ("g", "deltas"):
            self.parameters[param] = sufficient_statistics[param]
        if self.source_dimension != 0:
            self.parameters['betas'] = sufficient_statistics['betas']

        for param in ("tau", "xi"):
            param_old_mean = self.parameters[f"{param}_mean"]
            param_cur_mean = torch.mean(sufficient_statistics[param])
            param_variance_update = (
                torch.mean(sufficient_statistics[f"{param}_sqrd"]) -
                2. * param_old_mean * param_cur_mean
            )
            param_variance = param_variance_update + param_old_mean ** 2
            self.parameters[f"{param}_std"] = compute_std_from_variance(
                param_variance, varname=f"{param}_std"
            )
            self.parameters[f"{param}_mean"] = param_cur_mean

    def get_population_random_variable_information(self) -> DictParams:
        """
        Return the information on population random variables relative to the model.

        Returns
        -------
        DictParams :
            The information on the population random variables.
        """
        g_info = {
            "name": "g",
            "shape": torch.Size([1]),
            "rv_type": "multigaussian"
        }
        deltas_info = {
            "name": "deltas",
            "shape": torch.Size([self.dimension - 1]),
            "rv_type": "multigaussian",
            "scale": 1.  # cf. GibbsSampler
        }
        betas_info = {
            "name": "betas",
            "shape": torch.Size([self.dimension - 1, self.source_dimension]),
            "rv_type": "multigaussian",
            "scale": .5  # cf. GibbsSampler
        }
        variables_info = {
            "g": g_info,
            "deltas": deltas_info,
        }
        if self.source_dimension != 0:
            variables_info['betas'] = betas_info

        return variables_info

    def get_individual_random_variable_information(self) -> DictParams:
        """
        Return the information on individual random variables relative to the model.

        Returns
        -------
        DictParams :
            The information on the individual random variables.
        """
        tau_info = {
            "name": "tau",
            "shape": torch.Size([1]),
            "rv_type": "gaussian"
        }
        xi_info = {
            "name": "xi",
            "shape": torch.Size([1]),
            "rv_type": "gaussian"
        }
        sources_info = {
            "name": "sources",
            "shape": torch.Size([self.source_dimension]),
            "rv_type": "gaussian"
        }
        variables_info = {
            "tau": tau_info,
            "xi": xi_info,
        }
        if self.source_dimension != 0:
            variables_info['sources'] = sources_info

        return variables_info
