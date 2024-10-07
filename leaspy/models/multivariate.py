import torch

from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.models.utils.attributes import AttributesFactory
from leaspy.io.data.dataset import Dataset
from leaspy.io.realizations import CollectionRealization

from leaspy.utils.docs import doc_with_super, doc_with_
from leaspy.utils.subtypes import suffixed_method
from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.typing import DictParamsTorch, DictParams

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either
# model values or model values + jacobians wrt individual parameters

# TODO refact? subclass or other proper code technique to extract model's concrete
#  formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class MultivariateModel(AbstractMultivariateModel):
    """
    Manifold model for multiple variables of interest (logistic or linear formulation).

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    **kwargs
        Hyperparameters of the model (including `noise_model`)

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        * If `name` is not one of allowed sub-type: 'univariate_linear' or 'univariate_logistic'
        * If hyperparameters are inconsistent
    """

    SUBTYPES_SUFFIXES = {
        'linear': '_linear',
        'logistic': '_logistic',
        'mixed_linear-logistic': '_mixed',
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

        self.parameters["v0"] = None
        self.MCMC_toolbox['priors']['v0_std'] = None  # Value, Coef

        self._subtype_suffix = self._check_subtype()

        # enforce a prior for v0_mean --> legacy / never used in practice
        self._set_v0_prior = False

    def _check_subtype(self) -> str:
        if self.name not in self.SUBTYPES_SUFFIXES.keys():
            raise LeaspyModelInputError(
                f"{self.__class__.__name__} name should be among these valid sub-types: "
                f"{list(self.SUBTYPES_SUFFIXES.keys())}."
            )

        return self.SUBTYPES_SUFFIXES[self.name]

    @suffixed_method
    def compute_individual_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        pass

    def compute_individual_tensorized_linear(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        # Population parameters
        #positions, velocities, mixing_matrix = self._get_attributes(attribute_type)
        positions, velocities, orthonormal_basis = self._get_attributes(attribute_type)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # for automatic broadcast on n_features (last dim)

        # Model expected value
        model = positions + velocities * reparametrized_time

        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            betas = self.parameters['betas']
            mixing_matrix = np.matmul(orthonormal_basis, betas)
            wi = sources.matmul(mixing_matrix.t())
            model += wi.unsqueeze(-2)

        return model  # (n_individuals, n_timepoints, n_features)

    def compute_individual_tensorized_logistic(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        # Population parameters
        #g, v0, a_matrix = self._get_attributes(attribute_type)
        g, v0, orthonormal_basis = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # (n_individuals, n_timepoints, n_features)

        if self.is_ordinal:
            # add an extra dimension for the levels of the ordinal item
            reparametrized_time = reparametrized_time.unsqueeze(-1)
            g = g.unsqueeze(-1)
            b = b.unsqueeze(-1)
            v0 = v0.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)  # (features, max_level)
            deltas = deltas.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            # infinite deltas (impossible ordinal levels) will induce model = 0 which is intended
            reparametrized_time = reparametrized_time - deltas.cumsum(dim=-1)

        LL = v0 * reparametrized_time

        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            betas = self.parameters['betas']
            a_matrix = np.matmul(orthonormal_basis, betas)
            wi = sources.matmul(a_matrix.t()).unsqueeze(-2)  # unsqueeze for (n_timepoints)
            if self.is_ordinal:
                wi = wi.unsqueeze(-1)
            LL += wi

        # TODO? more efficient & accurate to compute `torch.exp(-t*b + log_g)` since we
        #  directly sample & stored log_g
        LL = 1. + g * torch.exp(-LL * b)
        model = 1. / LL

        # For ordinal models, compute pdf instead of survival function if needed
        model = self.compute_appropriate_ordinal_model(model)

        return model  # (n_individuals, n_timepoints, n_features [, extra_dim_ordinal_models])

    @suffixed_method
    def compute_individual_ages_from_biomarker_values_tensorized(
        self,
        value: torch.Tensor,
        individual_parameters: dict,
        feature: str,
    ) -> torch.Tensor:
        pass

    def compute_individual_ages_from_biomarker_values_tensorized_logistic(
        self,
        value: torch.Tensor,
        individual_parameters: dict,
        feature: str,
    ) -> torch.Tensor:
        if value.dim() != 2:
            raise LeaspyModelInputError(f"The biomarker value should be dim 2, not {value.dim()}!")

        if self.is_ordinal:
            return self._compute_individual_ages_from_biomarker_values_tensorized_logistic_ordinal(
                value, individual_parameters, feature
            )
        # avoid division by zero:
        value = value.masked_fill((value == 0) | (value == 1), float('nan'))

        # 1/ get attributes
        #g, v0, a_matrix = self._get_attributes(None)
        g, v0, orthonormal_basis = self._get_attributes(None)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            betas = self.parameters['betas']
            a_matrix = np.matmul(orthonormal_basis, betas)
            wi = sources.matmul(a_matrix.t())
        else:
            wi = 0

        # get feature value for g, v0 and wi
        feat_ind = self.features.index(feature)  # all consistency checks were done in API layer
        g = torch.tensor([g[feat_ind]])  # g and v0 were shape: (n_features in the multivariate model)
        v0 = torch.tensor([v0[feat_ind]])
        if self.source_dimension != 0:
            wi = wi[0, feat_ind].item()  # wi was shape (1, n_features)

        # 2/ compute age
        ages = tau + (torch.exp(-xi) / v0) * ((g / (g + 1) ** 2) * torch.log(g/(1 / value - 1)) - wi)
        # assert ages.shape == value.shape

        return ages

    def _compute_individual_ages_from_biomarker_values_tensorized_logistic_ordinal(
        self,
        value: torch.Tensor,
        individual_parameters: dict,
        feature: str,
    ) -> torch.Tensor:
        """
        For one individual, compute age(s) breakpoints at which the given features
        levels are the most likely (given the subject's individual parameters).

        Consistency checks are done in the main :term:`API` layer.

        Parameters
        ----------
        value : :class:`torch.Tensor`
            Contains the :term:`biomarker` level value(s) of the subject.

        individual_parameters : :obj:`dict`
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like.

        feature : :obj:`str`
            Name of the considered :term:`biomarker`

            .. note::
                Optional for :class:`.UnivariateModel`, compulsory
                for :class:`.MultivariateModel`.

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s)
            Shape of tensor is ``(1, n_values)``.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If computation is tried on more than 1 individual.
        """
        # 1/ get attributes
        #g, v0, a_matrix = self._get_attributes(None)
        g, v0, orthonormal_basis = self._get_attributes(None)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            betas = self.parameters['betas']
            a_matrix = np.matmul(orthonormal_basis, betas)
            wi = sources.matmul(a_matrix.t())
        else:
            wi = 0

        # get feature value for g, v0 and wi
        feat_ind = self.features.index(feature)  # all consistency checks were done in API layer
        g = torch.tensor([g[feat_ind]])  # g and v0 were shape: (n_features in the multivariate model)
        v0 = torch.tensor([v0[feat_ind]])
        if self.source_dimension != 0:
            wi = wi[0, feat_ind].item()  # wi was shape (1, n_features)

        # 2/ compute age
        ages_0 = tau + (torch.exp(-xi) / v0) * ((g / (g + 1) ** 2) * torch.log(g) - wi)
        deltas_ft = self._get_deltas(None)[feat_ind]
        delta_max = deltas_ft[torch.isfinite(deltas_ft)].sum()
        ages_max = tau + (torch.exp(-xi) / v0) * ((g / (g + 1) ** 2) * torch.log(g) - wi + delta_max)

        grid_timepoints = torch.linspace(ages_0.item(), ages_max.item(), 1000)

        return self._ordinal_grid_search_value(
            grid_timepoints,
            value,
            individual_parameters=individual_parameters,
            feat_index=feat_ind,
        )

    @suffixed_method
    def compute_jacobian_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> DictParamsTorch:
        pass

    def compute_jacobian_tensorized_linear(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> DictParamsTorch:
        # Population parameters
        _, v0, mixing_matrix = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # (n_individuals, n_timepoints, n_features)

        alpha = torch.exp(xi).reshape(-1, 1, 1)
        dummy_to_broadcast_n_ind_n_tpts = torch.ones_like(reparametrized_time)

        # Jacobian of model expected value w.r.t. individual parameters
        derivatives = {
            'xi': (v0 * reparametrized_time).unsqueeze(-1),  # add a last dimension for len param
            'tau': (v0 * -alpha * dummy_to_broadcast_n_ind_n_tpts).unsqueeze(-1),  # same
        }

        if self.source_dimension > 0:
            derivatives['sources'] = (
                mixing_matrix.expand((1, 1, -1, -1)) * dummy_to_broadcast_n_ind_n_tpts.unsqueeze(-1)
            )

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts, n_dims_param)]
        return derivatives

    def compute_jacobian_tensorized_logistic(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> DictParamsTorch:
        # TODO: refact highly inefficient (many duplicated code from `compute_individual_tensorized_logistic`)

        # Population parameters
        g, v0, a_matrix = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # (n_individuals, n_timepoints, n_features)
        alpha = torch.exp(xi).reshape(-1, 1, 1)

        if self.is_ordinal:
            # add an extra dimension for the levels of the ordinal item
            LL = reparametrized_time.unsqueeze(-1)
            g = g.unsqueeze(-1)
            b = b.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)  # (features, max_level)
            deltas = deltas.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            LL = LL - deltas.cumsum(dim=-1)
            LL = v0.unsqueeze(-1) * LL

        else:
            LL = v0 * reparametrized_time

        if self.source_dimension != 0:
            #sources = individual_parameters['sources']
            #wi = sources.matmul(a_matrix.t()).unsqueeze(-2)  # unsqueeze for (n_timepoints)
            wi = individual_parameters['wi']
            if self.is_ordinal:
                wi = wi.unsqueeze(-1)
            LL += wi
        LL = 1. + g * torch.exp(-LL * b)
        model = 1. / LL

        # Jacobian of model expected value w.r.t. individual parameters
        c = model * (1. - model) * b

        derivatives = {
            'xi': (v0 * reparametrized_time).unsqueeze(-1),
            'tau': (-v0 * alpha).unsqueeze(-1),
        }
        if self.source_dimension > 0:
            derivatives['sources'] = a_matrix.expand((1, 1, -1, -1))

        if self.is_ordinal:
            ordinal_lvls_shape = c.shape[-1]
            for param in derivatives:
                derivatives[param] = derivatives[param].unsqueeze(-2).repeat(1, 1, 1, ordinal_lvls_shape, 1)

        # Multiply by common constant, and post-process derivative for ordinal models if needed
        for param in derivatives:
            derivatives[param] = self.compute_appropriate_ordinal_model(
                c.unsqueeze(-1) * derivatives[param]
            )

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts [, extra_dim_ordinal_models], n_dims_param)]
        return derivatives

    ##############################
    ### MCMC-related functions ###
    ##############################

    def initialize_MCMC_toolbox(self) -> None:
        """
        Initialize the model's :term:`MCMC` toolbox attribute.
        """
        self.MCMC_toolbox = {
            'priors': {'g_std': 0.01, 'v0_std': 0.01, 'betas_std': 0.01},  # population parameters
        }
        # Initialize a prior for v0_mean (legacy code / never used in practice)
        if self._set_v0_prior:
            self.MCMC_toolbox['priors']['v0_mean'] = self.parameters['v0'].clone().detach()
            self.MCMC_toolbox['priors']['s_v0'] = 0.1
            # TODO? same on g?

        # specific priors for ordinal models
        self._initialize_MCMC_toolbox_ordinal_priors()

        self.MCMC_toolbox['attributes'] = AttributesFactory.attributes(
            self.name, self.dimension, self.source_dimension, **self._attributes_factory_ordinal_kws
        )
        # TODO? why not passing the ready-to-use collection realizations that is initialized
        #  at beginning of fit algo and use it here instead?
        self.update_MCMC_toolbox({"all"}, self._get_population_realizations())

    def update_MCMC_toolbox(self, vars_to_update: set, realizations: CollectionRealization) -> None:
        """
        Update the model's :term:`MCMC` toolbox attribute with the provided ``vars_to_update``.

        Parameters
        ----------
        vars_to_update : :obj:`set` of :obj:`str`
            The set of variable names to be updated.
        realizations : :class:`.CollectionRealization`
            The realizations to use for updating the :term:`MCMC` toolbox.
        """
        values = {}
        update_all = "all" in vars_to_update
        if update_all or "g" in vars_to_update:
            values["g"] = realizations["g"].tensor
        if update_all or len(vars_to_update.intersection({"v0", "v0_collinear"})):
            values["v0"] = realizations["v0"].tensor
        if self.source_dimension != 0 and (update_all or "betas" in vars_to_update):
            values["betas"] = realizations["betas"].tensor
        self._update_MCMC_toolbox_ordinal(vars_to_update, realizations, values)
        self.MCMC_toolbox['attributes'].update(vars_to_update, values)

    def _center_xi_realizations(self, realizations: CollectionRealization) -> None:
        """
        Center the ``xi`` realizations in place.

        .. note::
            This operation does not change the orthonormal basis
            (since the resulting ``v0`` is collinear to the previous one)
            Nor all model computations (only ``v0 * exp(xi_i)`` matters),
            it is only intended for model identifiability / ``xi_i`` regularization
            <!> all operations are performed in "log" space (``v0`` is log'ed)

        Parameters
        ----------
        realizations : :class:`.CollectionRealization`
            The realizations to use for updating the :term:`MCMC` toolbox.
        """
        mean_xi = torch.mean(realizations['xi'].tensor)
        realizations["xi"].tensor = realizations["xi"].tensor - mean_xi
        realizations["v0"].tensor = realizations["v0"].tensor + mean_xi
        self.update_MCMC_toolbox({'v0_collinear'}, realizations)

    def _center_wi_realizations(self, realizations: CollectionRealization) -> None:
        """
        Center the ``wi`` realizations in place.

        Parameters
        ----------
        realizations : :class:`.CollectionRealization`
            The realizations to use for updating the :term:`MCMC` toolbox.
        """
        mean_wi = torch.mean(realizations['wi'].tensor)
        realizations["wi"].tensor = realizations["wi"].tensor - mean_wi
        self.update_MCMC_toolbox(realizations)

    def compute_model_sufficient_statistics(
        self,
        data: Dataset,
        realizations: CollectionRealization,
    ) -> DictParamsTorch:
        """
        Compute the model's :term:`sufficient statistics`.

        Parameters
        ----------
        data : :class:`.Dataset`
            The input dataset.
        realizations : :class:`.CollectionRealization`
            The realizations from which to compute the model's :term:`sufficient statistics`.

        Returns
        -------
        DictParamsTorch :
            The computed :term:`sufficient statistics`.
        """
        # modify realizations in-place
        self._center_xi_realizations(realizations)
        self._center_wi_realizations(realizations)

        # unlink all sufficient statistics from updates in realizations!
        realizations = realizations.clone()
        sufficient_statistics = realizations[["g", "v0", "tau", "xi"]].tensors_dict
        if self.source_dimension != 0:
            #sufficient_statistics['betas'] = realizations["betas"].tensor
            sufficient_statistics["wi"] = realizations["wi"].tensor
        for param in ("tau", "xi"):
            sufficient_statistics[f"{param}_sqrd"] = torch.pow(realizations[param].tensor, 2)

        sufficient_statistics.update(
            self.compute_ordinal_model_sufficient_statistics(realizations)
        )

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data: Dataset, sufficient_statistics: DictParamsTorch) -> None:
        """
        Update the model's parameters during the burn in phase.

        During the burn-in phase, we only need to store the following parameters (cf. !66 and #60)
            - ``noise_std``
            - *_mean/std for :term:`regularization` of individual variables
            - others population parameters for :term:`regularization` of population variables

        We don't need to update the model "attributes" (never used during burn-in!).

        Parameters
        ----------
        data : :class:`.Dataset`
            The input dataset.
        sufficient_statistics : DictParamsTorch
            The :term:`sufficient statistics` to use for parameter update.
        """
        # Memoryless part of the algorithm
        self.parameters['g'] = sufficient_statistics['g']

        v0_emp = sufficient_statistics['v0']
        if self.MCMC_toolbox['priors'].get('v0_mean', None) is not None:
            v0_mean = self.MCMC_toolbox['priors']['v0_mean']
            s_v0 = self.MCMC_toolbox['priors']['s_v0']
            sigma_v0 = self.MCMC_toolbox['priors']['v0_std']
            self.parameters['v0'] = (1 / (1 / (s_v0 ** 2) + 1 / (sigma_v0 ** 2))) * (
                        v0_emp / (sigma_v0 ** 2) + v0_mean / (s_v0 ** 2))
        else:
            # new default
            self.parameters['v0'] = v0_emp

        if self.source_dimension != 0:
            self.parameters['betas'] = sufficient_statistics['betas']
            self.parameters['wi_mean'] = torch.mean(sufficient_statistics['wi'])
            self.parameters['wi_std'] = torch.std(sufficient_statistics['wi'])

        self.parameters['xi_std'] = torch.std(sufficient_statistics['xi'])
        self.parameters['tau_mean'] = torch.mean(sufficient_statistics['tau'])
        self.parameters['tau_std'] = torch.std(sufficient_statistics['tau'])

        self.parameters.update(
            self.get_ordinal_parameters_updates_from_sufficient_statistics(
                sufficient_statistics
            )
        )

    def update_model_parameters_normal(self, data: Dataset, sufficient_statistics: DictParamsTorch) -> None:
        """
        Stochastic :term:`sufficient statistics` used to update the parameters of the model.

        .. note::
            TODOs:
                - factorize ``update_model_parameters_***`` methods?
                - add a true, configurable, validation for all parameters?
                  (e.g.: bounds on tau_var/std but also on tau_mean, ...)
                - check the SS, especially the issue with mean(xi) and v_k
                - Learn the mean of xi and v_k
                - Set the mean of xi to 0 and add it to the mean of V_k

        Parameters
        ----------
        data : :class:`.Dataset`
            The input dataset.
        sufficient_statistics : DictParamsTorch
            The :term:`sufficient statistics` to use for parameter update.
        """
        from .utilities import compute_std_from_variance

        for param in ("g", "v0"):
            self.parameters[param] = sufficient_statistics[param]

        if self.source_dimension != 0:
            self.parameters['betas'] = sufficient_statistics['betas']
            if param == 'wi':
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
                #self.parameters[f"{param}_mean"] = param_cur_mean #fix it to zeros for now

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
            if param != 'xi':
                # xi-mean is fixed to 0 by design
                self.parameters[f"{param}_mean"] = param_cur_mean

        self.parameters.update(
            self.get_ordinal_parameters_updates_from_sufficient_statistics(
                sufficient_statistics
            )
        )

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
            "shape": torch.Size([self.dimension]),
            "rv_type": "multigaussian"
        }
        v0_info = {
            "name": "v0",
            "shape": torch.Size([self.dimension]),
            "rv_type": "multigaussian"
        }
        betas_info = {
            "name": "betas",
            "shape": torch.Size([self.dimension - 1, self.source_dimension]),
            "rv_type": "multigaussian",
            "scale": .5  # cf. GibbsSampler
        }
        variables_info = {
            "g": g_info,
            "v0": v0_info,
        }
        if self.source_dimension != 0:
            variables_info['betas'] = betas_info

        variables_info.update(self.get_additional_ordinal_population_random_variable_information())
        self.update_ordinal_population_random_variable_information(variables_info)

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

        wi_info = {
            "name": "wi",
            "shape": torch.Size([self.dimension]),  # or torch.Size([nb_clusters])
            "type": "individual",
            "rv_type": "gaussian"
        }

        variables_info = {
            "tau": tau_info,
            "xi": xi_info,
        }
        if self.source_dimension != 0:
            variables_info['sources'] = sources_info
            variables_info['wi'] = wi_info

        return variables_info


# document some methods (we cannot decorate them at method creation since they are
# not yet decorated from `doc_with_super`)
doc_with_(MultivariateModel.compute_individual_tensorized_linear,
          MultivariateModel.compute_individual_tensorized,
          mapping={'the model': 'the model (linear)'})
doc_with_(MultivariateModel.compute_individual_tensorized_logistic,
          MultivariateModel.compute_individual_tensorized,
          mapping={'the model': 'the model (logistic)'})

# doc_with_(MultivariateModel.compute_individual_tensorized_mixed,
#          MultivariateModel.compute_individual_tensorized,
#          mapping={'the model': 'the model (mixed logistic-linear)'})

doc_with_(MultivariateModel.compute_jacobian_tensorized_linear,
          MultivariateModel.compute_jacobian_tensorized,
          mapping={'the model': 'the model (linear)'})
doc_with_(MultivariateModel.compute_jacobian_tensorized_logistic,
          MultivariateModel.compute_jacobian_tensorized,
          mapping={'the model': 'the model (logistic)'})
# doc_with_(MultivariateModel.compute_jacobian_tensorized_mixed,
#          MultivariateModel.compute_jacobian_tensorized,
#          mapping={'the model': 'the model (mixed logistic-linear)'})

# doc_with_(MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized_linear,
#          MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
#          mapping={'the model': 'the model (linear)'})
doc_with_(MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized_logistic,
          MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
          mapping={'the model': 'the model (logistic)'})
# doc_with_(MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized_mixed,
#          MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
#          mapping={'the model': 'the model (mixed logistic-linear)'})
