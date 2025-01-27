import torch

from .abstract_sampler import AbstractSampler


class HMCSampler(AbstractSampler):
    """
    Hamiltonian Monte Carlo sampler class.

    Parameters
    ----------
    info : dict
        Information concerning the given variable to sample (defined in the leaspy model)
        Example::

            v0_infos = {
                "name": "v0",
                "shape": torch.Size([self.dimension]),
                "type": "population",
                "rv_type": "multigaussian"
            }

    n_patients : int
        Number of patients
    eps : float
        ... (noise level prior ? )

    Attributes
    ----------
    eps : float
        wanted level of noise (?)
    L : int
        ... (?)
    name : str
        Name of the wanted variable to be sampled
    std : float
        ... (?)
    type : str
        * ``'pop'`` for population variable
        * ``'ind'`` for individual variable
    """

    def __init__(self, info, n_patients, eps):
        super().__init__(info, n_patients)

        self.L = 10
        if self.name == "tau":
            self.eps = eps * 10
        elif self.type == "pop":
            if self.name == "betas":
                self.eps = eps * 0.01
                self.L = 15
            else:
                self.eps = eps * 0.1
        else:
            self.eps = eps
        self.std = 0.1

    def sample(self, data, model, realizations, temperature_inv):
        """
        Sample new realization for a given realization state, dataset, model and temperature

        Parameters
        ----------
        data : :class:`.Dataset`
            Dataset class object build with leaspy class object Data, model & algo
        model : :class:`.AbstractModel`
            Model used by the algorithm
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
            Contain the current state & informations of all the variables of interest
        temperature_inv : float > 0
            Inverse of the temperature used in tempered MCMC-SAEM
        """

        if self.type == "ind":
            if self.name == "sources":
                self._sample_individual_realizations(data, model, realizations, 1.0)
            else:
                self._sample_individual_realizations(
                    data, model, realizations, temperature_inv
                )
        else:
            self._sample_pop_realizations(data, model, realizations, 1.0)

    def _proposal(self, p, realizations, model, data, temperature_inv):
        """
        Compute a proposition for the new state of the given variable

        Parameters
        ----------
        p : :class:`torch.Tensor`
            Current momenta
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
            Contain the current state & informations of all the variables of interest
        model : :class:`.AbstractModel`
            Model used by the algorithm
        data : :class:`~.Dataset`
            Dataset class object build with leaspy class object Data, model & algo
        temperature_inv: float > 0
            Inverse of the temperature used in tempered MCMC-SAEM

        Returns
        -------
        :class:`torch.Tensor`
        """

        for l in range(self.L + torch.randint(low=-5, high=5, size=(1,)).item()):
            self._leapfrog_step(p, realizations, model, data, temperature_inv)
        return realizations[self.name].tensor_realizations

    def _sample_pop_realizations(self, data, model, realizations, temperature_inv):
        """

        Parameters
        ----------
        data : :class:`~.Dataset`
            Dataset class object build with leaspy class object Data, model & algo
        model : :class:`.AbstractModel`
            Model used by the algorithm
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
            Contain the current state & informations of all the variables of interest
        temperature_inv : float > 0
            Inverse of the temperature used in tempered MCMC-SAEM
        """

        # this returns a tensor with values for each indiv
        realizations[self.name].set_autograd()
        old_real = realizations[self.name].tensor_realizations.clone()
        p = self._initialize_momentum(old_real)

        old_H = self._compute_pop_hamiltonian(
            model, data, p, realizations, temperature_inv
        )

        realizations[self.name].tensor_realizations = self._proposal(
            p, realizations, model, data, temperature_inv
        )
        model.update_MCMC_toolbox([self.name], realizations)
        new_H = self._compute_pop_hamiltonian(
            model, data, p, realizations, temperature_inv
        )

        accepted = self._metropolis_step(torch.exp(old_H - new_H))
        self._update_acceptation_rate([accepted])
        with torch.no_grad():
            realizations[self.name].tensor_realizations = (
                realizations[self.name].tensor_realizations * accepted
                + (1.0 - accepted) * old_real
            )
        realizations[self.name].unset_autograd()
        model.update_MCMC_toolbox([self.name], realizations)

    def _sample_individual_realizations(
        self, data, model, realizations, temperature_inv
    ):
        """
        ...

        Parameters
        ----------
        data
        model
        realizations
        temperature_inv
        """

        # this returns a tensor with values for each indiv
        realizations[self.name].set_autograd()
        old_real = realizations[self.name].tensor_realizations.clone()
        p = self._initialize_momentum(old_real)
        old_H = self._compute_ind_hamiltonian(
            model, data, p, realizations, temperature_inv
        )
        realizations[self.name].tensor_realizations = self._proposal(
            p, realizations, model, data, temperature_inv
        )
        new_H = self._compute_ind_hamiltonian(
            model, data, p, realizations, temperature_inv
        )
        accepted = self._group_metropolis_step(torch.exp(old_H - new_H))
        self._update_acceptation_rate(accepted.detach())
        accepted = accepted.unsqueeze(1)
        with torch.no_grad():
            realizations[self.name].tensor_realizations = (
                realizations[self.name].tensor_realizations * accepted
                + (1.0 - accepted) * old_real
            )
        realizations[self.name].unset_autograd()

    def _compute_U(self, realizations, data, model, temperature_inv):
        """
        ...

        Parameters
        ----------
        realizations
        data
        model
        temperature_inv

        Returns
        -------
        ...
        """

        U = torch.sum(
            model.compute_individual_attachment_tensorized_mcmc(data, realizations)
        )
        U += (
            torch.sum(model.compute_regularity_realization(realizations[self.name]))
            * temperature_inv
        )
        return U

    def _update_p(self, p, realizations):
        """
        ...

        Parameters
        ----------
        p
        realizations

        Returns
        -------
        bool
            Always True
        """
        a = realizations[self.name].tensor_realizations.grad
        if (a != a).byte().any():
            # realizations[self.name].tensor_realizations.grad.zero_()
            # return False
            a = torch.zeros(realizations[self.name].tensor_realizations.shape)
        p = p - self.eps / 2 * a
        realizations[self.name].tensor_realizations.grad.zero_()
        # print(self.name,'gardient is ok')
        return True

    def _leapfrog_step(self, p, realizations, model, data, temperature_inv):
        """
        ...

        Parameters
        ----------
        p
        realizations
        model
        data
        temperature_inv
        """
        U = self._compute_U(realizations, data, model, temperature_inv)
        U.backward()
        if not self._update_p(p, realizations):
            self.eps = 0.1 * self.eps
            print(temperature_inv)
            print(self.eps)
            a = realizations[self.name].tensor_realizations.grad
            print((a != a).byte().any())
            # print(torch.mean(torch.tensor(self.acceptation_temp)).item())
            # print(np.mean(self.acceptation_temp))
            print(self.name, " nan in grad")
            return
        with torch.no_grad():
            realizations[self.name].tensor_realizations.data = (
                realizations[self.name].tensor_realizations.data + self.eps * p
            )
            realizations[self.name].tensor_realizations.grad.zero_()
        U = self._compute_U(realizations, data, model, temperature_inv)
        U.backward()
        self._update_p(p, realizations)

        return

    def _initialize_momentum(self, old_real):
        """
        Initialization of the momenta given ...

        Parameters
        ----------
        old_real : ...
            ...

        Returns
        -------
        :class:`torch.Tensor`
        """

        p = torch.randn(old_real.shape)
        return p

    def _compute_ind_hamiltonian(self, model, data, p, realizations, temperature_inv):
        """
        ...

        Parameters
        ----------
        model
        data
        p
        realizations
        temperature_inv

        Returns
        -------
        :class:`torch.Tensor`
        """

        H = model.compute_individual_attachment_tensorized_mcmc(data, realizations)
        reg = model.compute_regularity_realization(realizations[self.name])
        H += torch.sum(reg.reshape(reg.shape[0], -1), dim=1) * temperature_inv
        momentum = p**2
        H += 0.5 * torch.sum(momentum.reshape(momentum.shape[0], -1), dim=1)
        return H

    def _compute_pop_hamiltonian(self, model, data, p, realizations, temperature_inv):
        """
        ...

        Parameters
        ----------
        model
        data
        p
        realizations
        temperature_inv

        Returns
        -------
        :class:`torch.Tensor`
        """

        H = torch.sum(
            model.compute_individual_attachment_tensorized_mcmc(data, realizations)
        )
        reg = model.compute_regularity_realization(realizations[self.name])
        H += torch.sum(reg) * temperature_inv
        momentum = p**2
        H += 0.5 * torch.sum(momentum)
        return H
