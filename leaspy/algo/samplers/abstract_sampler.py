import numpy as np
import torch


class AbstractSampler:

    def __init__(self, info, n_patients):
        self.name = info["name"]
        self.temp_length = 25  # For now the same between pop and ind #TODO this is an hyperparameter
        self.shape = info["shape"]
        if info["type"] == "population":
            self.type = 'pop'
            # Initialize the acceptation history
            if len(self.shape) < 2:
                self.acceptation_temp = torch.zeros(size=self.shape).repeat(self.temp_length,
                                                                            1)  # convention : shape of pop is 2D
            elif len(self.shape) == 2:
                self.acceptation_temp = torch.zeros(size=self.shape).repeat(self.temp_length, 1, 1)
            else:
                raise ValueError("Dimension of population variable >2")
        elif info["type"] == "individual":
            self.type = 'ind'
            # Initialize the acceptation history
            self.acceptation_temp = torch.zeros(size=(n_patients,)).repeat(self.temp_length, 1)

    def _group_metropolis_step(self, alpha):
        accepted = (1. * (torch.rand(alpha.size(0)) < alpha)).float()
        return accepted.detach().clone()

    def _metropolis_step(self, alpha):
        """
        If better (alpha>=1) : accept
        If worse (alpha<1) : accept with probability alpha
        :param alpha:
        :return:
        """
        accepted = 0
        if alpha >= 1:
            # Case 1: we improved the LogL
            accepted = 1
        else:
            # Case 2: we decreased the LogL
            # Sample a realization from uniform law
            realization = np.random.uniform(low=0, high=1)
            # Choose to keep a lesser parameter value from it
            if realization < alpha:
                accepted = 1
        return accepted

    def _update_acceptation_rate(self, accepted):
        """
        Update acceptation rate from history of boolean accepted values for each dimension of each variable (except sources).
        :param accepted:
        :return:
        """


        # Ad the new acceptation result
        if self.type == "pop":
            self.acceptation_temp = torch.cat(
                [self.acceptation_temp, torch.tensor(accepted, dtype=torch.float32).reshape(self.shape).unsqueeze(0)])
        elif self.type == "ind":
            self.acceptation_temp = torch.cat(
                [self.acceptation_temp, torch.tensor(accepted, dtype=torch.float32).unsqueeze(0)])
        else:
            raise ValueError("Nor pop or ind")

        self.acceptation_temp = self.acceptation_temp[1:]
