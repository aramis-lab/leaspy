import numpy as np

from leaspy.utils.typing import KwargsType

from .base import BaseModel

__all__ = ["StatelessModel"]


class StatelessModel(BaseModel):
    """Stateless model do not use an internal state to keep track of variables.

    Parameters are stored in an internal dictionary.

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._parameters: KwargsType = {}

    @property
    def parameters(self):
        return self._parameters

    def load_parameters(self, parameters: KwargsType) -> None:
        """Instantiate or update the model's parameters.

        Parameters
        ----------
        parameters : :obj:`dict`
            Contains the model's parameters.
        """
        # <!> shallow copy only
        self._parameters = parameters.copy()
        # convert lists
        for k, v in self._parameters.items():
            if isinstance(v, list):
                self._parameters[k] = np.array(v)
