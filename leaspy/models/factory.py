from leaspy.models import ALL_MODELS
from leaspy.models.base import BaseModel
from leaspy.exceptions import LeaspyModelInputError


class ModelFactory:
    """
    Return the wanted model given its name.
    """

    @staticmethod
    def model(name: str, **kwargs) -> BaseModel:
        """
        Return the model object corresponding to ``name`` arg with possible ``kwargs``.

        Check name type and value.

        Parameters
        ----------
        name : :obj:`str`
            The model's name.
        **kwargs
            Contains model's hyper-parameters.
            Raise an error if the keyword is inappropriate for the given model's name.

        Returns
        -------
        :class:`.BaseModel`
            A child class object of :class:`.models.BaseModel` class object determined by ``name``.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If an incorrect model is requested.

        See Also
        --------
        :class:`~leaspy.api.Leaspy`
        """
        if not isinstance(name, str):
            raise LeaspyModelInputError("The `name` argument must be a string!")

        name = name.lower()
        if name not in ALL_MODELS:
            raise LeaspyModelInputError(
                "The name of the model you are trying to create does not exist! "
                f"It should be in {{{repr(tuple(ALL_MODELS.keys()))[1:-1]}}}"
            )
        # instantiate model with optional keyword arguments
        return ALL_MODELS[name](name, **kwargs)
