import json
from typing import Union

from leaspy import __version__
from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.typing import KwargsType

__all__ = ["ModelSettings"]


class ModelSettings:
    """
    Used in :meth:`.Leaspy.load` to create a :class:`.Leaspy` class object from a `json` file.

    Parameters
    ----------
    path_to_model_settings_or_dict : dict or str
        * If a str: path to a json file containing model settings
        * If a dict: content of model settings

    Raises
    ------
    :exc:`.LeaspyModelInputError`
    """

    def __init__(self, path_to_model_settings_or_dict: Union[str, dict]):
        if isinstance(path_to_model_settings_or_dict, dict):
            settings = path_to_model_settings_or_dict
        elif isinstance(path_to_model_settings_or_dict, str):
            with open(path_to_model_settings_or_dict) as fp:
                settings = json.load(fp)
        else:
            raise LeaspyModelInputError(
                "Bad type for model settings: should be a dict or a path "
                f"as a string, not {type(path_to_model_settings_or_dict)}"
            )

        ModelSettings._check_settings(settings)
        self.name: str = settings["name"].lower()
        self.parameters: KwargsType = settings["parameters"]
        self.hyperparameters: KwargsType = {
            k.lower(): v
            for k, v in settings.items()
            if k not in ("name", "parameters", "hyperparameters", "leaspy_version")
        }

    @staticmethod
    def _check_settings(settings: dict) -> None:
        for mandatory_key in ("name", "parameters"):
            if mandatory_key not in settings:
                raise LeaspyModelInputError(
                    f"The '{mandatory_key}' key is missing in the model "
                    "parameters (JSON file) you are loading."
                )
        if "leaspy_version" not in settings:
            raise LeaspyModelInputError(
                "The model you are trying to load was generated with a leaspy version < 1.1 "
                f"and is not compatible with your current version of leaspy == {__version__} "
                "because of a bug in the multivariate model which lead to under-optimal results.\n"
                "Please consider re-calibrating your model with your current leaspy version.\n"
                "If you really want to load it as is (at your own risk) please use leaspy == 1.0.*"
            )
