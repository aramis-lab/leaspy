from leaspy import Leaspy


def get_model_derived_parameters(model: dict):
    """Returns a dictionary of model derived parameters (i.e. not stored in the json file)."""

    # check that the model is loadable
    try:
        lsp = Leaspy.load(model)
    except Exception as e:
        return {"error": str(e)}

    derived_params = {}

    # currently we only return the mixing-matrix
    if getattr(lsp.model, "source_dimension", 0):
        derived_params["mixing_matrix"] = lsp.model.attributes.mixing_matrix.tolist()
    else:
        derived_params["mixing_matrix"] = None

    return derived_params
