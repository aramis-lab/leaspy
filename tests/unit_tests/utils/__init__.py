import torch

dummy_pop_variable_infos = {
    "name": "dummy-pop",
    "shape": torch.Size([4]),
    "type": "population",
    "rv_type": "multigaussian"
}

dummy_ind_variable_infos = {
    "name": "dummy-ind",
    "shape": torch.Size([1]),
    "type": "individual",
    "rv_type": "gaussian"
}
