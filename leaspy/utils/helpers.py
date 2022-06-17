import pandas as pd

from leaspy.utils.typing import Dict, List, ParamType, Optional, Union
from leaspy.exceptions import LeaspyTypeError


def nest_parameters(
    flattened_source: Union[pd.DataFrame, dict],
    param_names: Optional[List[ParamType]] = None,
    sep: str = "_"
) -> pd.DataFrame:
    if isinstance(flattened_source, dict):
        is_dict = True
        nested_result = flattened_source.copy()
        raw_parameters = flattened_source.keys()
    elif isinstance(flattened_source, pd.DataFrame):
        is_dict = False
        nested_result = flattened_source.copy(deep=True)
        raw_parameters = flattened_source.columns.values
    else:
        raise LeaspyTypeError("Not implemented")

    nesting_plan: Dict[ParamType, int] = {}
    cols_to_test: List[ParamType] = [
        c for c in raw_parameters
        if (param_names is None) or (c.split(sep)[0] in param_names)
    ]

    while len(cols_to_test) > 0:
        for k in cols_to_test:          # e.g. "sources_1"
            split = k.split(sep)        # e.g. ["sources", "1"]
            if len(split) > 1:
                parent = sep.join(split[:-1])       # e.g. "sources"
                nesting_plan[parent] = max(
                    nesting_plan.get(parent, -1),   # Current list_length
                    int(split[-1]) + 1              # Candidate list_length
                )

        for k in nesting_plan.keys():
            children = [k + sep + str(i) for i in range(nesting_plan[k])]
            if is_dict:
                nested_result[k] = [
                    list(e)
                    for e in zip(*[nested_result[c] for c in children])
                ]
                for c in children:
                    _ = nested_result.pop(c)
            else:
                nested_result[k] = nested_result[children].values.tolist()
                nested_result.drop(columns=children, inplace=True)

        # Only newly created columns are candidates for nesting
        cols_to_test = list(nesting_plan.keys())
        nesting_plan = {}
    
    return nested_result