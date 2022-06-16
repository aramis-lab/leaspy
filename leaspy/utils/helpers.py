import pandas as pd

from leaspy.utils.typing import Dict, List, ParamType, Optional


def nest_parameters(
    flattened_df: pd.DataFrame,
    param_names: Optional[List[ParamType]] = None,
    sep: str = "_"
) -> pd.DataFrame:
    # TODO Rather a base function for dicts, convert pd.Df -> dict before use
    nested_df = flattened_df.copy(deep=True)
    nesting_plan: Dict[ParamType, int] = {}
    cols_to_test: List[ParamType] = [
        c for c in nested_df.columns.values
        if (param_names is None) or (c.split(sep)[0] in param_names)
    ]

    while len(cols_to_test) > 0:
        for c in cols_to_test:          # e.g. "sources_1"
            split = c.split(sep)        # e.g. ["sources", "1"]
            if len(split) > 1:
                parent = sep.join(split[:-1])       # e.g. "sources"
                nesting_plan[parent] = max(
                    nesting_plan.get(parent, -1),   # Current list_length
                    int(split[-1]) + 1              # Candidate list_length
                )

        for c in nesting_plan.keys():
            children = [c + sep + str(i) for i in range(nesting_plan[c])]
            nested_df[c] = nested_df[children].values.tolist()
            nested_df.drop(columns=children, inplace=True)

        # Only newly created columns are candidates for nesting
        cols_to_test = list(nesting_plan.keys())
        nesting_plan = {}
    
    return nested_df