from dataclasses import dataclass

__all__ = ["constants"]

@dataclass(frozen=True)
class LeaspyConstants:
    # The real infinity is not used beacause there are multiplication by zero that creates errors (nan)
    INFINITY = float(10**307)

constants = LeaspyConstants()
