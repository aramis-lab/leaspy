from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, Iterator, Tuple, TypeVar
from typing import Mapping as TMapping

KT = TypeVar("KT")
VT = TypeVar("VT")


@dataclass(frozen=True)
class FilteredMappingProxy(Mapping, Generic[KT, VT]):
    """
    Efficient filtered (with order) proxy (= no direct assignment) of a referenced mapping.

    Attributes
    ----------
    mapping : TMapping[KT, VT]
    subset : Tuple[KT, ...]
    check_keys : bool
    """

    mapping: TMapping[KT, VT]
    subset: Tuple[KT, ...]
    check_keys: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        # TODO? only store `types.MappingProxyType(self.mapping)` to be sure no write access?
        set_subset = set(self.subset)
        if len(set_subset) != len(self.subset):
            raise ValueError("Duplicate keys in the subset")
        if self.check_keys:
            unknown_keys = set_subset.difference(self.mapping)
            if len(unknown_keys):
                raise ValueError(f"Unknown keys in the subset: {unknown_keys}")

    def __len__(self) -> int:
        return len(self.subset)

    def __iter__(self) -> Iterator[KT]:
        return iter(self.subset)

    def __getitem__(self, k: KT) -> VT:
        if k not in self.subset:
            raise KeyError(k)
        return self.mapping[k]
