from enum import Enum

__all__ = [
    "StrEnum",
    "CombinedStrEnum",
]


class StrEnum(str, Enum):
    """Enum where members are also strings"""

    pass


class CombinedStrEnum(StrEnum):
    @classmethod
    def create_union_enum(cls, *enums):
        members = {}
        for enum in enums:
            for item in enum:
                members[item.name] = item.value
        return StrEnum(cls.__name__, members)
