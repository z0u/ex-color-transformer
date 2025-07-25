from dataclasses import dataclass
from typing import Collection


@dataclass(slots=True)
class Mark:
    count: int
    label: str


@dataclass(slots=True)
class BarData:
    count: int
    total: int
    description: str
    suffix: str
    elapsed_time: float
    markers: Collection[Mark]

    @property
    def fraction(self) -> float:
        return (self.count / self.total) if self.total > 0 else 1
