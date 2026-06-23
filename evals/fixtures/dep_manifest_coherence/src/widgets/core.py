from dataclasses import dataclass


@dataclass
class Widget:
    name: str
    weight: float = 0.0

    def heavier_than(self, other: "Widget") -> bool:
        return self.weight > other.weight
