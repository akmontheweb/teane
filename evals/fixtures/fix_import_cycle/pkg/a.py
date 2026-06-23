from pkg.b import B


class A:
    def __init__(self) -> None:
        self.name = "A"

    def greet(self) -> str:
        return f"{self.name} hello from {B().name}"
