from pkg.a import A


class B:
    def __init__(self) -> None:
        self.name = "B"

    def partner(self) -> str:
        return A().name
