from typing import Any


class Environment:
    def __init__(self, parent: "Environment | None" = None) -> None:
        self.parent = parent
        self.shapes: dict[str, tuple[Any, ...] | None] = {}
        self.scalar_values: dict[str, int | float] = {}

    def create_child(self) -> "Environment":
        """Factory method to spawn a nested local scope."""
        return Environment(parent=self)

    def get_shape(self, name: str) -> tuple[Any, ...] | None:
        if name in self.shapes:
            return self.shapes[name]
        if self.parent is not None:
            return self.parent.get_shape(name)
        return None

    def set_shape(self, name: str, shape: tuple[Any, ...] | None) -> None:
        self.shapes[name] = shape

    def get_scalar(self, name: str) -> int | float | None:
        if name in self.scalar_values:
            return self.scalar_values[name]
        if self.parent is not None:
            return self.parent.get_scalar(name)
        return None

    def set_scalar(self, name: str, value: float) -> None:
        self.scalar_values[name] = value
