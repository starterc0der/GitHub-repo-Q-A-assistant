"""A tiny sample module used as a chunking fixture."""


def add(a: int, b: int) -> int:
    return a + b


def subtract(a: int, b: int) -> int:
    return a - b


class Calculator:
    def __init__(self, start: int = 0):
        self.value = start

    def apply(self, op: str, amount: int) -> int:
        if op == "add":
            self.value = add(self.value, amount)
        elif op == "subtract":
            self.value = subtract(self.value, amount)
        return self.value
