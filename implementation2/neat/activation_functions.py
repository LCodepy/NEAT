import math


def sigmoid(x: float, k: float = 2, l: float = 0) -> float:
    return 1 / (1 + math.exp(-x * k + l))
