import math


def softmax(x: list[float]):
    e_x = [math.exp(i) for i in x]
    total = sum(e_x)
    return [i / total for i in e_x]