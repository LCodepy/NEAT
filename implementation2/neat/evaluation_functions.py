from implementation2.neat.neural_network import NeuralNetwork


def xor_evaluation_function(nn: NeuralNetwork) -> float:
    out1 = nn.forward([0, 0])[0]
    out2 = nn.forward([0, 1])[0]
    out3 = nn.forward([1, 0])[0]
    out4 = nn.forward([1, 1])[0]

    return 1 - out1 + out2 + out3 + 1 - out4
