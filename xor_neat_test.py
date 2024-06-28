import time

from neat import Population, NeuralNetwork
from testing import Renderer

t1 = time.perf_counter()

population = Population(150, 2, 1, 0)
gen = 0
for p in population.run():
    best = sorted([y for s in p for y in s.individuals], key=lambda x: x.fitness)[-1]
    gen += 1
    if gen >= 200:
        break
    print(best.fitness)

t2 = time.perf_counter()

print("TIME: ", t2 - t1)

print()
print("BEST")
best.genome.show_info()
print(best.genome.connection_genes)
print(best.genome.node_genes)

nn = NeuralNetwork(best.genome)
print(nn.forward([0, 0]))
print(nn.forward([0, 1]))
print(nn.forward([1, 0]))
print(nn.forward([1, 1]))

r = Renderer(best.genome, 1000, 800)
r.init()
r.render(True)
