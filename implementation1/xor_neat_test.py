import time

from implementation1.neat import Population, NeuralNetwork
from implementation1.testing import Renderer


iters = 10
avg_gens = 0
avg_time = 0
avg_nodes = 0
for i in range(iters):
    print(i)
    pop = Population(150, 2, 1, 0)
    gen = 0
    t = time.perf_counter()
    for p in pop.run():
        best = sorted([y for s in p for y in s.individuals], key=lambda x: x.fitness)[-1]
        gen += 1
        if best.fitness > 3.8:
            inds = []
            for s in p:
                for i in s.individuals:
                    if i.fitness > 3.8:
                        inds.append(i)
            avg_nodes += len(sorted(inds, key=lambda i: len(i.genome.node_genes))[0].genome.node_genes)
            break

    avg_gens += gen
    avg_time += time.perf_counter() - t

print()
print("Avg Time: ", avg_time / iters)
print("Avg Generations: ", avg_gens / iters)
print("Avg Nodes Per Solution: ", avg_nodes / iters)

exit(0)

t1 = time.perf_counter()

population = Population(200, 2, 1, 0)
gen = 0
for p in population.run():
    best = sorted([y for s in p for y in s.individuals], key=lambda x: x.fitness)[-1]
    inds = []
    for s in p:
        for i in s.individuals:
            if i.fitness > 3.8:
                inds.append(i)

    if inds:
        best = sorted(inds, key=lambda i: len(i.genome.node_genes))[0]
    gen += 1
    if best.fitness > 3.8:
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
