from implementation2.neat.neatconfig import NEATConfig
from implementation2.neat.evaluation_functions import xor_evaluation_function
from implementation2.neat.population import Population
from implementation2.rendering.genome_renderer import GenomeRenderer

config = NEATConfig()

population = Population(200, 2, 1, 200, config, xor_evaluation_function)

gen = population.run()
while True:
    try:
        print("Generation: ", next(gen))
        print(population.get_sorted_individuals()[-1])
    except StopIteration:
        break

print(population.get_sorted_individuals()[-1])

genome = sorted(filter(lambda i: i.fitness > 3.5, population.get_sorted_individuals()), key=lambda i: i.genome.size())[0].genome
renderer = GenomeRenderer(genome, 800, 600)
renderer.run()
