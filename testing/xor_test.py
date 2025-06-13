from neat.config import Config
from neat.population import Population
from rendering.genome_renderer import GenomeRenderer

config = Config()

population = Population(1000, 2, 1, 200, config)

population.run()

print(population.get_sorted_individuals()[-1])

genome = sorted(filter(lambda i: i.fitness > 3.5, population.get_sorted_individuals()), key=lambda i: i.genome.size())[0].genome
renderer = GenomeRenderer(genome, 800, 600)
renderer.run()
