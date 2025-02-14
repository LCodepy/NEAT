from neat.population import Population

population = Population(200, 2, 1, 200)

population.run()

print(population.get_sorted_individuals()[-1])
