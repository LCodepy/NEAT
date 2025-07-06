import pickle
import time

import numpy

from implementation2.neat.neatconfig import NEATConfig
from implementation2.neat.neural_network import NeuralNetwork
from implementation2.neat.population import Population
from implementation2.snake.snake import SnakeEngine
from implementation2.utils.functions import softmax


def snake_evaluation(nn: NeuralNetwork) -> float:

    def look(dx, dy):
        wall_dist = 0
        tail_dist = 0
        apple_dist = 0
        x, y = snake_engine.snake[0]
        step = 0

        while 0 <= x < snake_engine.map_size and 0 <= y < snake_engine.map_size:
            x += dx
            y += dy
            step += 1

            if 0 <= x < snake_engine.map_size and 0 <= y < snake_engine.map_size:
                if wall_dist == 0:
                    wall_dist = step
                if (x, y) in snake_engine.snake and tail_dist == 0:
                    tail_dist = step / snake_engine.map_size
                if (x, y) == snake_engine.apple and apple_dist == 0:
                    apple_dist = step / snake_engine.map_size

        wall_dist = step / snake_engine.map_size
        return wall_dist, tail_dist, apple_dist

    steps = 0
    dry_steps = 0

    fitness = 0

    # print("--------------------------")
    snake_engine = SnakeEngine()
    while not snake_engine.game_over:
        if dry_steps > 400:
            fitness = 0
            break

        head = snake_engine.snake[0]
        apple = snake_engine.apple

        wall_dists = []
        tail_dists = []
        apple_dists = []

        for dx, dy in ((1, 0), (-1, 0), (0, -1), (0, 1)):  # Right, Left, Up, Down
            wd, td, ad = look(dx, dy)
            wall_dists.append(wd)
            tail_dists.append(td)
            apple_dists.append(ad)

        apple_right = int(apple[0] > head[0])
        apple_below = int(apple[1] > head[1])

        apple_x_dist = abs(apple[0] - head[0]) / snake_engine.map_size
        apple_y_dist = abs(apple[1] - head[1]) / snake_engine.map_size

        inputs = [
            *([0] * 4), #*wall_dists,
            *([0] * 4), #*tail_dists,
            apple_right,
            apple_below,
            apple_x_dist,
            apple_y_dist
        ]

        outputs = nn.forward(inputs)  # Right, Left, Up, Down
        softmax_outputs = softmax(outputs)

        # print(inputs, outputs, softmax_outputs, fitness)

        # new_dir = [(1, 0), (-1, 0), (0, -1), (0, 1)][numpy.random.choice(4, p=softmax_outputs)]

        # print(new_dir)

        new_dir = (1, 0)
        if abs(max(softmax_outputs) - softmax_outputs[1]) < 1e-6:
            new_dir = (-1, 0)
        elif abs(max(softmax_outputs) - softmax_outputs[2]) < 1e-6:
            new_dir = (0, -1)
        elif abs(max(softmax_outputs) - softmax_outputs[3]) < 1e-6:
            new_dir = (0, 1)

        # print(new_dir)

        prev_score = snake_engine.score
        prev_apple_dist = abs(apple[0] - head[0]) + abs(apple[1] - head[1])

        snake_engine.update(new_dir)

        new_apple_dist = abs(snake_engine.apple[0] - snake_engine.snake[0][0]) + abs(snake_engine.apple[1] - snake_engine.snake[0][1])
        # fitness += (prev_apple_dist - new_apple_dist) * 0.1

        steps += 1
        dry_steps += 1

        fitness += 0.1

        if snake_engine.score > prev_score:
            dry_steps = 0
            fitness += 2

        # if snake_engine.collided_with_tail:
        #     fitness -= 5

    return fitness


config = NEATConfig()

population = Population(150, 12, 4, 10000, config, snake_evaluation)

avg_hidden_nodes = 0

gen = population.run()
while True:
    try:
        t = time.perf_counter()
        generation = next(gen)
        individuals = population.get_sorted_individuals()
        print("\n-------------------------------{ Generation " + str(generation) + " }-------------------------------")
        print(" -> Time:", round(time.perf_counter() - t, 2))
        print(" -> Species:", len(population.species))
        print(" -> Best fitness:", individuals[-1].fitness)
        print(f" -> Avg. fitness: {sum(map(lambda i: i.fitness, individuals)) / len(individuals):.2f}")
        print(" -> Best pickle:", pickle.dumps(individuals[-1].genome))

        hidden = list(map(lambda i: len(i.genome.get_hidden_neurons()), individuals))
        avg_hidden = sum(hidden) / len(individuals)

        print(f" -> Hidden nodes: Average = {avg_hidden_nodes:.2f}, Max = {max(hidden)}")

    except StopIteration:
        break