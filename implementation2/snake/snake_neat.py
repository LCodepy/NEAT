import math
import pickle
import time

import numpy

from implementation2.snake.snake import SnakeEngine
from implementation2.neat.neatconfig import NEATConfig
from implementation2.neat.neural_network import NeuralNetwork
from implementation2.neat.population import Population
from implementation2.utils.functions import softmax


def snake_evaluation(nn: NeuralNetwork) -> float:
    DIR_VEC = {
        "U": (0, -1),
        "R": (1, 0),
        "D": (0, 1),
        "L": (-1, 0)
    }

    LEFT = {"U": "L", "L": "D", "D": "R", "R": "U"}
    RIGHT = {v: k for k, v in LEFT.items()}

    steps = 0
    dry_steps = 0
    penalty = 0
    fitness = 0

    snake_engine = SnakeEngine()
    while not snake_engine.game_over:
        if dry_steps > 200:
            # penalty += 25 + fitness
            fitness = 0
            break

        hx, hy = snake_engine.snake[0]
        ax, ay = snake_engine.apple
        body = set(snake_engine.snake)
        direction = list(filter(lambda k: DIR_VEC[k] == snake_engine.direction, DIR_VEC.keys()))[0]

        dirs = {
            "F": direction,
            "L": LEFT[direction],
            "R": RIGHT[direction]
        }

        # Wall or tail immediately ahead
        obstacles = []
        for key in ("F", "L", "R"):
            dx, dy = DIR_VEC[dirs[key]]
            nx, ny = hx + dx, hy + dy
            obstacles.append(int(snake_engine.is_wall(nx, ny) or (nx, ny) in body))

        # Distance to the nearest tail segment
        tail_dists = []
        for key in ("F", "L", "R"):
            dx, dy = DIR_VEC[dirs[key]]
            step, found = 1, 0
            x, y = hx + dx, hy + dy
            while not snake_engine.is_wall(x, y):
                if (x, y) in body:
                    found = 1 / step
                    break
                step += 1
                x, y = x + dx, y + dy

            tail_dists.append(found)

        # Apple direction
        food_right = int(ax > hx)
        food_below = int(ay > hy)

        # Current direction
        dir = [int(direction == d) for d in ("U", "R", "D")]

        inputs = [
            obstacles[0],   # Wall in front
            obstacles[1],   # Wall left
            obstacles[2],   # Wall right
            tail_dists[0],  # Tail distance forward
            tail_dists[1],  # Tail distance left
            tail_dists[2],  # Tail distance right
            food_right,     # Food to the right
            food_below,     # Food below
            dir[0],         # Currently going up
            dir[1],         # Currently going right
            dir[2]          # Currently going down
        ]

        outputs = nn.forward(inputs)
        softmax_outputs = softmax(outputs)

        prev_score = snake_engine.score

        # Calculate new direction based on the highest probability
        new_dir = snake_engine.direction
        if abs(max(softmax_outputs) - softmax_outputs[1]) < 1e-6:
            new_dir = DIR_VEC[LEFT[direction]]
        elif abs(max(softmax_outputs) - softmax_outputs[2]) < 1e-6:
            new_dir = DIR_VEC[RIGHT[direction]]

        prev_apple_dist = abs(ax - hx) + abs(ay - hy)

        snake_engine.update(new_dir)

        new_apple_dist = abs(snake_engine.apple[0] - snake_engine.snake[0][0]) + abs(snake_engine.apple[1] - snake_engine.snake[0][1])
        # fitness += (prev_apple_dist - new_apple_dist) * 0.1

        steps += 1
        dry_steps += 1

        fitness += 0.05

        # If snake eats an apple we reset the number of steps without eating apple
        if snake_engine.score > prev_score:
            dry_steps = 0
            fitness += 1

    return fitness


config = NEATConfig()
config.max_allowed_generations_since_improved = 100
config.add_node_mutation_chance = 0.05
config.remove_node_mutation_chance = 0.05

population = Population(150, 11, 3, 10000, config, snake_evaluation)

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
