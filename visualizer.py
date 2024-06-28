import math
import pickle
import random
import time

import numpy as np
import pygame

from events import EventHandler
from label import Label
from neat2 import Population, NeuralNetwork
from snake import SnakeRenderer, SnakeEngine

pygame.init()


class Visualizer:

    def __init__(self) -> None:
        self.WIDTH = 1500
        self.HEIGHT = 1000
        self.FPS = 60

        self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("NEAT Visualizer")

        self.canvas = pygame.Surface(self.win.get_size(), pygame.SRCALPHA)

        self.game_engine = SnakeEngine(map_size=9)
        self.game_renderer = SnakeRenderer(self.game_engine, 600, 600)
        self.game_fps = 10
        self.last_game_update = 0

        self.event_handler = EventHandler()

        self.font16 = pygame.font.SysFont("arial", 16)

        self.population = Population(400, 26, 4, 1000)
        self.gen = self.population.run()

        self.update_gen = True
        self.nn = None
        self.outputs = [0] * 4
        self.steps = 0

        self.update_generation()

        self.running = True
        self.run()

    def run(self) -> None:
        while self.running:

            self.update()
            self.render()

            if not self.event_handler.loop():
                self.running = False

    def update(self) -> None:
        if not self.event_handler.is_key_pressed(pygame.K_TAB):
            self.update_generation()
            print(pickle.dumps(sorted([y for s in self.population.species for y in s.individuals], key=lambda x: x.fitness)[-1].genome))
            print(sorted([y for s in self.population.species for y in s.individuals], key=lambda x: x.fitness)[-1].fitness)

            sorted([y for s in self.population.species for y in s.individuals], key=lambda x: x.fitness)[-1].genome.show_info()
            print("L1: ", NeuralNetwork.calculate_network_layers(sorted([y for s in self.population.species for y in s.individuals], key=lambda x: x.fitness)[-1].genome))
            print("L2: ", NeuralNetwork.calculate_network_layers2(sorted([y for s in self.population.species for y in s.individuals], key=lambda x: x.fitness)[-1].genome))

            return

        if time.time() - self.last_game_update > 1/self.game_fps:
            self.update_neat_game()

            self.game_engine.update(self.game_renderer.dir)

            if self.game_engine.game_over or self.steps > 200:
                self.update_generation()

            self.last_game_update = time.time()
            
    def update_neat_game(self):
        head = self.game_engine.snake[0]
        dist_top_wall = head[1]
        dist_bottom_wall = self.game_engine.map_size - head[1] - 1
        dist_left_wall = head[0]
        dist_right_wall = self.game_engine.map_size - head[0] - 1
        if head[0] < self.game_engine.map_size + 1 - head[1]:
            dist_diagonal_NW = dist_top_wall * math.sqrt(2)
            dist_diagonal_SE = dist_left_wall * math.sqrt(2)
        else:
            dist_diagonal_NW = dist_right_wall * math.sqrt(2)
            dist_diagonal_SE = dist_bottom_wall * math.sqrt(2)

        if head[0] < head[1]:
            dist_diagonal_NE = dist_left_wall * math.sqrt(2)
            dist_diagonal_SW = dist_bottom_wall * math.sqrt(2)
        else:
            dist_diagonal_NE = dist_top_wall * math.sqrt(2)
            dist_diagonal_SW = dist_right_wall * math.sqrt(2)

        x = 0
        while (head[0] + x, head[1]) not in self.game_engine.snake:
            x += 1
            if head[0] + x >= self.game_engine.map_size - 1:
                break

        tail_S = x * math.sqrt(2)

        x = 0
        while (head[0] - x, head[1]) not in self.game_engine.snake:
            x += 1
            if head[0] - x <= 0:
                break

        tail_N = x * math.sqrt(2)

        x = 0
        while (head[0], head[1] + x) not in self.game_engine.snake:
            x += 1
            if head[1] + x >= self.game_engine.map_size - 1:
                break

        tail_W = x * math.sqrt(2)

        x = 0
        while (head[0], head[1] - x) not in self.game_engine.snake:
            x += 1
            if head[1] - x <= 0:
                break

        tail_E = x * math.sqrt(2)

        x = 0
        while (head[0] + x, head[1] + x) not in self.game_engine.snake:
            x += 1
            if head[0] + x >= self.game_engine.map_size - 1 or head[1] + x >= self.game_engine.map_size - 1:
                break

        tail_SW = x * math.sqrt(2)

        x = 0
        while (head[0] - x, head[1] - x) not in self.game_engine.snake:
            x += 1
            if head[0] - x <= 0 or head[1] - x <= 0:
                break

        tail_NE = x * math.sqrt(2)

        x = 0
        while (head[0] + x, head[1] - x) not in self.game_engine.snake:
            x += 1
            if head[0] + x >= self.game_engine.map_size - 1 or head[1] - x <= 0:
                break

        tail_NW = x * math.sqrt(2)

        x = 0
        while (head[0] - x, head[1] + x) not in self.game_engine.snake:
            x += 1
            if head[1] + x >= self.game_engine.map_size - 1 or head[0] - x <= 0:
                break

        tail_SE = x * math.sqrt(2)

        apple_N = 0
        apple_S = 0
        apple_W = 0
        apple_E = 0
        if self.game_engine.apple[0] > head[0]:
            apple_W = self.game_engine.apple[0] - head[0]
        if self.game_engine.apple[0] < head[0]:
            apple_E = head[0] - self.game_engine.apple[0]
        if self.game_engine.apple[1] > head[1]:
            apple_S = self.game_engine.apple[1] - head[1]
        if self.game_engine.apple[1] < head[1]:
            apple_N = head[1] - self.game_engine.apple[1]

        apple_distance = math.sqrt((head[0] - self.game_engine.apple[0]) ** 2 + (head[1] - self.game_engine.apple[1]) ** 2)

        inputs = [
            dist_top_wall,
            dist_bottom_wall,
            dist_right_wall,
            dist_left_wall,
            dist_diagonal_NE,
            dist_diagonal_SE,
            dist_diagonal_SW,
            dist_diagonal_NW,
            tail_N,
            tail_S,
            tail_W,
            tail_E,
            tail_NE,
            tail_SE,
            tail_SW,
            tail_NW,
            apple_N,
            apple_S,
            apple_W,
            apple_E,
            self.outputs[3],
            self.outputs[2],
            self.outputs[0],
            self.outputs[1],
            len(self.game_engine.snake),
            apple_distance
        ]

        self.outputs = self.nn.forward(inputs)
        dirx = 0
        diry = 0
        if max(self.outputs) == self.outputs[0]:
            dirx = 1
        elif max(self.outputs) == self.outputs[1]:
            dirx = -1
        elif max(self.outputs) == self.outputs[2]:
            diry = 1
        elif max(self.outputs) == self.outputs[3]:
            diry = -1

        self.game_renderer.update(self.event_handler, inputs=[dirx, diry])

        self.steps += 1

    def render(self) -> None:
        self.win.fill((20, 20, 20))

        render = self.game_renderer.render()
        self.canvas.blit(render, (0, 0))

        self.win.blit(self.canvas, (0, 0))

        pygame.display.update()
        self.clock.tick(self.FPS)

    def update_generation(self):
        next(self.gen)
        self.outputs = [0] * 4
        self.game_engine = SnakeEngine(map_size=9)
        self.game_renderer = SnakeRenderer(self.game_engine, 600, 600)
        best = sorted([y for s in self.population.species for y in s.individuals], key=lambda x: x.fitness)[-1]
        self.nn = NeuralNetwork(best.genome)
        self.steps = 0


if __name__ == "__main__":
    v = Visualizer()
    v.run()
