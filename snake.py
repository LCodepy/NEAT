import random
import time

import pygame

from events import EventHandler
from label import Label


class SnakeEngine:

    def __init__(self, map_size=20):
        self.map_size = map_size

        self.snake = [(self.map_size // 2, self.map_size // 2)]

        self.score = 0
        self.apple = (random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1))

        self.game_over = False

    def update(self, direction):
        if self.game_over:
            return

        if self.snake[0][0] == self.apple[0] and self.snake[0][1] == self.apple[1]:
            empty_tiles = list(set((x, y) for x in range(self.map_size) for y in range(self.map_size))
                               .difference(set([self.apple] + self.snake)))
            self.apple = random.choice(empty_tiles)
            self.snake.append((self.snake[-1][0] - direction[0], self.snake[-1][1] - direction[1]))
            self.score += 1

            if len(self.snake) >= self.map_size ** 2:
                self.game_over = True

        if (self.snake[0][0] + direction[0], self.snake[0][1] + direction[1]) in self.snake[1:]:
            self.game_over = True

        snake = []
        for i in range(1, len(self.snake)):
            snake.append(self.snake[i - 1])
        self.snake[1:] = snake

        self.snake[0] = (self.snake[0][0] + direction[0], self.snake[0][1] + direction[1])

        if (
                self.snake[0][0] < 0 or self.snake[0][0] > self.map_size - 1 or
                self.snake[0][1] < 0 or self.snake[0][1] > self.map_size - 1
        ):
            self.game_over = True

    def reset(self):
        self.game_over = False
        self.snake = [(self.map_size // 2, self.map_size // 2)]

        self.score = 0
        self.apple = (random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1))


class SnakeRenderer:

    def __init__(self, snake_engine: SnakeEngine, width, height):
        self.width = width
        self.height = height

        self.snake_engine = snake_engine

        self.canvas = pygame.Surface((width, height), pygame.SRCALPHA)

        self.map_size = snake_engine.map_size
        self.tile_size = self.width // self.map_size

        self.dir = [0, 0]

    def update(self, event_handler: EventHandler, inputs: list[float] = None):
        if event_handler.key_just_pressed(pygame.K_SPACE):
            self.snake_engine.reset()
            self.dir = [0, 0]

        if self.snake_engine.game_over:
            self.dir = [0, 0]
            return

        if inputs:
            self.dir = inputs
        else:
            self.update_input_check(event_handler)

    def update_input_check(self, event_handler):
        if event_handler.key_just_pressed(pygame.K_RIGHT):
            self.dir = [1, 0]
        elif event_handler.key_just_pressed(pygame.K_LEFT):
            self.dir = [-1, 0]
        elif event_handler.key_just_pressed(pygame.K_DOWN):
            self.dir = [0, 1]
        elif event_handler.key_just_pressed(pygame.K_UP):
            self.dir = [0, -1]

    def render(self):
        self.canvas.fill((0, 0, 0))

        for i in range(self.map_size):
            for j in range(self.map_size):
                x = i * self.tile_size
                y = j * self.tile_size
                color = (119, 82, 254)
                if j % 2 and i % 2 or not j % 2 and not i % 2:
                    color = (142, 143, 250)
                pygame.draw.rect(self.canvas, color, [x, y, self.tile_size, self.tile_size])

        pygame.draw.rect(self.canvas, (230, 50, 100),
                         [self.snake_engine.apple[0] * self.tile_size, self.snake_engine.apple[1] * self.tile_size, self.tile_size,
                          self.tile_size])

        for part in self.snake_engine.snake:
            pygame.draw.rect(self.canvas, (25, 4, 130),
                             [part[0] * self.tile_size, part[1] * self.tile_size, self.tile_size, self.tile_size])

        Label.render_text(self.canvas, "Score: " + str(self.snake_engine.score), (50, 20),
                          pygame.font.SysFont("arial", 25), (255, 255, 255), bold=True)

        if self.snake_engine.game_over:
            Label.render_text(self.canvas, "Game Over!", (self.width // 2, self.height // 2),
                              pygame.font.SysFont("arial", 40), (255, 255, 255), bold=True)

        return self.canvas



