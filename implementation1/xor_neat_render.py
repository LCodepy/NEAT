import pygame

import pickle

from events import EventHandler
from label import Label
from neat import Population
from testing import Renderer


class Main:

    def __init__(self):
        self.WIDTH = 1200
        self.HEIGHT = 1000
        self.FPS = 60

        self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("NEAT Visualizer")

        self.canvas = pygame.Surface(self.win.get_size(), pygame.SRCALPHA)

        self.event_handler = EventHandler()

        self.population = Population(150, 2, 1, 100)
        self.generations = self.population.run()
        next(self.generations)

        self.paused = True

        self.running = False

    def run(self) -> None:
        self.running = True
        while self.running:

            self.update()
            self.render()

            if not self.event_handler.loop():
                self.running = False

    def update(self) -> None:
        if self.event_handler.is_key_pressed(pygame.K_RIGHT) or not self.paused:
            next(self.generations)
            print(sorted([y for s in self.population.species for y in s.individuals], key=lambda x: x.fitness)[-1].fitness)
            print(pickle.dumps(sorted([y for s in self.population.species for y in s.individuals], key=lambda x: x.fitness)[-1]))

        if self.event_handler.key_just_pressed(pygame.K_TAB):
            for _ in range(10):
                next(self.generations)
                print(sorted([y for s in self.population.species for y in s.individuals], key=lambda x: x.fitness)[-1].fitness)

        # if sorted([y for s in self.population.species for y in s.individuals], key=lambda x: x.fitness)[-1].fitness > 3 and not self.paused:
        #     self.paused = True
        #     print(sorted([y for s in self.population.species for y in s.individuals], key=lambda x: x.fitness)[-1].fitness)
        #     print(pickle.dumps(sorted([y for s in self.population.species for y in s.individuals], key=lambda x: x.fitness)[-1]))

        if self.event_handler.key_just_pressed(pygame.K_SPACE):
            self.paused = not self.paused

    def render(self) -> None:
        self.win.fill((0, 0, 0))
        self.canvas.fill((0, 0, 0))

        n_species = len(self.population.species)
        species = self.population.species

        if species:
            max_sp = max(list(map(lambda p: p.get_size(), species)))
            if max_sp > 30:
                max_sp = 30
        else:
            max_sp = 20

        rows = [[]]
        cur_row = 0
        for i in range(n_species):
            l = 0
            for s in rows[cur_row]:
                l += s.get_size()
            if l + species[i].get_size() > max_sp:
                cur_row += 1
                rows.append([])
            rows[cur_row].append(species[i])

        row_h = (self.canvas.get_height() - 40 - len(rows) * 5) // len(rows)

        for i in range(len(rows)):
            y = 20 + i * (row_h + 5)
            cur_width = 0
            for j, sp in enumerate(rows[i]):
                x = 20 + cur_width + 5 * j
                cell_w = (self.canvas.get_width() - 40 - 5 * j) // max_sp
                sp_w = cell_w * sp.get_size()
                cur_width += sp_w

                for k, individual in enumerate(sorted(sp.individuals, key=lambda ind: ind.fitness, reverse=True)):
                    renderer = Renderer(individual.genome, cell_w, row_h - 14)
                    renderer.render_network()
                    self.canvas.blit(renderer.canvas, (x + k * cell_w, y + 14))
                    Label.render_text(self.canvas, str(round(individual.fitness, 3)),
                                      (x + k * cell_w + cell_w // 2, y + row_h - 8), pygame.font.SysFont("arial", 12),
                                      (255, 255, 255), bold=True)

                Label.render_text(self.canvas, "S" + str(sp.id),
                                  (x + 4, y), pygame.font.SysFont("arial", 16),
                                  (255, 255, 255), bold=True, centered=False)

                pygame.draw.rect(self.canvas, (60, 60, 60), (x, y, sp_w, row_h), border_radius=3, width=1)

        self.win.blit(self.canvas, (0, 0))

        pygame.display.update()
        self.clock.tick(self.FPS)


m = Main()
m.run()
