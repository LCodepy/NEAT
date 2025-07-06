import time

import pygame

from implementation2.neat.neatconfig import NEATConfig
from implementation2.neat.evaluation_functions import xor_evaluation_function
from implementation2.neat.population import Population
from implementation2.rendering.button import Button
from implementation2.rendering.event_handler import EventHandler
from implementation2.rendering.genome_renderer import GenomeRenderer
from implementation2.rendering.label import Label


class XORRenderer:

    def __init__(self) -> None:
        pygame.init()

        self.fps = 60

        self.width, self.height = (1200, 800)

        self.win = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE | pygame.SRCALPHA)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("NEAT XOR Renderer")

        self.event_handler = EventHandler()

        self.simulation_surface = pygame.Surface((self.width * 3 / 4, self.height * 5 / 6), pygame.SRCALPHA)
        self.simulation_surface_pos = (10, 60)

        def on_pause_btn_click(cls: XORRenderer, x, y):
            if cls.neat_running:
                cls.pause_button.set_text("Run")
            else:
                cls.pause_button.set_text("Pause")
            cls.neat_running = not cls.neat_running
            if not cls.neat_started:
                cls.start_neat()

        self.pause_button = Button(
            self.win,
            (10 + self.simulation_surface.get_width() // 2, self.simulation_surface_pos[1] + self.simulation_surface.get_height() + 35),
            (100, 40),
            pygame.font.SysFont("arial", 28),
            text="Run",
            color=(100, 100, 100),
            hover_color=(80, 80, 80),
            bold=True,
            border_radius=4
        )
        self.pause_button.set_on_click_listener(on_pause_btn_click, self)

        # NEAT

        self.neat_config = NEATConfig()

        self.population = Population(200, 2, 1, 500, self.neat_config, xor_evaluation_function)

        self.rendering_genome = False
        self.genome_to_render = None

        self.neat = None
        self.neat_started = False
        self.neat_running = False
        self.neat_frame_duration = 0.4
        self.neat_last_frame = 0

        self.current_neat_generation = 0

        self.running = False

    def run(self) -> None:
        self.running = True

        while self.running:
            self.clock.tick(self.fps)
            pygame.display.update()

            if self.event_handler.resizing:
                self.on_resize()

            self.update()
            self.render()

            if not self.event_handler.loop():
                self.running = False

    def update(self) -> None:
        self.pause_button.update(self.event_handler)

        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            self.genome_to_render = None
            self.rendering_genome = False
            self.pause_button.set_text("Run")

        if not self.neat or not self.neat_running:
            return

        if time.time() - self.neat_last_frame >= self.neat_frame_duration:
            self.current_neat_generation = next(self.neat)
            self.neat_last_frame = time.time()

    def render(self) -> None:
        self.win.fill((50, 50, 50))

        self.simulation_surface.fill((20, 20, 20))

        if self.rendering_genome:
            self.neat_running = False
            surf = pygame.Surface((self.simulation_surface.get_width() - 20, self.simulation_surface.get_height() - 20), pygame.SRCALPHA)
            GenomeRenderer.render_genome(surf, self.genome_to_render)
            self.simulation_surface.blit(surf, (10, 10))

            self.win.blit(self.simulation_surface, self.simulation_surface_pos)
            return

        Label.render_text(self.win, "Generation: " + str(self.current_neat_generation), (self.simulation_surface_pos[0] + self.simulation_surface.get_width() // 2, 30), pygame.font.SysFont("arial", 40), (255, 255, 255), True)

        self.pause_button.render()

        n = len(self.population.species)
        for i in range(n):
            width = self.simulation_surface.get_width() - 16
            height = min((self.simulation_surface.get_height() - 12) / n, 100)
            pygame.draw.rect(
                self.simulation_surface,
                (100, 120, 255),
                [8, 6 + i * height + 2, width, height - 4],
                width=1, border_radius=8
            )

            individuals = self.population.species[i].individuals.copy()
            if len(individuals) > 15:
                individuals = individuals[:15]

            for j, individual in enumerate(individuals):
                network_w = min(width / len(individuals), 60)
                surf = pygame.Surface((network_w - 8, height - 10), pygame.SRCALPHA)
                surf.fill((0, 0, 0, 0))
                GenomeRenderer.render_genome(surf, individual.genome, draw_text=False)
                self.simulation_surface.blit(surf, (12 + j * network_w, 12 + i * height))

                if pygame.Rect(self.simulation_surface_pos[0] + 12 + j * network_w, self.simulation_surface_pos[1] + 12 + i * height, network_w - 8, height - 10).collidepoint(*pygame.mouse.get_pos()):
                    surf2 = pygame.Surface((network_w - 4, height - 6), pygame.SRCALPHA)
                    surf2.fill((0, 0, 0, 0))
                    pygame.draw.rect(surf2, (200, 200, 200, 100), [0, 0, network_w - 4, height - 6], border_radius=4)
                    self.simulation_surface.blit(surf2, (10 + j * network_w, 10 + i * height))

                    if pygame.mouse.get_pressed()[0]:
                        self.rendering_genome = True
                        self.genome_to_render = individual.genome

        self.win.blit(self.simulation_surface, self.simulation_surface_pos)

    def on_resize(self) -> None:
        self.width, self.height = pygame.display.get_window_size()

        self.simulation_surface = pygame.Surface((self.width * 3 / 4, self.height * 5 / 6), pygame.SRCALPHA)

        self.pause_button.move(10 + self.simulation_surface.get_width() // 2, self.simulation_surface_pos[1] + self.simulation_surface.get_height() + 35)

    def start_neat(self) -> None:
        self.neat = self.population.run()
        self.neat_started = True


xor = XORRenderer()
xor.run()
