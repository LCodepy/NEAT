import math
import random

import pygame
import pygame.gfxdraw

from neat.connection_factory import ConnectionFactory
from neat.genome import Genome
from neat.neural_network import NeuralNetwork
from neat.node_factory import NodeFactory
from rendering.event_handler import EventHandler
from rendering.label import Label


class GenomeRenderer:

    def __init__(self, genome: Genome, width: int, height: int):
        self.genome = genome
        self.width = width
        self.height = height

        self.fps = 60

        self.win = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE | pygame.SRCALPHA)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("NEAT Renderer")

        self.event_handler = EventHandler()

        self.running = False

    def run(self):
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
        if not self.event_handler.key_just_pressed(pygame.K_RETURN):
            return

        if random.random() > 0.2:
            if random.random() > 0.1:
                self.genome.mutate_change_weight()
            else:
                self.genome.mutate_assign_new_weight()
        if random.random() > 0.2:
            if random.random() > 0.1:
                self.genome.mutate_change_bias()
            else:
                self.genome.mutate_assign_new_bias()
        if random.random() > 0.7:
            for _ in range(20):
                if self.genome.mutate_add_connection():
                    break
        if random.random() > 0.9:
            self.genome.mutate_add_node()
        if random.random() > 0.9:
            self.genome.mutate_change_enabled()
        if random.random() > 0.9:
            self.genome.mutate_remove_node()
        if random.random() > 0.9:
            self.genome.mutate_remove_connection()

    def render(self) -> None:
        self.win.fill((20, 20, 20))

        # Render genome
        canvas = pygame.Surface((self.width * 3 / 4, self.height * 3 / 4), pygame.SRCALPHA)
        canvas.fill((0, 0, 0, 0))
        self.render_genome(canvas, self.genome)

        self.win.blit(canvas, (self.width // 2 - self.width * 3 // 8, self.height // 2 - self.height * 3 // 8))

    def on_resize(self) -> None:
        self.width, self.height = pygame.display.get_window_size()

    @staticmethod
    def render_genome(canvas: pygame.Surface, genome: Genome) -> None:
        width, height = canvas.get_size()

        layers = NeuralNetwork.calculate_network_layers(genome)

        node_width = min(width // (len(layers) + 1), 40)
        x_step = (width - node_width) // (len(layers) - 1)

        node_coordinates = {node.id: (0, 0) for node in genome.node_genes.values()}

        for i, layer in enumerate(layers):
            x = x_step * i
            y_step = (height - node_width) // (len(layer) - 1) if len(layer) > 1 else 0
            for j, node in enumerate(layer):
                y = y_step * j
                if len(layer) == 1:
                    y = height // 2 - node_width // 2
                node_coordinates[node] = (x, y)

        font_size = int(width / math.sqrt(width))

        for connection in genome.connection_genes.values():
            input_node = connection.input_node
            output_node = connection.output_node

            center1 = (node_coordinates[input_node][0] + node_width // 2, node_coordinates[input_node][1] + node_width // 2)
            center2 = (node_coordinates[output_node][0] + node_width // 2, node_coordinates[output_node][1] + node_width // 2)

            tx = center1[0] + (center2[0] - center1[0]) // 2
            ty = center1[1] + (center2[1] - center1[1]) // 2

            color = (160, 160, 160)
            color2 = (255, 255, 255)
            if not connection.enabled:
                color = (70, 70, 70)
                color2 = (90, 90, 90)
            pygame.draw.line(canvas, color, center1, center2)

            font = pygame.font.SysFont("arial", font_size - 4)
            angle = math.degrees(math.atan((center1[1] - center2[1]) / (center2[0] - center1[0])))
            text = pygame.transform.rotate(
                font.render(f"{connection.input_node}->{connection.output_node}", 1, color2), angle
            )
            canvas.blit(text, (tx - text.get_width() // 2, ty + (0 if ty > height // 2 else -1) * text.get_height()))

        for node_id, (x, y) in node_coordinates.items():
            pygame.gfxdraw.filled_circle(canvas, x + node_width // 2, y + node_width // 2, node_width // 2 - 1, (0, 120, 240))

            Label.render_text(canvas, str(node_id), (x + node_width // 2, y + node_width // 2), pygame.font.SysFont("arial", font_size),
                              (255, 255, 255), bold=True)


if __name__ == "__main__":
    genome = Genome(0, 2, 1, ConnectionFactory(2), NodeFactory(4))
    renderer = GenomeRenderer(genome, 600, 600)
    renderer.run()
