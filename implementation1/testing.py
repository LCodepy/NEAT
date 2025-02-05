import threading

import pygame

from events import EventHandler
from label import Label
# from neat2 import NodeFactory, ConnectionFactory, GenomeFactory, Individual, NeuralNetwork, Genome, Population
from neat import *


class Renderer:

    def __init__(self, genome: Genome, width: int, height: int):
        self.genome = genome

        self.W, self.H = width, height
        self.FPS = 60

        self.canvas = pygame.Surface((self.W, self.H))

    def init(self):
        self.win = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("NEAT Renderer")

        self.canvas = pygame.Surface(self.win.get_size())

        self.event_handler = EventHandler()

        self.running = False

    def render(self, show_labels=False, show_data=False):
        self.running = True

        while self.running:

            self.clock.tick(self.FPS)
            pygame.display.update()

            self.win.fill((0, 0, 0))

            self.W, self.H = self.win.get_size()
            self.canvas = pygame.Surface(self.win.get_size())

            self.render_network(show_labels=show_labels, show_data=show_data)

            self.win.blit(self.canvas, (0, 0))

            if not self.event_handler.loop():
                self.running = False

    def render_network(self, show_labels=False, show_data=False):
        self.canvas.fill((0, 0, 0))

        layers = NeuralNetwork.calculate_network_layers(self.genome)

        x_dist = self.W // len(layers)

        if isinstance(self.genome.node_genes, list):
            node_coordinates = {node.id: (0, 0) for node in self.genome.node_genes}
        else:
            node_coordinates = {node.id: (0, 0) for node in self.genome.node_genes.values()}

        for i, layer in enumerate(layers):
            x = self.W // 2 - (len(layers) * x_dist - x_dist) // 2 + i * x_dist
            y_dist = self.H // max(map(len, layers)) // 1.5
            if i != 0 and i != len(layers)-1:
                y_dist *= 1.5
            for j, node in enumerate(layer):
                y = self.H // 2 - (len(layer) * y_dist - y_dist) // 2 + j * y_dist
                node_coordinates[node] = (x, y)

        font_size = 18
        if min(self.W, self.H) < 800:
            font_size = 14
        elif min(self.W, self.H) < 500:
            font_size = 12

        connection_genes = self.genome.connection_genes
        if isinstance(connection_genes, dict):
            connection_genes = connection_genes.values()

        for connection in connection_genes:
            input_node = connection.input_node
            output_node = connection.output_node

            center1 = node_coordinates[input_node]
            center2 = node_coordinates[output_node]

            tx = center1[0] + (center2[0] - center1[0]) // 2
            ty = center1[1] + (center2[1] - center1[1]) // 2

            color = (180, 180, 180)
            color2 = (255, 100, 100)
            if not connection.enabled:
                color = (40, 40, 40)
                color2 = (80, 20, 20)
            pygame.draw.line(self.canvas, color, center1, center2, 1)

            if show_labels:
                Label.render_text(self.canvas, f"{connection.input_node}->{connection.output_node}",
                                  (tx, ty), pygame.font.SysFont("arial", font_size - 4), color2, bold=True)
            if show_data:
                Label.render_text(self.canvas, str(round(connection.weight, 2)), (tx, ty - font_size),
                                  pygame.font.SysFont("arial", font_size - 4), color2, bold=True)

        for node_id, (x, y) in node_coordinates.items():
            pygame.draw.circle(self.canvas, (0, 0, 0), (x, y), max(self.W // 80, 6))
            pygame.draw.circle(self.canvas, (255, 255, 255), (x, y), max(self.W // 80, 6), 2)

            if show_labels:
                Label.render_text(self.canvas, str(node_id), (x, y), pygame.font.SysFont("arial", font_size - 5),
                                  (255, 255, 255), bold=True)

            if show_data:
                Label.render_text(self.canvas, str(round(self.genome.get_node(node_id).bias, 2)), (x, y - font_size - 5),
                                  pygame.font.SysFont("arial", font_size), (255, 255, 255), bold=True)

    def update_size(self, width, height):
        self.W = width
        self.H = height
        self.canvas = pygame.Surface((self.W, self.H))


# from neat2 import *


class Renderer2:

    def __init__(self, genome: Genome, width: int, height: int):
        self.genome = genome

        self.W, self.H = width, height
        self.FPS = 60

        self.canvas = pygame.Surface((self.W, self.H))

    def init(self):
        self.win = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("NEAT Renderer")

        self.canvas = pygame.Surface(self.win.get_size())

        self.event_handler = EventHandler()

        self.running = False

    def render(self, show_labels=False, show_data=False):
        self.running = True

        while self.running:

            self.clock.tick(self.FPS)
            pygame.display.update()

            self.win.fill((0, 0, 0))

            self.W, self.H = self.win.get_size()
            self.canvas = pygame.Surface(self.win.get_size())

            self.render_network(show_labels=show_labels, show_data=show_data)

            self.win.blit(self.canvas, (0, 0))

            if not self.event_handler.loop():
                self.running = False

    def render_network(self, show_labels=False, show_data=False):
        self.canvas.fill((0, 0, 0))

        layers = NeuralNetwork.calculate_network_layers(self.genome)

        x_dist = self.W // len(layers)

        node_coordinates = {node.id: (0, 0) for node in self.genome.node_genes.values()}

        for i, layer in enumerate(layers):
            x = self.W // 2 - (len(layers) * x_dist - x_dist) // 2 + i * x_dist
            y_dist = self.H // max(map(len, layers)) // 1.5
            if i != 0 and i != len(layers)-1:
                y_dist *= 1.5
            for j, node in enumerate(layer):
                y = self.H // 2 - (len(layer) * y_dist - y_dist) // 2 + j * y_dist
                node_coordinates[node] = (x, y)

        font_size = 18
        if min(self.W, self.H) < 800:
            font_size = 14
        elif min(self.W, self.H) < 500:
            font_size = 12
        for connection in self.genome.connection_genes.values():
            input_node = connection.input_node
            output_node = connection.output_node

            center1 = node_coordinates[input_node]
            center2 = node_coordinates[output_node]

            tx = center1[0] + (center2[0] - center1[0]) // 2
            ty = center1[1] + (center2[1] - center1[1]) // 2

            color = (180, 180, 180)
            color2 = (255, 100, 100)
            if not connection.enabled:
                color = (40, 40, 40)
                color2 = (80, 20, 20)
            pygame.draw.line(self.canvas, color, center1, center2, 1)

            if show_labels:
                Label.render_text(self.canvas, f"{connection.input_node}->{connection.output_node}",
                                  (tx, ty), pygame.font.SysFont("arial", font_size - 4), color2, bold=True)
            if show_data:
                Label.render_text(self.canvas, str(round(connection.weight, 2)), (tx, ty - font_size),
                                  pygame.font.SysFont("arial", font_size - 4), color2, bold=True)

        for node_id, (x, y) in node_coordinates.items():
            pygame.draw.circle(self.canvas, (0, 0, 0), (x, y), max(self.W // 80, 6))
            pygame.draw.circle(self.canvas, (255, 255, 255), (x, y), max(self.W // 80, 6), 2)

            if show_labels:
                Label.render_text(self.canvas, str(node_id), (x, y), pygame.font.SysFont("arial", font_size - 5),
                                  (255, 255, 255), bold=True)

            if show_data:
                Label.render_text(self.canvas, str(round(self.genome.get_node(node_id).bias, 2)), (x, y - font_size - 5),
                                  pygame.font.SysFont("arial", font_size), (255, 255, 255), bold=True)

    def update_size(self, width, height):
        self.W = width
        self.H = height
        self.canvas = pygame.Surface((self.W, self.H))


class Main:

    def __init__(self, genomes, rows, cols):
        self.W, self.H = 1200, 800
        self.FPS = 60

        self.win = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("NEAT Renderer")

        self.canvas = pygame.Surface(self.win.get_size())

        self.event_handler = EventHandler()

        self.rows = rows
        self.cols = cols
        self.renderers = [Renderer(genome, 1200, 1000) for genome in genomes]

        self.running = False

    def run(self):
        self.running = True

        while self.running:

            self.clock.tick(self.FPS)
            pygame.display.update()

            self.win.fill((0, 0, 0))
            self.canvas.fill((0, 0, 0))

            for i, renderer in enumerate(self.renderers):
                renderer.update_size(self.win.get_width() // self.cols, self.win.get_height() // self.rows)
                renderer.render_network(True)
                self.canvas.blit(renderer.canvas, ((i - (i // self.cols) * self.cols) * renderer.W, (i // self.cols) * renderer.H))

            self.win.blit(self.canvas, (0, 0))

            if not self.event_handler.loop():
                self.running = False


if __name__ == "__main__":
    num_inputs = 2
    num_outputs = 1

    node_factory = NodeFactory(num_inputs + num_outputs)
    connection_factory = ConnectionFactory(num_inputs * num_outputs - 1)
    genome_factory = GenomeFactory()

    # genome1 = genome_factory.create_genome(num_inputs, num_outputs)
    # genome2 = genome_factory.create_genome(num_inputs, num_outputs)
    #
    # for i in range(3):
    #     genome1.mutate_add_connection(connection_factory)
    #     genome1.mutate_add_node(node_factory, connection_factory)
    #     genome2.mutate_add_connection(connection_factory)
    #     genome2.mutate_add_node(node_factory, connection_factory)
    #
    # genome1.show_info()
    # genome2.show_info()
    #
    # individual1 = Individual(genome1, 0)
    # individual2 = Individual(genome2, 0)
    #
    # offspring = individual1.crossover(genome_factory, individual2)
    #
    # renderer1 = Renderer(genome1, 1200, 1000)
    # renderer2 = Renderer(genome2, 1200, 1000)
    #
    # renderer3 = Renderer(offspring, 1200, 1000)
    #
    # m = Main()
    # m.run()
    #
    # nn1 = NeuralNetwork(genome1)
    # nn2 = NeuralNetwork(genome2)
    # nn3 = NeuralNetwork(offspring)
    # print(nn1.forward([1, 2, 3, 4]))
    # print(nn2.forward([1, 2, 3, 4]))
    # print(nn3.forward([1, 2, 3, 4]))

    population = Population(100, 2, 1, 100)

    genomes = [genome_factory.create_genome(num_inputs, num_outputs) for _ in range(1)]
    genome_deltas = {}

    for genome in genomes:
        for i in range(10):
            genome.mutate_add_connection(connection_factory)
            genome.mutate_add_node(node_factory, connection_factory)

    #for i in range(10):
    #    for j in range(i+1, 10):
    #        genome_deltas[(i, j)] = population.calculate_genome_distance(genomes[i], genomes[j])

    #print(genome_deltas)

    m = Main(genomes, 1, 1)
    m.run()

