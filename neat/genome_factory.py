from neat.genome import Genome


class GenomeFactory:

    def __init__(self, connection_factory, node_factory):
        self.global_genome_number = 0
        self.connection_factory = connection_factory
        self.node_factory = node_factory

    def create_genome(self, num_inputs, num_outputs) -> Genome:
        self.global_genome_number += 1
        return Genome(self.global_genome_number, num_inputs, num_outputs, self.connection_factory, self.node_factory)
