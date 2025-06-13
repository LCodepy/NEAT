from neat.connection import ConnectionGene


class ConnectionFactory:

    def __init__(self, global_innovation_number: int) -> None:
        self.global_innovation_number = global_innovation_number
        self.generation_mutations = []

    def create_connection(self, input_node: int, output_node: int, weight: float) -> ConnectionGene:
        for mutation in self.generation_mutations:
            if mutation[0] == input_node and mutation[1] == output_node:
                innovation_number = mutation[2]
                break
        else:
            self.global_innovation_number += 1
            innovation_number = self.global_innovation_number

            self.generation_mutations.append((input_node, output_node, innovation_number))

        return ConnectionGene(input_node, output_node, weight, True, innovation_number)
