from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConnectionGene:
    input_node: int
    output_node: int
    weight: float
    enabled: bool
    innovation_number: int

    def copy(self) -> ConnectionGene:
        return ConnectionGene(self.input_node, self.output_node, self.weight, self.enabled, self.innovation_number)
