from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable


class NodeType(Enum):
    INPUT_NODE = auto()
    OUTPUT_NODE = auto()
    HIDDEN_NODE = auto()


@dataclass
class NodeGene:
    id: int
    type: NodeType
    bias: float
    activation: Callable

    def copy(self) -> NodeGene:
        return NodeGene(self.id, self.type, self.bias, self.activation)
