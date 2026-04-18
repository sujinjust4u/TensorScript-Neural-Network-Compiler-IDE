from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass
class Node:
    pass

@dataclass
class InputNode(Node):
    shape: int

@dataclass
class LayerNode(Node):
    pass

@dataclass
class DenseLayerNode(LayerNode):
    units: int
    activation: str

@dataclass
class DropoutLayerNode(LayerNode):
    rate: float

@dataclass
class LossNode(Node):
    name: str

@dataclass
class OptimizerNode(Node):
    name: str
    kwargs: dict

@dataclass
class TrainNode(Node):
    kwargs: dict

@dataclass
class ModelNode(Node):
    name: str
    input_node: Optional[InputNode] = None
    layers: List[LayerNode] = None
    loss: Optional[LossNode] = None
    optimizer: Optional[OptimizerNode] = None
    train_params: Optional[TrainNode] = None

    def __post_init__(self):
        if self.layers is None:
            self.layers = []
