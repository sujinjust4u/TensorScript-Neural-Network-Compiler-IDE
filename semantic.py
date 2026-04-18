from ast_nodes import *

class SemanticAnalyzer:
    def __init__(self, model_node):
        self.model = model_node

    def analyze(self):
        if not self.model.input_node:
            raise ValueError("Model must have an input node with a specified shape.")
        if not self.model.layers:
            raise ValueError("Model must have at least one layer.")
        if not self.model.loss:
            raise ValueError("Model must have a loss function specified.")
        
        last_layer = self.model.layers[-1]

        # Verify output layer activation vs loss
        if self.model.loss.name == 'categorical_crossentropy':
            if not isinstance(last_layer, DenseLayerNode) or last_layer.activation != 'softmax':
                raise ValueError("Categorical crossentropy loss requires a final Dense layer with softmax activation.")

        # Validate layers
        for layer in self.model.layers:
            if isinstance(layer, DenseLayerNode):
                if layer.units <= 0:
                    raise ValueError(f"Dense layer units must be > 0. Got {layer.units}")
                if layer.activation not in ['relu', 'softmax', 'sigmoid', 'tanh']:
                    raise ValueError(f"Unsupported activation: {layer.activation}")
            elif isinstance(layer, DropoutLayerNode):
                if not (0.0 <= layer.rate < 1.0):
                    raise ValueError(f"Dropout rate must be between 0 and 1. Got {layer.rate}")

        if self.model.train_params:
            if 'epochs' not in self.model.train_params.kwargs:
                raise ValueError("Training parameters must include 'epochs'.")
            if 'batch' not in self.model.train_params.kwargs:
                raise ValueError("Training parameters must include 'batch'.")

        print("Semantic analysis passed successfully.")
