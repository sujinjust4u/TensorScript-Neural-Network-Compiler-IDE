from ast_nodes import *
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class IRInstruction:
    dest: str
    op: str
    args: List[str]
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        args_str = ", ".join(self.args)
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        combined = ", ".join(filter(None, [args_str, kwargs_str]))
        if self.dest:
            return f"{self.dest} = {self.op} {combined}"
        return f"{self.op} {combined}"

class IRGenerator:
    def __init__(self, model_node):
        self.model = model_node
        self.instructions = []
        self.var_count = 0

    def new_var(self):
        v = f"%{self.var_count}"
        self.var_count += 1
        return v

    def generate(self):
        # 1. alloc input tensor
        batch_var = "batch" 
        input_dim = self.model.input_node.shape
        dest = self.new_var()
        self.instructions.append(IRInstruction(dest, "alloc_tensor", [batch_var, str(input_dim)]))
        
        current_input = dest

        # Emit an unused dummy instruction to test Dead Code Elimination
        dummy_dest = self.new_var()
        self.instructions.append(IRInstruction(dummy_dest, "dummy_op", [current_input]))
        
        for layer in self.model.layers:
            if isinstance(layer, DenseLayerNode):
                dest = self.new_var()
                self.instructions.append(IRInstruction(dest, "dense", [current_input], {"units": layer.units}))
                current_input = dest
                
                if layer.activation:
                    dest = self.new_var()
                    self.instructions.append(IRInstruction(dest, layer.activation, [current_input]))
                    current_input = dest
            
            elif isinstance(layer, DropoutLayerNode):
                dest = self.new_var()
                self.instructions.append(IRInstruction(dest, "dropout", [current_input], {"rate": layer.rate}))
                current_input = dest

        return self.instructions
