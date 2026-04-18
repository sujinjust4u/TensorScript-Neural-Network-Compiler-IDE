from ir import IRInstruction

class Optimizer:
    def __init__(self, instructions):
        self.instructions = instructions

    def optimize(self):
        print("Running optimization passes...")
        instructions = self.constant_folding(self.instructions)
        instructions = self.dead_code_elimination(instructions)
        return instructions

    def constant_folding(self, instructions):
        optimized = []
        alias_map = {}
        for inst in instructions:
            # Re-map arguments if they were bypassed by constant folding
            new_args = [alias_map.get(arg, arg) for arg in inst.args]
            inst.args = new_args
            
            # Remove dropout layers with exactly 0.0 rate
            if inst.op == "dropout" and inst.kwargs.get("rate") == 0.0:
                print(f"  [Opt] Constant Folding: Removing non-functional '{inst}'")
                # Route any future uses of this destination back to its input
                alias_map[inst.dest] = inst.args[0]
            else:
                optimized.append(inst)
        return optimized

    def dead_code_elimination(self, instructions):
        # Backward pass to find used variables
        used_vars = set()
        
        # The last instruction's destination is considered the model output, so it's used
        if instructions:
            used_vars.add(instructions[-1].dest)

        for inst in reversed(instructions):
            if inst.dest in used_vars:
                for arg in inst.args:
                    if arg.startswith("%"):
                        used_vars.add(arg)
        
        optimized = []
        for inst in instructions:
            if inst.dest in used_vars or inst.op == "alloc_tensor":
                optimized.append(inst)
            else:
                print(f"  [Opt] Dead Code Elimination: Removed unused instruction '{inst}'")
        return optimized
