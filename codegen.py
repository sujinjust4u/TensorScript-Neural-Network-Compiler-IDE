from ast_nodes import ModelNode

class PyTorchCodeGenerator:
    def __init__(self, model_node, instructions):
        self.model = model_node
        self.instructions = instructions

    def generate(self, output_file):
        code = self._build_code()
        with open(output_file, 'w') as f:
            f.write(code)
        print(f"PyTorch code emitted to {output_file}")

    def _build_code(self):
        imports = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.optim as optim\n\n"
        )

        class_def = f"class {self.model.name}(nn.Module):\n"
        class_def += "    def __init__(self):\n"
        class_def += "        super().__init__()\n"
        
        layer_defs = ""
        forward_pass = "    def forward(self, x):\n"
        
        def clean_var(v):
            return v.replace('%', 'v_') if isinstance(v, str) and v.startswith('%') else v

        for inst in self.instructions:
            dest_var = clean_var(inst.dest)
            
            if inst.op == "alloc_tensor":
                forward_pass += f"        {dest_var} = x\n"
                continue
            
            if inst.op == "dummy_op":
                continue 

            if inst.op == "dense":
                in_var = clean_var(inst.args[0])
                units = inst.kwargs["units"]
                layer_defs += f"        self.l_{inst.dest.strip('%')} = nn.LazyLinear({units})\n"
                forward_pass += f"        {dest_var} = self.l_{inst.dest.strip('%')}({in_var})\n"
            
            elif inst.op in ["relu", "softmax", "sigmoid", "tanh"]:
                in_var = clean_var(inst.args[0])
                dim = "dim=1" if inst.op == "softmax" else ""
                func_map = {"relu": "torch.relu", "softmax": "torch.softmax", "sigmoid": "torch.sigmoid", "tanh": "torch.tanh"}
                func = func_map[inst.op]
                args = f"{in_var}"
                if inst.op == "softmax": args += f", {dim}"
                forward_pass += f"        {dest_var} = {func}({args})\n"

            elif inst.op == "dropout":
                in_var = clean_var(inst.args[0])
                rate = inst.kwargs["rate"]
                layer_defs += f"        self.d_{inst.dest.strip('%')} = nn.Dropout({rate})\n"
                forward_pass += f"        {dest_var} = self.d_{inst.dest.strip('%')}({in_var})\n"

        output_var = clean_var(self.instructions[-1].dest) if self.instructions else "x"
        forward_pass += f"        return {output_var}\n\n"

        loss_map = {"categorical_crossentropy": "nn.CrossEntropyLoss()"}
        loss_fn = loss_map.get(self.model.loss.name, "nn.MSELoss()")

        epochs = self.model.train_params.kwargs.get("epochs", 10)
        batch_size = self.model.train_params.kwargs.get("batch", 32)
        lr = self.model.optimizer.kwargs.get("lr", 0.001)

        training_code = f"""
if __name__ == '__main__':
    print("Initializing {self.model.name}...")
    model = {self.model.name}()
    
    criterion = {loss_fn}
    optimizer = optim.{self.model.optimizer.name.capitalize()}(model.parameters(), lr={lr})

    print("Generating simulated dataloader...")
    input_shape = {self.model.input_node.shape}
    X = torch.randn(100 * {batch_size}, input_shape)
    y = torch.randint(0, 10, (100 * {batch_size},))

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size={batch_size})

    print("Starting Training Loop...")
    for epoch in range({epochs}):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch [{{epoch+1}}/{epochs}], Loss: {{epoch_loss/len(dataloader):.4f}}")
    print("Training Complete!")
"""
        return imports + class_def + layer_defs + "\n" + forward_pass + training_code
