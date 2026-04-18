from ast_nodes import *

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current(self):
        return self.tokens[self.pos]

    def consume(self, expected_type):
        if self.current().type == expected_type:
            token = self.current()
            self.pos += 1
            return token
        else:
            raise SyntaxError(f"Expected {expected_type}, got {self.current().type} at line {self.current().line}")

    def parse(self):
        return self.parse_model()

    def parse_model(self):
        self.consume('MODEL')
        name = self.consume('ID').value
        self.consume('LBRACE')
        
        input_node = None
        layers = []
        loss = None
        optimizer = None
        train_params = None

        while self.current().type != 'RBRACE' and self.current().type != 'EOF':
            token_type = self.current().type
            if token_type == 'INPUT':
                input_node = self.parse_input()
            elif token_type == 'LAYER':
                layers.append(self.parse_layer())
            elif token_type == 'LOSS':
                loss = self.parse_loss()
            elif token_type == 'OPTIMIZER':
                optimizer = self.parse_optimizer()
            elif token_type == 'TRAIN':
                train_params = self.parse_train()
            else:
                raise SyntaxError(f"Unexpected token {self.current()} inside model")
        
        self.consume('RBRACE')
        return ModelNode(name, input_node, layers, loss, optimizer, train_params)

    def parse_input(self):
        self.consume('INPUT')
        self.consume('SHAPE')
        self.consume('LPAREN')
        shape_val = self.consume('NUMBER').value
        self.consume('RPAREN')
        return InputNode(int(shape_val))

    def parse_layer(self):
        self.consume('LAYER')
        layer_type = self.current().type
        if layer_type == 'DENSE':
            self.consume('DENSE')
            self.consume('LPAREN')
            units = self.consume('NUMBER').value
            self.consume('COMMA')
            activation = self.current().value
            if self.current().type in ['RELU', 'SOFTMAX']:
                self.consume(self.current().type)
            elif self.current().type == 'ID':
                self.consume('ID')
            else:
                raise SyntaxError(f"Expected activation function, got {self.current().type}")
            self.consume('RPAREN')
            return DenseLayerNode(int(units), activation)
        elif layer_type == 'DROPOUT':
            self.consume('DROPOUT')
            self.consume('LPAREN')
            rate = self.consume('NUMBER').value
            self.consume('RPAREN')
            return DropoutLayerNode(float(rate))
        else:
            raise SyntaxError(f"Unknown layer type {self.current()}")

    def parse_loss(self):
        self.consume('LOSS')
        loss_name = self.current().value
        if self.current().type == 'CATEGORICAL_CROSSENTROPY':
            self.consume('CATEGORICAL_CROSSENTROPY')
        else:
            self.consume('ID')
        return LossNode(loss_name)

    def parse_optimizer(self):
        self.consume('OPTIMIZER')
        opt_name = self.current().value
        if self.current().type == 'ADAM':
            self.consume('ADAM')
        else:
            self.consume('ID')
        
        kwargs = {}
        if self.current().type == 'LPAREN':
            self.consume('LPAREN')
            while self.current().type != 'RPAREN':
                key = self.current().value
                if self.current().type == 'LR':
                    self.consume('LR')
                else:
                    self.consume('ID')
                self.consume('EQUALS')
                val = self.consume('NUMBER').value
                kwargs[key] = float(val)
                if self.current().type == 'COMMA':
                    self.consume('COMMA')
            self.consume('RPAREN')
        return OptimizerNode(opt_name, kwargs)

    def parse_train(self):
        self.consume('TRAIN')
        kwargs = {}
        while self.current().type in ['EPOCHS', 'BATCH', 'ID']:
            key = self.current().value
            self.consume(self.current().type)
            self.consume('EQUALS')
            val = self.consume('NUMBER').value
            kwargs[key] = int(val)
            if self.current().type == 'COMMA':
                self.consume('COMMA')
            else:
                break
        return TrainNode(kwargs)
