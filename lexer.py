import re

class Token:
    def __init__(self, type, value, line, column):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f'{self.type}({self.value}) at {self.line}:{self.column}'

def tokenize(code):
    keywords = {
        'model', 'input', 'shape', 'layer', 'dense', 'relu', 'dropout', 
        'softmax', 'loss', 'categorical_crossentropy', 'optimizer', 'adam', 
        'lr', 'train', 'epochs', 'batch'
    }
    
    token_specification = [
        ('NUMBER',   r'\d+(\.\d*)?'),
        ('ID',       r'[A-Za-z_][A-Za-z0-9_]*'),
        ('LBRACE',   r'\{'),
        ('RBRACE',   r'\}'),
        ('LPAREN',   r'\('),
        ('RPAREN',   r'\)'),
        ('EQUALS',   r'='),
        ('COMMA',    r','),
        ('NEWLINE',  r'\n'),
        ('SKIP',     r'[ \t]+'),
        ('MISMATCH', r'.'),
    ]
    
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    line_num = 1
    line_start = 0
    tokens = []
    
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        
        if kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)
        elif kind == 'ID' and value in keywords:
            kind = value.upper()
        elif kind == 'NEWLINE':
            line_start = mo.end()
            line_num += 1
            continue
        elif kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected on line {line_num}')
            
        tokens.append(Token(kind, value, line_num, column))
        
    tokens.append(Token('EOF', '', line_num, 0))
    return tokens
