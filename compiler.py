import sys
from lexer import tokenize
from parser import Parser
from semantic import SemanticAnalyzer
from ir import IRGenerator
from optimizer import Optimizer
from codegen import PyTorchCodeGenerator

def compile_ts(source_file, target_file):
    print(f"Reading {source_file}...")
    with open(source_file, 'r') as f:
        code = f.read()

    print("\n[Phase 1] Tokenizing...")
    tokens = tokenize(code)

    print("\n[Phase 2] Parsing to AST...")
    parser = Parser(tokens)
    ast = parser.parse()

    print("\n[Phase 3] Semantic Analysis...")
    analyzer = SemanticAnalyzer(ast)
    analyzer.analyze()

    print("\n[Phase 4] IR Generation...")
    ir_gen = IRGenerator(ast)
    ir = ir_gen.generate()
    for inst in ir:
        print(f"  {inst}")

    print("\n[Phase 5] Optimization...")
    opt = Optimizer(ir)
    optimized_ir = opt.optimize()
    print("  Optimized IR:")
    for inst in optimized_ir:
        print(f"  {inst}")

    print("\n[Phase 6] Code Generation...")
    codegen = PyTorchCodeGenerator(ast, optimized_ir)
    codegen.generate(target_file)

    print(f"\nCompilation Successful! Artifact saved to {target_file}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compiler.py <source.ts> <target.py>")
        sys.exit(1)
    compile_ts(sys.argv[1], sys.argv[2])
