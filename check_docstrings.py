import ast

with open('simba/core/brain.py', 'r') as f:
    source = f.read()

tree = ast.parse(source)

for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if ast.get_docstring(node) is None:
            print(f"{node.__class__.__name__} {node.name} at line {node.lineno} lacks docstring.")

