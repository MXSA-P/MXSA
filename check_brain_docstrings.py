import ast

with open('simba/core/brain.py', 'r') as f:
    source = f.read()

tree = ast.parse(source)

for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'SimbaBrain':
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(child) is None:
                    print(f"{child.name} at line {child.lineno} lacks docstring.")
