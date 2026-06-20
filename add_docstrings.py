import ast
import re

with open('simba/core/brain.py', 'r') as f:
    lines = f.readlines()

source = "".join(lines)
tree = ast.parse(source)

missing_funcs = []
missing_classes = []

for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if ast.get_docstring(node) is None:
            missing_funcs.append(node)
    elif isinstance(node, ast.ClassDef):
        if ast.get_docstring(node) is None:
            missing_classes.append(node)

# We must process from bottom to top so that line insertions don't affect earlier line numbers!
# But wait, node.body[0].lineno is the line in the original file. 
# We can collect all insertions and then apply them.

insertions = [] # list of (line_idx, text_to_insert)
replacements = {} # list of (line_idx, new_line_text)

for node in missing_classes:
    doc = f'mock class for {node.name}.' if 'Mock' in node.name else f'{node.name} class.'
    body_start = node.body[0].lineno - 1
    # Find the indent of the body
    body_line = lines[body_start]
    indent_match = re.match(r'^(\s*)', body_line)
    indent = indent_match.group(1) if indent_match else ''
    if node.body[0].lineno == node.lineno:
        # One liner class? Rare but possible.
        pass
    else:
        insertions.append((body_start, f'{indent}"""{doc}"""\n'))

for node in missing_funcs:
    name = node.name
    doc = f'{name}.'
    if name == '__init__':
        doc = 'initialize.'
    elif name.startswith('_handle_'):
        doc = f"handle {name.replace('_handle_', '')} command."
        
    body_start = node.body[0].lineno - 1
    
    if node.body[0].lineno == node.lineno:
        # single line function
        line = lines[body_start]
        # find the first colon
        # to be safe, split by ':' but we must be careful about colons in type hints
        # ast node has end_lineno and col_offset?
        # Actually, python 3.8+ has node.body[0].col_offset which tells us where the body starts!
        body_col = node.body[0].col_offset
        # Split the line at body_col
        prefix = line[:body_col]
        suffix = line[body_col:]
        # Replace the line
        # Instead of replacing on the same line, let's make it multi-line or just insert docstring and keep on same line.
        # Format: def foo(): """doc"""; pass
        # But wait, PEP8 doesn't like `;`. Let's just make it multiline!
        indent_match = re.match(r'^(\s*)', line)
        indent = indent_match.group(1) if indent_match else ''
        body_indent = indent + '    '
        
        # We replace the line with prefix + '\n' + body_indent + '"""' + doc + '"""\n' + body_indent + suffix
        # But wait, what if prefix doesn't end with whitespace? It ends with `: ` usually.
        prefix = prefix.rstrip()
        suffix = suffix.lstrip()
        new_line = f'{prefix}\n{body_indent}"""{doc}"""\n{body_indent}{suffix}\n'
        if not suffix.endswith('\n'):
            new_line += '\n'
        replacements[body_start] = new_line
    else:
        # multi line function
        body_line = lines[body_start]
        indent_match = re.match(r'^(\s*)', body_line)
        indent = indent_match.group(1) if indent_match else ''
        insertions.append((body_start, f'{indent}"""{doc}"""\n'))

# Apply replacements and insertions from bottom to top
changes = []
for idx in replacements:
    changes.append((idx, 'replace', replacements[idx]))
for idx, text in insertions:
    changes.append((idx, 'insert', text))

changes.sort(key=lambda x: x[0], reverse=True)

for change in changes:
    idx = change[0]
    action = change[1]
    text = change[2]
    if action == 'replace':
        lines[idx] = text
    elif action == 'insert':
        lines.insert(idx, text)

with open('simba/core/brain.py', 'w') as f:
    f.writelines(lines)

