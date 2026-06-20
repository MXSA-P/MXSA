def f(): """doc"""; print(1)
import ast
print(ast.get_docstring(ast.parse(open('test_single_line_doc.py').read()).body[0]))
