import ply.lex as lex
import ply.yacc as yacc
import tokenizer
import grammar
from ir import IntermediateRepresentation
from validate import hdccAST
import sys

print(sys.argv[1])
astDirective = hdccAST.astDirective

# Build the lexer
lexer = lex.lex(module=tokenizer)

parser = yacc.yacc(module=grammar)

f = open(sys.argv[1], 'r')
data = f.read()
f.close()

ast = hdccAST()

for x in parser.parse(data):
    ast.validateDirective(x)
ast.validateRequiredArgs()
ast.validateVarsDeclaration()
ast.validateVarsUsage()
ast.print_parsed_and_validated_input()

name, classes, dimensions, used_vars, input, encoding, embeddings, debug, encoding_fun, encoding_fun_call = ast.get_ast_obj()

ir = IntermediateRepresentation(name, classes, dimensions, used_vars, input, encoding, embeddings, debug, encoding_fun, encoding_fun_call)
ir.run()
