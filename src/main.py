import ply.lex as lex
import ply.yacc as yacc
import tokenizer
import grammar
from ir import IntermediateRepresentation
from validate import hdccAST
import sys

debug = False
astDirective = hdccAST.astDirective

# Build the lexer
lexer = lex.lex(module=tokenizer)

parser = yacc.yacc(module=grammar)

f = open(sys.argv[1], 'r')
path = sys.argv[1][:sys.argv[1].rfind('/')+1]
data = f.read()
f.close()

ast = hdccAST()

for x in parser.parse(data):
    if debug:
        print(x)
    ast.validateDirective(x)
ast.validateRequiredArgs()
ast.validateVarsDeclaration()
ast.validateVarsUsage()
if debug:
    print()
    ast.print_parsed_and_validated_input()

name, classes, dimensions, used_vars, weight, encoding, embeddings, debug, encoding_fun, train_size, test_size, \
num_threads, vector_size, type, input_dim, high, basic, padding, permutes, ngram, not_multiset = ast.get_ast_obj()

ir = IntermediateRepresentation(name, classes, dimensions, used_vars, weight, encoding, embeddings, debug,
                                encoding_fun, train_size, test_size, num_threads, vector_size, type, input_dim,
                                high, basic, padding, permutes, ngram, path, not_multiset)

ir.run()
