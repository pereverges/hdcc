import tokenizer
from validate import hdccAST
tokens = tokenizer.tokens
astDirective = hdccAST.astDirective
# AST CONSTRUCTION

# program :: directivelist

# directivelist :: /* empty */
#                | directive ; directivelist

# directive : NAME ID
#           | EMBEDDING embeddings
#           | DIMENSIONS NUMBER
#           | CLASSES NUMBER

# embeddings : value embedding_list

# embedding_list : embedding_param embedding_list
# embedding_list : _

# embedding_param : LPAREN ID embedding_type NUMBER RPAREN

# embedding_type : RANDOM
#                | LEVEL

# encoding : ENCODING expression
# expression : BIND LPAREN expression COLON expression RPAREN
#            | BUNDLE LPAREN expression COLON expression RPAREN
#            | MULTISET LPAREN expression RPAREN
#            | ID
#            | VALUE

def p_program_directive_list(p):
    'program : directive_list'
    p[0] = p[1]

def p_directive_list_none(p):
    'directive_list : '
    p[0] = []

def p_directive_list_some(p):
    'directive_list : directive SEMICOLON directive_list'
    p[0] = [p[1]] + p[3]

def p_name_directive(p):
    '''directive : NAME ID '''
    p[0] = astDirective(action='NAME', params=[p.lineno(1), p[2]])

def p_embedding_directive(p):
    '''directive : EMBEDDING embeddings '''
    p[0] = astDirective(action='EMBEDDING', params=[p.lineno(1), p[2]])

def p_debug(p):
    '''directive : DEBUG bool'''
    p[0] = astDirective(action='DEBUG', params=[p.lineno(1), p[2]])

def p_true(p):
    '''bool : TRUE'''
    p[0] = p[1]

def p_false(p):
    '''bool : FALSE'''
    p[0] = p[1]

def p_embeddings_list(p):
    '''embeddings : value_embedding
                  | value '''
    p[0] = p[1]

def p_value_and_embedding_list(p):
    'value_embedding : embedding_list value embedding_list'
    p[0] = p[1] + [p[2]] + p[3]

def p_embedding_list_none(p):
    'embedding_list : '
    p[0] = []

def p_embedding_list_some(p):
    '''embedding_list : embedding_param embedding_list'''
    p[0] = [p[1]] + p[2]
def p_embedding_param(p):
    '''embedding_param : LPAREN ID embedding_type NUMBER RPAREN'''
    p[0] = p.lineno(1), 'WEIGHT', p[2].upper(), p[3], p[4]


def p_value(p):
    '''value : LSBRAQ ID embedding_type NUMBER RSBRAQ'''
    p[0] = p.lineno(1), p[2].upper(), p[3], p[4]

def p_embedding_type(p):
    '''embedding_type : RANDOM
                      | LEVEL '''
    p[0] = p[1]

def p_dimensions_directive(p):
    '''directive : DIMENSIONS NUMBER '''
    p[0] = astDirective(action='DIMENSIONS', params=[p.lineno(1), p[2]])

def p_classes_directive(p):
    '''directive : CLASSES NUMBER '''
    p[0] = astDirective(action='CLASSES', params=[p.lineno(1), p[2]])

def p_encoding(p):
    'directive : ENCODING expression'
    p[0] = astDirective(action='ENCODING', params=[p.lineno(1), p[2]])

def p_var(p):
    ''' expression : ID'''
    p[0] = p[1].upper()

def p_bind(p):
    '''expression : BIND LPAREN expression COLON expression RPAREN'''
    p[0] = p.lineno(1), p[1], p[3], p[5]

def p_bundle(p):
    '''expression : BUNDLE LPAREN expression COLON expression RPAREN'''
    p[0] = p.lineno(1), p[1], p[3], p[5]

def p_multiset(p):
    '''expression : MULTISET LPAREN expression RPAREN'''
    p[0] = p.lineno(1), p[1], p[3]

def p_error(p):
    print(f'Syntax error at {p.value!r}')
