# --- Tokenizer

# TOKEN NAMES

reserved = {
    'NAME',
    'RANDOM',
    'RANDOM_PADDING',
    'LEVEL',

    'WEIGHT_EMBED'
    'EMBEDDING',

    'MULTIBIND',
    'MULTIBUNDLE',
    'PERMUTE',
    'NGRAM',

    'DEBUG',

    'TRUE',
    'FALSE',

    'SEQUENTIAL',
    'PARALLEL',
    'PARALLEL_MEMORY_EFFICIENT',
}


tokens = (
    'NAME',

    'RANDOM',
    'RANDOM_PADDING',
    'LEVEL',

    'WEIGHT_EMBED',
    'EMBEDDING',

    'MULTIBIND',
    'MULTIBUNDLE',
    'PERMUTE',
    'NGRAM',

    'DEBUG',
    'NUMBER',
    'ID',

    'ENCODING',
    'DIMENSIONS',
    'CLASSES',

    'LPAREN',
    'RPAREN',

    'SEMICOLON',
    'COLON',

    'TRUE',
    'FALSE',

    'TRAIN_SIZE',
    'TEST_SIZE',
    'INPUT_DIM',

    'VECTOR_SIZE',
    'NUM_THREADS',

    'TYPE',
    'SEQUENTIAL',
    'PARALLEL',
    'PARALLEL_MEMORY_EFFICIENT',
)

# REGULAR EXPRESSIONS ASSIGNED TO EACH TOKEN

# Ignored characters
t_ignore = ' \t'

# Token matching rules are written as regexs

t_NAME = r'.NAME'

t_EMBEDDING = r'.EMBEDDING'
t_RANDOM = r'RANDOM'
t_LEVEL = r'LEVEL'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_SEMICOLON = r'\;'
t_COLON = r'\,'
t_DIMENSIONS = r'.DIMENSIONS'
t_ENCODING = r'.ENCODING'
t_CLASSES = r'.CLASSES'
t_DEBUG = r'.DEBUG'
t_TRUE = r'TRUE'
t_NGRAM = r'NGRAM'
t_WEIGHT_EMBED = r'.WEIGHT_EMBED'
t_FALSE = r'FALSE'
t_TRAIN_SIZE = r'.TRAIN_SIZE'
t_TEST_SIZE = r'.TEST_SIZE'
t_INPUT_DIM = r'.INPUT_DIM'
t_VECTOR_SIZE = r'.VECTOR_SIZE'
t_NUM_THREADS = r'.NUM_THREADS'
t_TYPE = r'.TYPE'


def t_COMMENT(t):
    r'\//.*'
    r'\/*.*'
    r'\*/.*'
    pass

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    reserved_type = t.value in reserved
    if reserved_type:
        t.type = t.value
        return t
    return t


# A function can be used if there is an associated action.
# Write the matching regex in the docstring.
def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

# Ignored token with an action associated with it
def t_ignore_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count('\n')

# Error handler for illegal characters
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
