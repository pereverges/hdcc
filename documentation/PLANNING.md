# STEPS ON CREATING THE COMPILER #

1. LEXER AND PARSER
2. IR
3. FISM
4. RETARGATABLE BACKEND
5. EXPERIMENTS AND VALIDATION

## LEXER AND PARSER ##

For this step I have looked at some parsing and lexing tools:

- PLY - Python Lex-Yacc https://github.com/dabeaz/ply 
- TDParser: https://tdparser.readthedocs.io/en/latest/

There are more, but I think that PLY probably is the best way to go.

##  INTERMEDIATE REPRESENTATION ##

Generate the intermediate representation that will correspond to the directives used in the language specification.

## Fictitious Instruction Set Machine FISM ##

Generate the set of instructions needed.

## RETARGATABLE BACKEND ##
 
Generate ANSI C code?

## EXPERIMENTS AND VALIDATION ##

Try different applications such as:

VoiceHD (ISOLET)
MNIST

Comparing against:

TorchHD
OpenHD