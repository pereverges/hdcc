from execution_type import Types

class hdccAST:
    def __init__(self):
        self.set_arguments = set()
        self.declared_vars = set()
        self.used_vars = set()
        self.weight = None
        self.input_dim = None
        self.encoding = None
        self.encoding_fun = None
        self.embeddings = []
        self.name = None
        self.classes = None
        self.dimensions = None
        self.wait = None
        self.debug = False
        self.simple = False

        self.permutes = []

        self.train_size = None
        self.test_size = None
        self.num_threads = None
        self.type = None
        self.vector_size = 128

        self.high = 0
        self.padding = None

        self.vectorial = None
        self.performance = None
        self.basic = True
        self.multiset = False
        self.multibind = False
        self.ngram = None
        self.required_arguments = {'NAME', 'ENCODING', 'CLASSES', 'DIMENSIONS', 'TEST_SIZE', 'TRAIN_SIZE', 'TYPE', 'INPUT_DIM'}

    def get_ast_obj(self):
        return self.name, self.classes, self.dimensions, self.used_vars, self.weight, self.encoding, self.embeddings, self.debug, \
               self.encoding_fun, self.train_size, self.test_size, self.num_threads, self.vector_size, self.type,\
               self.input_dim, self.high, self.basic, self.padding, self.permutes, self.ngram, self.multiset, self.vectorial, self.performance

    def validateDirective(self, x):
        action, params = self.astDirective.resolve(x)

        if action not in self.set_arguments:
            self.set_arguments.add(action)
        else:
            self.error_repeated_directive(action, params[0])
        if action == 'ENCODING':
            if self.dimensions is not None and self.type is not None and self.vectorial is not None and self.performance is not None:
                self.wait = False
                enc = ''
                _, self.encoding, _, _ = self.unpack_encoding(params[1], enc)
                self.encoding_build(self.encoding, self.encoding)
            else:
                self.wait = params[1]
        if action == 'EMBEDDING':
            self.set_declared_vars(params)
        if action == 'WEIGHT_EMBED':
            self.declared_vars.add(params[1])
            self.weight = params[1]
            if isinstance(params[2], list):
                if params[2][0] == 'LEVEL':
                    self.embeddings.append([params[2][0], params[1], params[2][1]])
                else:
                    self.embeddings.append([params[2][0], params[1], params[2][1]])
                    # TODO CHANGE EMBEDDING TO OBJECT AND STORE THIS VARS
                    self.padding = params[3]
                self.high = params[3]
            else:
                self.embeddings.append([params[2], params[1], params[3]])
        if action == 'NAME':
            self.name = params[1].lower()
        if action == 'CLASSES':
            self.classes = params[1]
            if self.high == 0:
                self.high = params[1]
        if action == 'DEBUG':
            self.debug = True
        if action == 'DIMENSIONS':
            self.dimensions = params[1]
            if self.wait is not None and self.type is not None and self.vectorial is not None and self.performance is not None:
                enc = ''
                _, self.encoding, _, _ = self.unpack_encoding(self.wait, enc)
                self.encoding_build(self.encoding, self.encoding)
                self.wait = None
        if action == 'TRAIN_SIZE':
            self.train_size = params[1]
        if action == 'TEST_SIZE':
            self.test_size = params[1]
        if action == 'NUM_THREADS':
            self.num_threads = params[1]
            if self.wait is not None and self.vectorial is not None and self.performance is not None:
                enc = ''
                _, self.encoding, _, _ = self.unpack_encoding(self.wait, enc)
                self.encoding_build(self.encoding, self.encoding)
                self.wait = None
        if action == 'VECTOR_SIZE':
            self.vector_size = params[1]
        if action == 'TYPE':
            if params[1] == 'SEQUENTIAL':
                self.type = Types.SEQUENTIAL
            elif params[1] == 'PARALLEL':
                self.type = Types.PARALLEL
        if action == 'INPUT_DIM':
            self.input_dim = params[1]
        if action == 'MODE':
            self.basic = params[1]
        if action == 'SIMPLE':
            if params[1] == 'TRUE':
                self.simple = True
            else:
                self.simple = False
        if action == 'PERFORMANCE':
            if params[1] == 'TRUE':
                self.performance = True
            else:
                self.performance = False
            if self.wait is not None and self.vectorial is not None:
                enc = ''
                if self.simple:
                    _, self.encoding, _, _ = self.unpack_encoding_simple(self.wait, enc)
                else:
                    _, self.encoding, _, _ = self.unpack_encoding(self.wait, enc)
                self.encoding_build(self.encoding, self.encoding)
                self.wait = None
        if action == 'VECTORIAL':
            if params[1] == 'TRUE':
                self.vectorial = True
            else:
                self.vectorial = False
            if self.wait is not None and self.performance is not None:
                enc = ''
                if self.simple:
                    _, self.encoding, _, _ = self.unpack_encoding_simple(self.wait, enc)
                else:
                    _, self.encoding, _, _ = self.unpack_encoding(self.wait, enc)
                self.encoding_build(self.encoding, self.encoding)
                self.wait = None

    def unpack_encoding(self, i, enc):
        if self.basic:
            if i[1] == 'MULTIBIND':
                self.multibind = True
                if i[3] == self.weight:
                    b1, enc1, _, _ = self.unpack_encoding(i[2],enc)
                    b2, enc2, _, _ = self.unpack_encoding(i[3],enc)
                    enc += enc1 + enc2
                    enc += '\n    enc = multibind_forward(' + b1 + ',' + b2 + ', indices, enc' + ');'
                    return 'enc', enc, 'bind_forward', [b1,b2]
                elif i[2] == self.weight:
                    b1, enc1, _, _ = self.unpack_encoding(i[2], enc)
                    b2, enc2, _, _ = self.unpack_encoding(i[3], enc)
                    enc += enc1 + enc2
                    enc += '\n    enc = multibind_forward(' + b2 + ',' + b1 + ', indices, enc' + ');'
                    return 'enc', enc, 'bind_forward', [b2, b1]
                else:
                    b1, enc1, _, _ = self.unpack_encoding(i[2],enc)
                    b2, enc2, _, _ = self.unpack_encoding(i[3],enc)
                    enc += enc1 + enc2
                    enc += '\n    enc = multibind(' + b1 + ',' + b2 + ', enc);'
                    return 'enc', enc, 'bind', [b1,b2]
            elif i[1] == 'BUNDLE':
                return '    bundle(' + self.unpack_encoding(i[1][2]) + ')', '', 'bundle'
            elif i[1] == 'PERMUTE':
                b, enc_aux, _, _ = self.unpack_encoding(i[2], enc)
                enc += enc_aux
                self.permutes += [i[3]]
                if self.vectorial:
                    enc += '\n    enc = permute'+str(i[3])+'(' + b + ',' + str(i[3]) + ',0,1);'
                else:
                    enc += '\n    permute(' + b + ',' + str(i[3]) + ',0,1, enc);'
                return 'enc', enc, '', ''
            elif i[1] == 'NGRAM':
                b, enc_aux, _, _ = self.unpack_encoding(i[2], enc)
                enc += enc_aux
                # THIS IN NGRAM FORWARD, CREATE NGRAM NORMAL
                if self.ngram != None:
                    if self.ngram < i[3]:
                        self.ngram = i[3]
                else:
                    self.ngram = i[3]
                if i[2] == self.weight:
                    self.multiset = True
                    enc += '\n    enc = ngram_forward(' + i[2] + ',indices,enc,' + str(i[3]) + ');'
                else:
                    enc += '\n    enc = ngram(' + b + ',enc,' + str(i[3]) + ');'

                return 'enc', enc, '', ''
            elif i[1] == 'MULTIBUNDLE':
                self.multiset = True
                b, enc_aux, fun, varaibles = self.unpack_encoding(i[2], enc)
                if fun == 'bind':
                    enc = enc_aux[:enc_aux.rfind('\n')]
                    enc += '\n    enc = multiset_multibind(' + varaibles[0] + ',' + varaibles[1] + ', enc);'
                elif fun == 'bind_forward':
                    enc = enc_aux[:enc_aux.rfind('\n')]
                    enc += '\n    enc = multiset_multibind_forward(' + varaibles[0] + ',' + varaibles[1] + ', indices, enc' + ');'
                else:
                    enc += enc_aux
                    enc += '    enc = multiset(' + b + ');\n'
                return 'enc', enc, '', ''
            else:
                self.set_used_var(i)
                if i == self.weight:
                    return self.weight, enc, '', ''
                return i.upper(), enc, '', ''

    def unpack_encoding_simple(self, i, enc):
        if self.basic:
            if i[1] == 'MULTIBIND':
                self.multibind = True
                if i[3] == self.weight:
                    b1, enc1, _, _ = self.unpack_encoding_simple(i[2],enc)
                    b2, enc2, _, _ = self.unpack_encoding_simple(i[3],enc)
                    enc += enc1 + enc2
                    enc += '\n    enc = multibind_forward(' + b1 + ',' + b2 + ', indices, enc' + ');'
                    return 'enc', enc, 'bind_forward', [b1,b2]
                elif i[2] == self.weight:
                    b1, enc1, _, _ = self.unpack_encoding_simple(i[2], enc)
                    b2, enc2, _, _ = self.unpack_encoding_simple(i[3], enc)
                    enc += enc1 + enc2
                    enc += '\n    enc = multibind_forward(' + b2 + ',' + b1 + ', indices, enc' + ');'
                    return 'enc', enc, 'bind_forward', [b2, b1]
                else:
                    b1, enc1, _, _ = self.unpack_encoding_simple(i[2],enc)
                    b2, enc2, _, _ = self.unpack_encoding_simple(i[3],enc)
                    enc += enc1 + enc2
                    enc += '\n    enc = multibind(' + b1 + ',' + b2 + ', enc);'
                    return 'enc', enc, 'bind', [b1,b2]
            elif i[1] == 'BUNDLE':
                return '    bundle(' + self.unpack_encoding_simple(i[1][2]) + ')', '', 'bundle'
            elif i[1] == 'PERMUTE':
                b, enc_aux, _, _ = self.unpack_encoding_simple(i[2], enc)
                enc += enc_aux
                self.permutes += [i[3]]
                if self.vectorial:
                    enc += '\n    enc = permute'+str(i[3])+'(' + b + ',' + str(i[3]) + ',0,1);'
                else:
                    enc += '\n    permute(' + b + ',' + str(i[3]) + ',0,1, enc);'
                return 'enc', enc, '', ''
            elif i[1] == 'NGRAM':
                b, enc_aux, _, _ = self.unpack_encoding_simple(i[2], enc)
                enc += enc_aux
                # THIS IN NGRAM FORWARD, CREATE NGRAM NORMAL
                if self.ngram != None:
                    if self.ngram < i[3]:
                        self.ngram = i[3]
                else:
                    self.ngram = i[3]
                if i[2] == self.weight:
                    self.multiset = True
                    enc += '\n    enc = ngram_forward(' + i[2] + ',indices,enc,' + str(i[3]) + ');'
                else:
                    enc += '\n    enc = ngram(' + b + ',enc,' + str(i[3]) + ');'

                return 'enc', enc, '', ''
            elif i[1] == 'MULTIBUNDLE':
                self.multiset = True
                b, enc_aux, fun, varaibles = self.unpack_encoding_simple(i[2], enc)
                enc += enc_aux
                enc += '\n    enc = multiset(' + b + ');\n'
                return 'enc', enc, '', ''
            else:
                self.set_used_var(i)
                if i == self.weight:
                    return self.weight, enc, '', ''
                return i.upper(), enc, '', ''

    def set_used_var(self, var):
        self.used_vars.add(var)

    def set_declared_vars(self,params):
        if type(params[1]) == list:
            for i in params[1]:
                if i[1] == 'WEIGHT':
                    self.declared_vars.add(i[2])
                    self.weight = i[2]
                    if isinstance(i[3], list):
                        self.embeddings.append([i[3][0], i[2], i[3][1]])
                        self.high = i[4]
                    else:
                        self.embeddings.append([i[3], i[2], i[4]])
                else:
                    self.declared_vars.add(i[1])
                    if isinstance(i[2], list):
                        self.high = i[3]
                        self.embeddings.append([i[2][0], i[1], i[2][1], i[3]])
                    else:
                        self.embeddings.append([i[2], i[1], i[3]])
        else:
            self.declared_vars.add(params[1][2])

    def encoding_build(self, var, t):
        if self.type == Types.SEQUENTIAL:
            self.encoding_build_sequential(var, t)
        elif self.type == Types.PARALLEL:
            self.encoding_build_parallel(var)

    def encoding_build_sequential(self,var,t):
        if self.simple:
            space = 'DIMENSIONS*INPUT_DIM'
        else:
            space = 'DIMENSIONS'
            if self.multiset == False:
                space += '*INPUT_DIM'
        fun_head_train = '''
void encode_train_task(void* task){'''

        fun_head_test = '''
void encode_test_task(void* task){'''

        if self.vectorial:
            enc = '''f4si * enc = calloc(''' + space + ''', sizeof(int));'''
        else:
            enc = '''float * enc = calloc(''' + space + ''', sizeof(int));'''
        self.encoding_fun = fun_head_train + '''
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,''' + self.weight + '''_DIM-1, indices);
    ''' + enc + '''
    ''' + var + '''
    hard_quantize((float*)enc,1);
    update_weight((float*)enc,label);
    free(enc);
    free(indices);
    free(data);
}

        ''' + fun_head_test + '''
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,''' + self.weight + '''_DIM-1, indices);
    ''' + enc + '''
    ''' + var + '''
    float *l = linear((float*)enc);
    if(argmax(l) == label){
        free(l);
        update_correct_predictions();
    }
    free(indices);
    free(data);
    free(enc);
}
                                    '''

    def encoding_build_parallel(self, var):
        if self.simple:
            space = 'DIMENSIONS*INPUT_DIM'
        else:
            space = 'DIMENSIONS'
            if self.multiset == False:
                space += '*INPUT_DIM'
        fun_head_train = '''
void encode_train_task(void* task){'''

        fun_head_test = '''
void encode_test_task(void* task){'''

        if self.vectorial:
            enc = '''f4si * enc = calloc(''' + space + ''', sizeof(int));'''
        else:
            enc = '''float * enc = calloc(''' + space + ''', sizeof(int));'''

        self.encoding_fun = fun_head_train + '''
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,''' + self.weight + '''_DIM-1, indices);
    ''' + enc + '''
    ''' + var + '''
    hard_quantize((float*)enc,1);
    update_weight((float*)enc,label);
    free(enc);
    free(indices);
    free(data);
}

''' + fun_head_test + '''
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,''' + self.weight + '''_DIM-1, indices);
    ''' + enc + '''
    ''' + var + '''
    float *l = linear((float*)enc);
    if(argmax(l) == label){
        free(l);
        update_correct_predictions();
    }
    free(indices);
    free(data);
    free(enc);
}
                            '''

    def validateRequiredArgs(self):
        for i in self.required_arguments:
            if i not in self.set_arguments:
               self.error_missing_required_arguemnts(i)

    def validateVarsDeclaration(self):
        for i in self.used_vars:
            if i not in self.declared_vars:
               self.error_missing_var_declaration(i)

    def validateVarsUsage(self):
        for i in self.declared_vars:
            if i not in self.used_vars:
               self.warning_unused_var(i,ln=None)

    def error_missing_required_arguemnts(self, argument):
        self.error(msg='Missing argument: '+argument, ln=None)

    def error_repeated_directive(self, directive, ln):
        self.error(msg='Repeated directive: ' + directive, ln=ln)

    def error_missing_var_declaration(self, var):
        self.error(msg='Missing variable definition: ' + str(var), ln=None)

    def error(self, msg, ln=None):
        if ln:
            print('[ERROR]: ', msg, 'at line', ln)
        else:
            print('[ERROR]: ', msg)
        exit(1)

    def warning_unused_var(self, var, ln=None):
        self.warning(msg='Unused variable: ' + str(var), ln=None)

    def warning(self, msg, ln=None):
        if ln:
            print('[WARN]: ', msg, 'at line', ln)
        else:
            print('[WARN]: ', msg)

    def print_parsed_and_validated_input(self):
        print('NAME:', self.name)
        print('CLASSES:', self.classes)
        print('DIMENSIONS:', self.dimensions)
        print('TRAIN SIZE:', self.train_size)
        print('TEST SIZE:', self.test_size)
        print('VECTOR SIZE:', self.vector_size)
        print('NUM THREADS:', self.num_threads)
        print('EXEC TYPE:', self.type)
        print('EMBEDDINGS:', self.embeddings)
        print('ENCODING:', self.encoding)
        print('WEIGHT VARIABLE:', self.weight)
        print('DEBUG:', self.debug)
        # print('Encoding fun:', self.encoding_fun)

    class astDirective:
        action = None
        params = None

        def __init__(self, action=None, params=None):
            self.action = action
            self.params = params

        def execute(self):
            return self.action, self.params

        def __str__(self):
            return '[AST] %s %s' % (self.action, ' '.join(str(x) for x in self.params))

        @staticmethod
        def isADelayedAction(x=None):
            return ('x' != None and isinstance(x, hdccAST.astDirective))

        @staticmethod
        def resolve(x):
            if not hdccAST.astDirective.isADelayedAction(x):
                return x
            else:
                return x.execute()
