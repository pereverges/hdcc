from execution_type import Types

class hdccAST:
    def __init__(self):
        self.set_arguments = set()
        self.declared_vars = set()
        self.used_vars = set()
        self.weight = None
        self.encoding = None
        self.encoding_fun = None
        self.embeddings = []
        self.name = None
        self.classes = None
        self.dimensions = None
        self.wait = None
        self.debug = False
        self.multiset = False
        self.multibind = False
        self.train_size = None
        self.test_size = None
        self.num_threads = None
        self.type = None
        self.vector_size = 128
        self.required_arguments = {'NAME', 'EMBEDDING', 'ENCODING', 'CLASSES', 'DIMENSIONS', 'TEST_SIZE', 'TRAIN_SIZE', 'TYPE'}

    def get_ast_obj(self):
        return self.name, self.classes, self.dimensions, self.used_vars, self.weight, self.encoding, self.embeddings, self.debug, self.encoding_fun, self.train_size, self.test_size, self.num_threads, self.vector_size, self.type

    def validateDirective(self, x):
        action, params = self.astDirective.resolve(x)

        if action not in self.set_arguments:
            self.set_arguments.add(action)
        else:
            self.error_repeated_directive(action, params[0])
        if action == 'ENCODING':
            if self.dimensions is not None and self.type is not None:
                self.wait = False
                self.encoding, var, t = self.unpack_encoding(params[1])
                self.encoding_build(var, t)
            else:
                self.wait = params[1]
            # print('Resulting encoding', encoding)
        if action == 'EMBEDDING':
            self.set_declared_vars(params)
            # print(self.declared_vars)
        if action == 'NAME':
            self.name = params[1].lower()
        if action == 'CLASSES':
            self.classes = params[1]
        if action == 'DEBUG':
            self.debug = True
        if action == 'DIMENSIONS':
            self.dimensions = params[1]
            if self.wait is not None and self.type is not None:
                self.encoding, var, t = self.unpack_encoding(self.wait)
                self.encoding_build(var, t)
                self.wait = None
        if action == 'TRAIN_SIZE':
            self.train_size = params[1]
        if action == 'TEST_SIZE':
            self.test_size = params[1]
        if action == 'NUM_THREADS':
            self.num_threads = params[1]
            if self.wait is not None:
                self.encoding, var, t = self.unpack_encoding(self.wait)
                self.encoding_build(var, t)
                self.wait = None
        if action == 'VECTOR_SIZE':
            self.vector_size = params[1]
        if action == 'TYPE':
            if params[1] == 'SEQUENTIAL':
                self.type = Types.SEQUENTIAL
            elif params[1] == 'PARALLEL':
                self.type = Types.PARALLEL

    def unpack_encoding(self, i):
        if i[1] == 'BIND':
            self.multibind = True
            if self.unpack_encoding(i[2])[1] not in self.used_vars:
                return 'bind(' + self.unpack_encoding(i[2])[0] + ',' + self.unpack_encoding(i[3])[0] + ',INPUT_DIM' + ')', self.unpack_encoding(i[3])[1] + '(NUM_BATCH * i) + j] * (' + self.unpack_encoding(i[2])[1] + '* NUM_BATCH + j])', 'bind'
            return 'bind(' + self.unpack_encoding(i[2])[0] + ',' + self.unpack_encoding(i[3])[0] + ',INPUT_DIM' + ')', self.unpack_encoding(i[2])[1] + ' +(NUM_BATCH * i) + j] * (' + self.unpack_encoding(i[3])[1] + '* NUM_BATCH + j])', 'bind'
        elif i[1] == 'BUNDLE':
            return 'bundle(' + self.unpack_encoding(i[1][2])[0] + ',' + 'INPUT_DIM' + ')', '', 'bundle'
        elif i[1] == 'MULTISET':
            self.multiset = True
            return 'multiset(' + self.unpack_encoding(i[2])[0] + ',' + 'INPUT_DIM' + ')', self.unpack_encoding(i[2])[1], 'multiset'
        else:
            self.set_used_var(i)
            if i == self.weight:
                return self.weight, ''+self.weight+'[(int)indices[i]', 'var'
            return i.upper(), i.upper()+'[', 'var'

    def set_used_var(self, var):
        self.used_vars.add(var)

    def set_declared_vars(self,params):
        if type(params[1]) == list:
            for i in params[1]:
                if i[1] == 'WEIGHT':
                    self.declared_vars.add(i[2])
                    self.weight = i[2]
                    self.embeddings.append([i[3], i[2], i[4]])
                else:
                    self.declared_vars.add(i[1])
                    self.embeddings.append([i[2], i[1], i[3]])
        else:
            self.declared_vars.add(params[1][2])

    def encoding_build(self, var, t):
        if self.type == Types.SEQUENTIAL:
            self.encoding_build_sequential(var, t)
        elif self.type == Types.PARALLEL:
            self.encoding_build_parallel(var, t)

    def encoding_build_sequential(self,var,t):
        if t == 'multiset':
            var = 'arr[j] += ' + var + ';'
        else:
            var = ' arr[(DIMENSIONS*i) + j] = ' + var + ';'

        fun_head = '''
void* encode_function(float* indices){'''
        if self.multiset and self.multibind:
            self.encoding_fun = fun_head + '''
    int i, j;
    f4si *arr = calloc(DIMENSIONS, sizeof(int));
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
            ''' + var + '''
        }
    }
    return arr;
}
                                '''

        else:
            self.encoding_fun = fun_head + '''
    int i, j;
    f4si *arr = calloc(DIMENSIONS * INPUT_DIM, sizeof(int));
    for(i = 0; i < NUM_BATCH; ++i){
        for(j = 0; j < BATCH; j++){
            ''' + var + '''
        }
    }
    return arr;
}
                        '''

    def encoding_build_parallel(self, var, t):
        if t == 'multiset':
            var = 'aux[j] += ' + var + ';'
        else:
            var = ' res[(DIMENSIONS*i) + j] = ' + var + ';'

        fun_head = '''
void encode_fun(void* task){'''
        if self.multiset and self.multibind:
            self.encoding_fun = fun_head + '''
    int index = ((struct EncodeTask*)task) -> split_start;
    float* indices = ((struct EncodeTask*)task) -> indices;
    f4si* res = ((struct EncodeTask*)task) -> res;
    int i, j;
    f4si *aux = calloc(DIMENSIONS,sizeof(int));
    for(i = index; i < SPLIT_SIZE+index; ++i){
        if (index < INPUT_DIM){
            for(j = 0; j < NUM_BATCH; j++){
                ''' + var + '''
            }
        }
    }
    for(j = 0; j < NUM_BATCH; j++){
        lock_condition(pool);
        res[j] += aux[j];
        unlock_condition(pool);
    }
    free(aux);
    free(task);
}
                            '''

        else:
            self.encoding_fun = fun_head + '''
    int index = ((struct EncodeTask*)task) -> split_start;
    float* indices = ((struct EncodeTask*)task) -> indices;
    f4si* res = ((struct EncodeTask*)task) -> res;
    int i, j;
    f4si *aux = calloc(DIMENSIONS * INPUT_DIM,sizeof(int));
    for(i = index; i < SPLIT_SIZE+index; ++i){
        if (index < INPUT_DIM){
            for(j = 0; j < NUM_BATCH; j++){
                ''' + var + '''
            }
        }
    }
    free(task);
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