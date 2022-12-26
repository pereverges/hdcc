from execution_type import Types

class hdccAST:
    def __init__(self):
        self.set_arguments = set()
        self.declared_vars = set()
        self.used_vars = set()
        self.weight = None
        self.input = None
        self.input_dim = None
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
        self.memory_batch = 200
        self.high = 0
        self.basic = True
        self.required_arguments = {'NAME', 'EMBEDDING', 'ENCODING', 'CLASSES', 'DIMENSIONS', 'TEST_SIZE', 'TRAIN_SIZE', 'TYPE', 'INPUT_DIM'}

    def get_ast_obj(self):
        return self.name, self.classes, self.dimensions, self.used_vars, self.weight, self.encoding, self.embeddings, self.debug, self.encoding_fun, self.train_size, self.test_size, self.num_threads, self.vector_size, self.type, self.memory_batch, self.input, self.input_dim, self.high, self.basic

    def validateDirective(self, x):
        action, params = self.astDirective.resolve(x)

        if action not in self.set_arguments:
            self.set_arguments.add(action)
        else:
            self.error_repeated_directive(action, params[0])
        if action == 'ENCODING':
            if self.dimensions is not None and self.type is not None:
                self.wait = False
                enc = ''
                _, self.encoding = self.unpack_encoding(params[1], enc)
                self.encoding_build(self.encoding, self.encoding)
                print("EEEENC", self.encoding)
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
                enc = ''
                enc = ''
                _, self.encoding = self.unpack_encoding(self.wait, enc)
                self.encoding_build(self.encoding, self.encoding)
                print("EEEENC", self.encoding)
                self.wait = None
        if action == 'TRAIN_SIZE':
            self.train_size = params[1]
        if action == 'TEST_SIZE':
            self.test_size = params[1]
        if action == 'NUM_THREADS':
            self.num_threads = params[1]
            if self.wait is not None:
                enc = ''
                _, self.encoding = self.unpack_encoding(self.wait, enc)
                self.encoding_build(self.encoding, self.encoding)
                print("EEEENC", self.encoding)
                self.wait = None
        if action == 'VECTOR_SIZE':
            self.vector_size = params[1]
        if action == 'TYPE':
            if params[1] == 'SEQUENTIAL':
                self.type = Types.SEQUENTIAL
            elif params[1] == 'PARALLEL':
                self.type = Types.PARALLEL
            elif params[1] == 'PARALLEL_MEMORY_EFFICIENT':
                self.type = Types.PARALLEL_MEMORY_EFFICIENT
        if action == 'MEMORY_BATCH':
            self.memory_batch = params[1]
        if action == 'INPUT_DIM':
            self.input_dim = params[1]
        if action == 'MODE':
            self.basic = params[1]

    def unpack_encoding(self, i, enc):
        '''
        if i[1] == 'BIND':
            self.multibind = True
            if self.unpack_encoding(i[2])[1] not in self.used_vars:
                # return 'bind(' + self.unpack_encoding(i[2])[0] + ',' + self.unpack_encoding(i[3])[0] + ',INPUT_DIM' + ')', self.unpack_encoding(i[3])[1] + '(NUM_BATCH * i) + j] * (' + self.unpack_encoding(i[2])[1] + '* NUM_BATCH + j])', 'bind'
                return 'bind(' + self.unpack_encoding(i[2])[0] + ',' + self.unpack_encoding(i[3])[0] + ',INPUT_DIM' + ')', self.unpack_encoding(i[3])[1] + '* NUM_BATCH + j] * (' + self.unpack_encoding(i[2])[1] + '(NUM_BATCH * i) + j])', 'bind'
            #return 'bind(' + self.unpack_encoding(i[2])[0] + ',' + self.unpack_encoding(i[3])[0] + ',INPUT_DIM' + ')', self.unpack_encoding(i[2])[1] + ' +(NUM_BATCH * i) + j] * (' + self.unpack_encoding(i[3])[1] + '* NUM_BATCH + j])', 'bind'
            return 'bind(' + self.unpack_encoding(i[2])[0] + ',' + self.unpack_encoding(i[3])[0] + ',INPUT_DIM' + ')', self.unpack_encoding(i[2])[1] + ' * NUM_BATCH + j] * (' + self.unpack_encoding(i[3])[1] + '+(NUM_BATCH * i) + j])', 'bind'
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
        '''
        '''
        f4si * enc = calloc(DIMENSIONS, sizeof(int));
        enc = bind_forward(CHANNELS, SIGNALS, indices, enc);
        enc = multiset(enc);
        enc = permute(enc, 1);
        '''
        '''
        if self.basic:
            if i[1] == 'BIND':
                self.multibind = True
                if self.unpack_encoding(i[3])[1] == self.weight:
                    return 'bind_forward(' + self.unpack_encoding(i[2])[1] + ',' + self.unpack_encoding(i[3])[1] + ',indices' + ')', 'bind_forward(' + self.unpack_encoding(i[2])[1] + ',' + self.unpack_encoding(i[3])[1] + ',indices' + ')', 'bind_forward'
                else:
                    return 'bind(' + self.unpack_encoding(i[2])[1] + ',' + self.unpack_encoding(i[3])[1] + '' + ')', 'bind(' + self.unpack_encoding(i[2])[1] + ',' + self.unpack_encoding(i[3])[1] + '' + ')', 'bind'
            elif i[1] == 'BUNDLE':
                return 'bundle(' + self.unpack_encoding(i[1][2])[0] + ')', '', 'bundle'
            elif i[1] == 'PERMUTE':
                return 'permute(' + self.unpack_encoding(i[2])[0] + ',' + str(i[3]) +')', 'permute(' + self.unpack_encoding(i[2])[0] + ',' + str(i[3]) +')', 'permute'
            elif i[1] == 'MULTISET':
                self.multiset = True
                return 'multiset(' + self.unpack_encoding(i[2])[0] + ')', 'multiset(' + self.unpack_encoding(i[2])[0] + ')', 'multiset'
            else:
                self.set_used_var(i)
                if i == self.weight:
                    return self.weight, self.weight, 'var'
                return i.upper(), i.upper(), 'var'
        '''
        if self.basic:
            if i[1] == 'BIND':
                self.multibind = True
                if i[3] == self.weight:
                    b1, enc1 = self.unpack_encoding(i[2],enc)
                    b2, enc2 = self.unpack_encoding(i[3],enc)
                    enc += enc1 + enc2
                    enc += 'enc = bind_forward(' + b1 + ',' + b2 + ', indices, enc' + ');\n'
                    return 'enc', enc
                else:
                    b1, enc1 = self.unpack_encoding(i[2],enc)
                    b2, enc2 = self.unpack_encoding(i[3],enc)
                    enc += enc1 + enc2
                    enc += 'enc = bind(' + b1 + ',' + b2 + ', enc);\n'
                    return 'enc', enc
            elif i[1] == 'BUNDLE':
                return '    bundle(' + self.unpack_encoding(i[1][2]) + ')', '', 'bundle'
            elif i[1] == 'PERMUTE':
                b, enc_aux = self.unpack_encoding(i[2], enc)
                enc += enc_aux
                enc += '    enc = permute(' + b + ',' + str(i[3]) + ');\n'
                return 'enc', enc
            elif i[1] == 'MULTISET':
                self.multiset = True
                b, enc_aux = self.unpack_encoding(i[2], enc)
                enc += enc_aux
                enc += '    enc = multiset(' + b + ');\n'
                return 'enc', enc
            else:
                self.set_used_var(i)
                if i == self.weight:
                    return self.weight, enc
                return i.upper(), enc

    def set_used_var(self, var):
        self.used_vars.add(var)

    def set_declared_vars(self,params):
        if type(params[1]) == list:
            for i in params[1]:
                print(isinstance(i[3], list))
                if i[1] == 'WEIGHT':
                    self.declared_vars.add(i[2])
                    self.weight = i[2]
                    if isinstance(i[3], list):
                        self.embeddings.append([i[3][0], i[2], i[3][1]])
                        self.high = i[4]
                    else:
                        self.embeddings.append([i[3], i[2], i[4]])
                elif i[1] == 'INPUT':
                    self.declared_vars.add(i[2])
                    self.input = i[2]
                    if isinstance(i[3], list):
                        self.high = i[4]
                        self.embeddings.append([i[3][0], i[2], i[3][1], i[4]])
                    else:
                        self.embeddings.append([i[3], i[2], i[4]])
                else:
                    self.declared_vars.add(i[1])
                    if isinstance(i[3], list):
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
            self.encoding_build_parallel(var, t)
        elif self.type == Types.PARALLEL_MEMORY_EFFICIENT:
            self.encoding_build_parallel_memory_efficient(var, t)

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

    def encoding_build_parallel_memory_efficient(self, var, t):
        fun_head_train = '''
void encode_train_task(void* task){'''

        fun_head_test = '''
void encode_test_task(void* task){'''

        if self.basic:
            #var = '= ' + var + ';'
            self.encoding_fun = fun_head_train + '''
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,''' + self.weight + '''_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS, sizeof(int));
    ''' + var + '''
    hard_quantize((float*)enc,1);
    update_weight((float*)enc,label);
    //free(enc);
    //free(indices);
    //free(data);
}

''' + fun_head_test + '''
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,''' + self.weight + '''_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS, sizeof(int));
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

        else:
            if t == 'multiset':
                var = 'enc[j] += ' + var + ';'
            else:
                var = ' enc[(DIMENSIONS*i) + j] = ' + var + ';'
            if self.multiset and self.multibind:
                self.encoding_fun = fun_head_train + '''
    f4si *enc = calloc(DIMENSIONS,sizeof(int));
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,''' + self.weight + '''_DIM-1, indices);
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
            ''' + var + '''
        }
    }
    hard_quantize((float*)enc,1);
    update_weight((float*)enc,label);
    //free(enc);
    //free(indices);
    //free(data);
}

''' + fun_head_test + '''
    f4si *enc = calloc(DIMENSIONS,sizeof(int));
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,''' + self.weight + '''_DIM-1, indices);
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
            ''' + var + '''
        }
    }
    float *l = linear((float*)enc);
    free(enc);
    if(argmax(l) == label){
        free(l);
        update_correct_predictions();
    }
    free(indices);
    free(data);
}
                            '''

            else:
                self.encoding_fun = fun_head_train + '''
    int index = ((struct EncodeTaskTrain*)task) -> split_start;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    float* data = ((struct EncodeTaskTrain*)task) -> data;
    int label = ((struct EncodeTaskTrain*)task) -> label;
    map_range_clamp_one(data,INPUT_DIM-1, indices);
    f4si *res = calloc(DIMENSIONS*INPUT_DIM,sizeof(int));
    int i, j;
    for(i = index; i < SPLIT_SIZE+index; ++i){
        if (index < INPUT_DIM){
            for(j = 0; j < NUM_BATCH; j++){
                ''' + var + '''
            }
        }
    }
    hard_quantize((float *)res,1);
    update_weight((float *)res, label);
    free(res);
    free(indices);
}

''' + fun_head_test + '''
    int index = ((struct EncodeTaskTrain*)task) -> split_start;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    float* data = ((struct EncodeTaskTrain*)task) -> data;
    int label = ((struct EncodeTaskTrain*)task) -> label;
    map_range_clamp_one(data,INPUT_DIM-1, indices);
    f4si *res = calloc(DIMENSIONS*INPUT_DIM,sizeof(int));
    int i, j;
    for(i = index; i < SPLIT_SIZE+index; ++i){
        if (index < INPUT_DIM){
            for(j = 0; j < NUM_BATCH; j++){
                ''' + var + '''
            }
        }
    }
    hard_quantize((float *)res,1);
    update_weight((float *)res, label);
    free(res);
    free(indices);
} '''

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