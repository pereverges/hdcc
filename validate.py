class hdccAST:
    def __init__(self):
        self.set_arguments = set()
        self.declared_vars = set()
        self.used_vars = set()
        self.weight = None
        self.encoding = None
        self.embeddings = []
        self.name = None
        self.classes = None
        self.dimensions = None
        self.wait = None
        self.required_arguments = {'NAME', 'EMBEDDING', 'ENCODING', 'CLASSES', 'DIMENSIONS'}

    def get_ast_obj(self):
        return self.name, self.classes, self.dimensions, self.used_vars, self.weight, self.encoding, self.embeddings

    def unpack_encoding(self, i):
        if i[1] == 'BIND':
            return 'bind(' + self.unpack_encoding(i[2]) + ',' + self.unpack_encoding(i[3]) + ',INPUT_DIM' + ')'
        elif i[1] == 'BUNDLE':
            return 'bundle(' + self.unpack_encoding(i[1][2]) + ',' + 'INPUT_DIM' + ')'
        elif i[1] == 'MULTISET':
            return 'multiset(' + self.unpack_encoding(i[2]) + ',' + 'INPUT_DIM' + ')'
        else:
            self.set_used_var(i)
            if i == self.weight:
                return 'f'
            return i.upper()

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

    def validateDirective(self, x):
        action, params = self.astDirective.resolve(x)

        if action not in self.set_arguments:
            self.set_arguments.add(action)
        else:
            self.error_repeated_directive(action, params[0])
        if action == 'ENCODING':
            if self.dimensions is not None:
                self.wait = False
                self.encoding = self.unpack_encoding(params[1])
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
        if action == 'DIMENSIONS':
            self.dimensions = params[1]
            if self.wait is not None:
                self.encoding = self.unpack_encoding(self.wait)

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
        print('Name:', self.name)
        print('Classes:', self.classes)
        print('Dimensions:', self.dimensions)
        print('Embeddings:', self.embeddings)
        print('Encoding:', self.encoding)
        print('Weight variable:', self.weight)

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