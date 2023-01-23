class SequentialRepresentation:
    def __init__(self, name, classes, dimensions, vars, weight_var, encoding, embeddings, debug, encoding_fun,
                 train_size, test_size, num_threads, vector_size, type, input_dim, high, basic, padding):
        self.name = name
        self.basic_name = self.get_basic_name(name)
        self.classes = classes
        self.dimensions = dimensions
        self.vars = vars
        self.weight_var = weight_var
        self.encoding = encoding
        self.embeddings = embeddings
        self.debug = debug
        self.encoding_fun = encoding_fun
        self.type = type
        self.train_size = train_size
        self.test_size = test_size
        self.vector_size = vector_size
        self.num_threads = num_threads
        self.input_dim = input_dim
        self.high = high
        self.basic = basic
        self.padding = padding

    def get_basic_name(self, name):
        temp = len(name)
        for c in name:
            if c.isdigit():
                temp = name.index(c)
                break
        return name[0:temp]

    def define_embeddings(self):
        embedding = ''
        for i in self.embeddings:
            if i[0] == 'LEVEL':
                if i[1] == self.weight_var:
                    embedding += ("\n    " + str(i[1].upper() + " = level_hv(" + str(i[1]) + '_DIM' + ");"))
                else:
                    embedding += ("\n    " + str(i[1].upper() + " = level_hv(" + str(i[1]) + '_DIM' + ");"))
            if i[0] == 'RANDOM':
                if i[1] == self.weight_var:
                    embedding += ("\n    " + str(i[1].upper() + " = random_hv(" + str(i[1]) + '_DIM' + ");"))
                else:
                    embedding += ("\n    " + str(i[1].upper() + " = random_hv(" + str(i[1]) + '_DIM' + ");"))
            if i[0] == 'RANDOM_PADDING':
                if i[1] == self.weight_var:
                    embedding += ("\n    " + str(i[1].upper() + " = random_hv(" + str(i[1]) + '_DIM' + ");"))
                else:
                    embedding += ("\n    " + str(i[1].upper() + " = random_hv(" + str(i[1]) + '_DIM' + ");"))
        return embedding

    def run_sequential(self):
        self.makefile()
        self.define_header()
        self.define_dataset_loaders()
        self.define_math_functions()
        self.define_hdc_functions()
        self.define_train_and_test()
        self.general_main()

    def makefile(self):
        doto = '.o'
        with open('../Makefile', 'w') as file:
            file.write('CC=clang' + '\n')
            file.write('CFLAGS=-I.' + '\n')
            file.write(self.name + ': ' + self.name + doto + '\n')
            file.write('\t$(CC) -o ' + self.name + ' ' + self.name + doto + ' -lm -O3\n')

    def define_header(self):
        self.define_includes()
        self.define_constants()

    def define_includes(self):
        with open(self.name.lower() + '.c', 'w') as file:
            file.write('/*********** ' + self.name + ' ***********/\n')
            file.write('#include <stdio.h>\n')
            file.write('#include <stdlib.h>\n')
            file.write('#include <stdint.h>\n')
            file.write('#include <string.h>\n')
            file.write('#include <math.h>\n')
            if self.debug:
                file.write('#include <time.h>\n')
            file.write('#ifndef max\n')
            file.write('#define max(a,b) (((a) > (b)) ? (a) : (b))\n')
            file.write('#endif\n')
            file.write('#ifndef min\n')
            file.write('#define min(a,b) (((a) < (b)) ? (a) : (b))\n')
            file.write('#endif\n')

    def define_constants(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write('/*********** CONSTANTS ***********/\n')
            file.write('#define TRAIN ' + str(self.train_size) + '\n')
            file.write('#define TEST ' + str(self.test_size) + '\n')

            file.write('#define DIMENSIONS ' + str(self.dimensions) + '\n')
            file.write('#define CLASSES ' + str(self.classes) + '\n')
            file.write('#define VECTOR_SIZE ' + str(self.vector_size) + '\n')
            file.write('float *WEIGHT;\n')
            file.write('typedef float f4si __attribute__ ((vector_size (' + str(self.vector_size) + ')));\n')

            file.write('#define INPUT_DIM ' + str(self.input_dim) + '\n')

            for i in self.embeddings:
                file.write('f4si* ' + str(i[1]) + ';\n')
                file.write('#define ' + str(i[1]) + '_DIM ' + str(i[2]) + '\n')

            file.write('#define BATCH ' + str(int(self.vector_size / 4)) + '\n')
            file.write('#define NUM_BATCH ' + str(int(self.dimensions / (self.vector_size / 4))) + '\n')

            file.write('float *TRAIN_DATA[TRAIN];\n')
            file.write('float *TRAIN_LABELS[TRAIN];\n')
            file.write('float *TEST_DATA[TEST];\n')
            file.write('float *TEST_LABELS[TEST];\n')

            file.write('struct EncodeTask {\n')
            file.write('    int split_start;\n')
            file.write('    float* indices;\n')
            file.write('    f4si *res;\n')
            file.write('};\n')

    def define_dataset_loaders(self):
        self.load_data()
        self.load_labels()
        self.load_dataset()

    def load_data(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
    void load_data(float* data[], char* path){
        FILE * fp;
        char * line = NULL;
        size_t len = 0;
        ssize_t read;
        fp = fopen(path, "r");
        if (fp == NULL)
            exit(EXIT_FAILURE);
        char* token;
        int count = 0;
        while ((read = getline(&line, &len, fp)) != -1) {
            token = strtok(line, ",");
            for (int i = 0; i < INPUT_DIM; i++){
              *(data[count] + i) = (float) atof(token);
              token = strtok(NULL, ",");
            }
            count += 1;
        }
        fclose(fp);
        if (line)
            free(line);`
    }
            '''
            )

    def load_labels(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                    '''
    void load_label(float* data[], char* path){
        FILE * fp;
        char * line = NULL;
        size_t len = 0;
        ssize_t read;
        fp = fopen(path, "r");
        if (fp == NULL)
            exit(EXIT_FAILURE);
        char* token;
        int count = 0;
        while ((read = getline(&line, &len, fp)) != -1) {
            *(data[count]) = atoi(line);
            count += 1;
        }
        fclose(fp);
        if (line)
            free(line);
    }       
                    '''
                )

    def load_dataset(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                    '''
    void load_dataset(char **argv){
        for (int i = 0; i < TRAIN; i++){
            TRAIN_DATA[i] = (float *) calloc(INPUT_DIM, sizeof(float));
            TRAIN_LABELS[i] = (float *) calloc(1, sizeof(int));
        }
    
        for (int i = 0; i < TEST; i++){
            TEST_DATA[i] = (float *) calloc(INPUT_DIM, sizeof(float));
            TEST_LABELS[i] = (float *) calloc(1, sizeof(int));
        }
    
        load_data(TRAIN_DATA, argv[1]);
        load_data(TEST_DATA, argv[3]);
        load_label(TRAIN_LABELS, argv[2]);
        load_label(TEST_LABELS, argv[4]);
    }
                    '''
                )

    def define_math_functions(self):
        self.define_random_number()
        self.define_random_vector()
        self.define_weights()
        self.define_update_weights()
        self.define_linear()
        self.define_norm2()
        self.define_normalize()
        self.define_argmax()
        self.define_map_range_clamp()
        self.define_hard_quantize()

    def define_random_number(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                    '''
    float get_rand(float low_bound, float high_bound){
        return (float) ((float)rand() / (float)RAND_MAX) * (high_bound - low_bound) + low_bound;
    }
                    '''
                )

    def define_random_vector(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                    '''
    f4si *random_vector(int size, float low_bound, float high_bound){
       f4si *arr = calloc(size * DIMENSIONS, sizeof(float));
       int i, j, k;
       for (i = 0; i < size; i++){
          for (j = 0; j < NUM_BATCH; j++){
             for(k = 0; k < BATCH; k++){
                arr[i*NUM_BATCH+j][k] = get_rand(low_bound, high_bound);
             }
          }
       }
       return arr;
    }
                    '''
                )

    def define_weights(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                    '''
    void weights(){
        WEIGHT = (float *)calloc(CLASSES * DIMENSIONS, sizeof(float));
    }
                        '''
                )

    def define_update_weights(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                    '''
    void update_weight(float* encoding, int feature){
        int i;
        for(i = 0; i < DIMENSIONS; i++){
            *(WEIGHT + feature*DIMENSIONS + i) += (float)*(encoding + i);
        }
    }
                        '''
                )

    def define_update_correct_predictions(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                    '''
    void update_correct_predictions(){
        lock_condition(pool);
        CORRECT_PREDICTIONS += 1;
        unlock_condition(pool);
    }
                        '''
                )

    def define_linear(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                    '''
    float* linear(float* m){
        int j, k;
        float *arr = (float *)calloc(CLASSES, sizeof(float));
        for (j = 0; j < DIMENSIONS; ++j) {
          for (k = 0; k < CLASSES; ++k) {
             *(arr + k) += (float)*(m + j) * *(WEIGHT + k*DIMENSIONS + j);
          }
       }
        return arr;
    }
                        '''
                )

    def define_norm2(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                    '''
    float norm2(int feature){
       float norm = 0.0;
       int i;
       for (i = 0; i < DIMENSIONS; i++){
          norm += *(WEIGHT + feature*DIMENSIONS + i) * *(WEIGHT + feature*DIMENSIONS + i);
       }
       return sqrt(norm);
    }
                        '''
                )

        def define_normalize(self):
            with open(self.name.lower() + '.c', 'a') as file:
                file.write(
                    '''
    void normalize(){
       float eps = 1e-12;
       int i, j;
       for (i = 0; i < CLASSES; i++){
          float norm = norm2(i);
          for (j = 0; j < DIMENSIONS; j++){
            *(WEIGHT + i*DIMENSIONS + j) = *(WEIGHT + i*DIMENSIONS + j) / max(norm,eps);
          }
       }
    }
                        '''
                )

    def define_argmax(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                    '''
    int argmax(float* classify){
       int i;
       int max_i = 0;
       float max = 0;
       for (i = 0; i < CLASSES; i++){
           if(*(classify + i) > max){
               max = *(classify + i);
               max_i = i;
           }
       }
       return max_i;
    }
                        '''
                )

    def define_map_range_clamp(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                    '''
    float** map_range_clamp(float* arr[], int size, float out_max, float* res[]){
       float in_min = 0;
       float in_max = 1;
       float out_min = 0;
       int i, j;
       for (i = 0; i < size; i++){
          for (j = 0; j < INPUT_DIM; j++){
            float map_range = round(out_min + (out_max - out_min) * (*(arr[i] + j) - in_min) / (in_max - in_min));
            *(res[i] + j) = min(max(map_range,out_min),out_max);
          }
          free(arr[i]);
       }
       return res;
    }
                        '''
                )

    def define_hard_quantize(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void hard_quantize(float *arr, int size){
   int i, j;
   for (i = 0; i < size; i++){
      for (j = 0; j < DIMENSIONS; j++){
        int value = *(arr + i*DIMENSIONS + j);
        if (value > 0){
          *(arr + i*DIMENSIONS + j) = 1.0;
        } else {
            *(arr + i*DIMENSIONS + j) = -1.0;
        }
      }
   }
}
                    '''
            )

    def bind(self):
        """Binds a set of hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *bind(f4si *a, f4si *b){
    int i, j;
    f4si *enc = (f4si *)calloc(DIMENSIONS * INPUT_DIM, sizeof(int));
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
             enc[(NUM_BATCH * i) + j] = a[(NUM_BATCH * i) + j] * b[NUM_BATCH + j];
        }
    }
    return enc;
}
                '''
            )

    def multibind_forward(self):
        """Binds a set of hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *multibind_forward(f4si *a, f4si *b, float* indices, f4si* enc){
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
            enc[(NUM_BATCH * i) + j] = a[(NUM_BATCH * i) + j] * b[(int)indices[i]* NUM_BATCH + j];
        }
    }
    return enc;
}

                '''
            )

    def ngram(self):
        """Ngram a set of hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''

f4si* ngram(f4si* arr, float* indices, f4si* enc, const int n){
    int i, j, k;
    f4si aux;
    f4si * forward_arr = calloc(DIMENSIONS * n, sizeof(int));
    f4si actual;
    float n1, p1;
    float n2, p2;
    for (i = 0; i < (INPUT_DIM-(n-1)); ++i){
        for (j = 0; j < NUM_BATCH; j++){
            for (k = 0; k < n; ++k){
                if (k == 0){
                    if (j == NUM_BATCH-1){
                        for (int m = 0; m < n; m++){
                            if (indices[i] != padding){
                                forward_arr[(m*NUM_BATCH)+j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                            } else {
                                forward_arr[(m*NUM_BATCH)+j] *= 0;
                            }
                        }
                       aux = __builtin_shufflevector(forward_arr[k*NUM_BATCH+j],forward_arr[k*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       aux[0] = p1;
                       aux[1] = p2;
                   } else if (j == 0){
                        for (int m = 0; m < n; m++){
                            if (indices[i] != padding){
                                forward_arr[(m*NUM_BATCH)+j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                            } else {
                                forward_arr[(m*NUM_BATCH)+j] *= 0;
                            }
                        }
                       aux = __builtin_shufflevector(forward_arr[k*NUM_BATCH+j],forward_arr[k*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       p1 = aux[0];
                       p2 = aux[1];
                   } else {
                        for (int m = 0; m < n; m++){
                            if (indices[i] != padding){
                                forward_arr[(m*NUM_BATCH)+j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                            } else {
                                forward_arr[(m*NUM_BATCH)+j] *= 0;
                            }
                        }
                       aux = __builtin_shufflevector(forward_arr[k*NUM_BATCH+j],forward_arr[k*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       n1 = aux[0];
                       n2 = aux[1];
                       aux[0] = p1;
                       aux[1] = p2;
                       p1 = n1;
                       p2 = n2;
                    }
                    actual = aux; // 2 pos
                } else if (k == n-1){
                    actual = actual * forward_arr[k*NUM_BATCH+j];
                } else {
                       if (j == NUM_BATCH-1){
                           aux = __builtin_shufflevector(forward_arr[k*NUM_BATCH+j],forward_arr[k*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           aux[0] = p1;
                       } else if (j == 0){
                           aux = __builtin_shufflevector(forward_arr[k*NUM_BATCH+j],forward_arr[k*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           p1 = aux[0];
                       } else {
                           aux = __builtin_shufflevector(forward_arr[k*NUM_BATCH+j],forward_arr[k*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           n1 = aux[0];
                           aux[0] = p1;
                           p1 = n1;
                       }
                    //res[i*NUM_BATCH+j] = res[(i)*NUM_BATCH+j] * shuffle(arr[(i+k)*NUM_BATCH+j], (n-1-k))

                    actual = aux * actual;
                }

              }
               enc[j] = enc[j] + actual;



        }
    }
    free(forward_arr);
    return enc;
}

                '''
            )


    def bundle(self):
        """Bundles two hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *multiset(f4si *a){
    int i, j;
    for(i = 1; i < INPUT_DIM; i++){
        for(j = 0; j < NUM_BATCH; ++j){
            a[j] += a[(NUM_BATCH * i) + j];
        }
    }
    return a;
}
                '''
            )

    def multiset_bind_forward(self):
        """Bundles two hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *multiset_bind_forward(f4si *a, f4si *b, float* indices, f4si* enc){
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
            enc[j] += a[(NUM_BATCH * i) + j] * b[(int)indices[i]* NUM_BATCH + j];
        }
    }
    return enc;
}
                '''
            )

    def multiset_bind(self):
        """Bundles two hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *multiset_bind(f4si *a, f4si *b, f4si* enc){
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
             enc[j] += a[(NUM_BATCH * i) + j] * b[NUM_BATCH + j];
        }
    }
    return enc;
}
                '''
            )

    def bundle_forward(self):
        """Bundles two hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *multiset_forward(f4si *a, float* indices){
    int i, j;
    for(i = 0; i < INPUT_DIM; i++){
        for(j = 0; j < NUM_BATCH; ++j){
            a[j] += a[(int)indices[i] * i + j];
        }
    }
    return a;
}
                '''
            )

    def permute(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
 f4si *permute(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float last;
    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
         for(k = 0; k < BATCH; k++){
            if ((BATCH*j)+k+dd < ((BATCH*NUM_BATCH))){
                if (k+dd >= BATCH){
                    int num = (k+dd) % BATCH;
                    res[(i-ini)*NUM_BATCH+j+1][num] = arr[i*NUM_BATCH+j][k];
                } else {
                    res[(i-ini)*NUM_BATCH+j][k+dd] = arr[i*NUM_BATCH+j][k];
                }
            } else {
                int num = (k+dd) % BATCH;
                res[(i-ini-1)*NUM_BATCH+j+(1)][num] = arr[i*NUM_BATCH+j][k];
            }
         }

      }

    }

    return res;
}
                '''
            )

    def define_hdc_functions(self):
        self.define_random_hv()
        self.define_level_hv()
        self.define_encoding_function()
        self.define_encoding()

    def define_random_hv(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *random_hv(int size){
   f4si *arr = calloc(size * DIMENSIONS, sizeof(float));
   int P = 50;
   int i, j, k;
   for (i = 0; i < size; i++){
      for (j = 0; j < NUM_BATCH; j++){
         for(k = 0; k < BATCH; k++){
            arr[i*NUM_BATCH+j][k] = rand() % 100 > P ? 1 : -1;
         }
      }
   }
   return arr;
}
                    '''
            )

    def define_level_hv(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *level_hv(int levels){
    int levels_per_span = levels-1;
    int span = 1;
    f4si *span_hv = random_hv(span+1);
    f4si *threshold_hv = random_vector(span,0,1);
    f4si *hv = calloc(levels * DIMENSIONS, sizeof(float));
    int i, j, k;
    for(i = 0; i < levels; i++){
        float t = 1 - ((float)i / levels_per_span);
        for(j = 0; j < NUM_BATCH; j++){
            for(k = 0; k < BATCH; k++){
                if((t > threshold_hv[j][k] || i == 0) && i != levels-1){
                    hv[i*NUM_BATCH+j][k] = span_hv[0*NUM_BATCH+j][k];
                } else {
                    hv[i*NUM_BATCH+j][k] = span_hv[1*NUM_BATCH+j][k];
                }
             }
        }
    }
    return hv;
}
                    '''
            )

    def define_encoding_function(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(self.encoding_fun)

    def define_encoding(self):
        """Generates the encoding"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write('''
    float* encodings(float* x){
        float* enc = (float*)encode_function(x);
        hard_quantize(enc,1);
        return enc;
    }
                    ''')

    def define_train_and_test(self):
        self.define_train_loop()
        self.define_test_loop()

    def define_train_loop(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void train_loop(){
    float *res[TRAIN];
    int i;
    for (i = 0; i < TRAIN; i++){
        res[i] = (float *)calloc(INPUT_DIM, sizeof(float));
    }
    map_range_clamp(TRAIN_DATA,TRAIN,''' + self.weight_var + '''_DIM-1, res);
    for(i = 0; i < TRAIN; i++){
        float* enc = encodings(res[i]);
        update_weight(enc,*(TRAIN_LABELS[i]));
        free(res[i]);
        free(enc);
    }
    normalize();
}
                    '''
            )

    def define_test_loop(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
float test_loop(){
float *res[TEST];
int i;
for (i = 0; i < TEST; i++){
    res[i] = (float *)calloc(INPUT_DIM, sizeof(float));
}
map_range_clamp(TEST_DATA,TEST,INPUT_DIM-1, res);
int correct_pred = 0;
for(i = 0; i < TEST; i++){
    float* enc = encodings(res[i]);
    float *l = linear(enc);
    int index = argmax(l);
    if((int)index == (int)*(TEST_LABELS[i])){
        correct_pred += 1;
    }
    free(l);
    free(res[i]);
    free(enc);
}
return correct_pred/(float)TEST;
}
                '''
            )

    def general_main(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int main(int argc, char **argv) {
    '''
                +
                str(self.define_embeddings())
                +
                '''
    weights();
    load_dataset(argv);
    '''
            )
            if self.debug:
                file.write(
                    '''
    struct timespec begin, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &begin);
                    '''
                )

            file.write(
                '''
    train_loop();
    float acc = test_loop();
                        ''')
            if self.debug:
                file.write(
                    '''
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("''' + self.basic_name + ''',%d,%f,%f \\n", DIMENSIONS,elapsed, acc);
                    '''
                )
            else:
                file.write(
                    '''
        printf("''' + self.basic_name + ''',%d,%f", DIMENSIONS, acc);
                    '''
                )

            file.write(
                '''
    }              
                '''
            )