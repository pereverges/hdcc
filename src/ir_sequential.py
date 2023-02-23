import math

def generate(length, shift):
    s = ''
    length = int(length)
    for i in range(length):
        if i < shift:
            s += ',' + str(length - shift + i)
        else:
            s += ',' + str((i - shift))
    return s


def generate_shuffle(n, vector_size):
    sif = ''
    for i in range(1, n):
        sif += '''
    if (d == ''' + str(i) +'''){
        res = __builtin_shufflevector(arr,arr''' + generate(vector_size / 4, i) + ''');
        return res;
    }'''
    shuffle = '''
f4si shuffle(f4si arr, int d){
    f4si res;
    ''' + sif + '''
    return res;
}
    '''
    return shuffle

class SequentialRepresentation:
    def __init__(self, name, classes, dimensions, vars, weight_var, encoding, embeddings, debug, encoding_fun,
                 train_size, test_size, num_threads, vector_size, type, input_dim, high, basic, padding, ngram, permutes):
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
        self.ngram_perm = ngram
        self.permutes = permutes

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
        import os
        cwd = os.getcwd()
        with open(str(cwd)+'/Makefile', 'w') as file:
            file.write('CC=clang' + '\n')
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
            file.write('#define SIZE ' + str(math.floor(self.input_dim / self.num_threads) * self.num_threads) + '\n')
            file.write('#define HIGH ' + str(self.high) + '\n')
            file.write('int CORRECT_PREDICTIONS;\n')

            file.write('struct DataReader {\n')
            file.write('    char *path;\n')
            file.write('    FILE *fp;\n')
            file.write('    char * line;\n')
            file.write('    size_t len;\n')
            file.write('    ssize_t read;\n')
            file.write('};\n')

            file.write('struct DataReader* train_data;\n')
            file.write('struct DataReader* train_labels;\n')
            file.write('struct DataReader* test_data;\n')
            file.write('struct DataReader* test_labels;\n')

            file.write('struct Task {\n')
            file.write('    float* data;\n')
            file.write('    int label;\n')
            file.write('};\n')

            if self.padding is not None:
                file.write('float padding = '+str(self.padding)+'.0;')
            else:
                file.write('float padding = -1.0;')

    def define_dataset_loaders(self):
        self.set_data_loaders()
        self.load_data()
        self.load_labels()
        self.load_dataset()
        self.close_files()

    def set_data_loaders(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
struct DataReader* set_load_data(char* path, struct DataReader* data_reader){
    data_reader = (struct DataReader *)calloc(1, sizeof(struct DataReader));
    data_reader -> path = path;
    data_reader -> fp = fopen(path, "r");
    data_reader -> line = NULL;
    data_reader -> len = 0;

    if (data_reader -> fp == NULL)
        exit(EXIT_FAILURE);

    return data_reader;
}
            '''
            )

    def load_data(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
float* load_data_next_line(struct DataReader* data_reader){
float* data = (float *) calloc(INPUT_DIM, sizeof(float));
char* token;
if ((data_reader -> read = getline(&data_reader -> line, &data_reader -> len, data_reader -> fp)) != -1) {
    token = strtok(data_reader -> line, ",");
    for (int i = 0; i < INPUT_DIM; i++){
      *(data + i) = (float) atof(token);
      token = strtok(NULL, ",");
    }
}
return data;
}
            '''
            )

    def load_labels(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int load_labels_next_line(struct DataReader* data_reader){
    int label;
    char* token;
    if ((data_reader -> read = getline(&data_reader -> line, &data_reader -> len, data_reader -> fp)) != -1) {
         label = atoi(data_reader -> line);
    }
    return label;
}
                '''
            )

    def load_dataset(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void prepare_to_load_data(char **argv){
    train_data = set_load_data(argv[1], train_data);
    train_labels = set_load_data(argv[2], train_labels);
    test_data = set_load_data(argv[3], test_data);
    test_labels = set_load_data(argv[4], test_labels);
}
                '''
            )

    def close_files(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void close_file(struct DataReader* data_reader){
    fclose(data_reader -> fp );
    if (data_reader -> line)
        free(data_reader -> line);
}
                '''
            )

    def define_math_functions(self):
        self.define_random_number()
        self.define_random_vector()
        self.define_weights()
        self.define_update_weights()
        self.define_update_correct_predictions()
        self.define_linear()
        self.define_norm2()
        self.define_normalize()
        self.define_argmax()
        self.define_map_range_clamp_one()
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
    CORRECT_PREDICTIONS += 1;
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

    def define_map_range_clamp_one(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void map_range_clamp_one(float* arr, float out_max, float* res){
    float in_min = 0;
    float in_max = HIGH;
    float out_min = 0;
    int i, j;
    for (j = 0; j < INPUT_DIM; j++){
        float map_range = round(out_min + (out_max - out_min) * (*(arr + j) - in_min) / (in_max - in_min));
        *(res + j) = min(max(map_range,out_min),out_max);
    }
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

    def multibind(self):
        """Binds a set of hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *multibind(f4si *a, f4si *b){
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
f4si* ngram(f4si* arr, f4si* enc, const int d){
    int i, j, k;
    f4si aux;
    f4si * forward_arr = calloc(DIMENSIONS * d, sizeof(int));
    f4si actual;
    float n[d];
    float p[d];
    for (i = 0; i < (INPUT_DIM-(d-1)); ++i){
        for (j = 0; j < NUM_BATCH; j++){
            for (k = 0; k < d; ++k){
                if (k == d-1){
                    actual = actual * forward_arr[k*NUM_BATCH+j];
                } else {
                       if (j == NUM_BATCH-1){
                           aux = shuffle(forward_arr[k*NUM_BATCH+j], (d-k-1));
                           aux[k] = p[k];
                       } else if (j == 0){
                           aux = shuffle(forward_arr[k*NUM_BATCH+j], (d-k-1));
                           p[k] = aux[k];
                       } else {
                           aux = shuffle(forward_arr[k*NUM_BATCH+j], (d-k-1));
                           n[k] = aux[k];
                           aux[k] = p[k];
                           p[k] = n[k];
                       }
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

    def generate_shuffle(self):
        """Ngram a set of hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(generate_shuffle(self.ngram_perm, self.vector_size))

    def ngram_forward(self):
        """Ngram a set of hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si* ngram_forward(f4si* arr, float* indices, f4si* enc, const int d){
    int i, j, k;
    f4si aux;
    f4si * forward_arr = calloc(DIMENSIONS * d, sizeof(int));
    f4si actual;
    float n[d];
    float p[d];
    for (i = 0; i < (INPUT_DIM-(d-1)); ++i){
        for (j = 0; j < NUM_BATCH; j++){
            for (k = 0; k < d; ++k){
                if (k == 0){
                    if (j == NUM_BATCH-1){
                        for (int m = 0; m < d; m++){
                            if (indices[i] != padding){
                                forward_arr[(m*NUM_BATCH)+j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                            } else {
                                forward_arr[(m*NUM_BATCH)+j] *= 0;
                            }
                        }
                       aux = shuffle(forward_arr[k*NUM_BATCH+j], (d-k-1));
                       aux[k] = p[k];
                   } else if (j == 0){
                        for (int m = 0; m < d; m++){
                            if (indices[i] != padding){
                                forward_arr[(m*NUM_BATCH)+j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                            } else {
                                forward_arr[(m*NUM_BATCH)+j] *= 0;
                            }
                        }
                       aux = shuffle(forward_arr[k*NUM_BATCH+j], (d-k-1));
                       p[k] = aux[k];

                   } else {
                        for (int m = 0; m < d; m++){
                            if (indices[i] != padding){
                                forward_arr[(m*NUM_BATCH)+j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                            } else {
                                forward_arr[(m*NUM_BATCH)+j] *= 0;
                            }
                        }
                       aux = shuffle(forward_arr[k*NUM_BATCH+j], (d-k-1));
                       n[k] = aux[k];
                       aux[k] = p[k];
                       p[k] = n[k];
                    }
                    actual = aux; 
                } else if (k == d-1){
                    actual = actual * forward_arr[k*NUM_BATCH+j];
                } else {
                       if (j == NUM_BATCH-1){
                           aux = shuffle(forward_arr[k*NUM_BATCH+j], (d-k-1));
                           aux[k] = p[k];
                       } else if (j == 0){
                           aux = shuffle(forward_arr[k*NUM_BATCH+j], (d-k-1));
                           p[k] = aux[k];
                       } else {
                           aux = shuffle(forward_arr[k*NUM_BATCH+j], (d-k-1));
                           n[k] = aux[k];
                           aux[k] = p[k];
                           p[k] = n[k];
                       }
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

    def multiset(self):
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

    def multiset_multibind_forward(self):
        """Bundles two hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *multiset_multibind_forward(f4si *a, f4si *b, float* indices, f4si* enc){
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

    def multiset_multibind(self):
        """Bundles two hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *multiset_multibind(f4si *a, f4si *b, f4si* enc){
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

    def multiset_forward(self):
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

    def forward(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *forward(f4si *a, float* indices, f4si* enc){
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
            enc[(NUM_BATCH * i) + j] = a[(int)indices[i]* NUM_BATCH + j];
        }
    }
    return enc;
}              
                ''')

    def permute(self):
        for i in self.permutes:
            with open(self.name.lower() + '.c', 'a') as file:
                file.write(
                    '''
f4si *permute''' + str(i) + '''(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float n[dd];
    float p[dd];

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j]''' + generate(
                        self.vector_size / 4, i) + ''');
           for (k = 0; k < dd; k++){
               res[i*NUM_BATCH+j][k] = p[k];
           }
       } else if (j == 0){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j]''' + generate(
                        self.vector_size / 4, i) + ''');
           for (k = 0; k < dd; k++){
               p[k] = res[i*NUM_BATCH+j][k];
           }
       } else {
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j]''' + generate(
                        self.vector_size / 4, i) + ''');
           for (k = 0; k < dd; k++){
               n[k] = res[i*NUM_BATCH+j][k];
               res[i*NUM_BATCH+j][k] = p[k];
               p[k] = n[k];
               n[k] = p[k];
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
        self.multibind()
        self.multibind_forward()
        self.multiset()
        self.multiset_forward()
        self.permute()
        self.multiset_multibind()
        self.multiset_multibind_forward()
        if self.ngram_perm != None:
            self.generate_shuffle()
            self.ngram()
            self.ngram_forward()
        self.define_encoding_function()

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

    def define_train_and_test(self):
        self.define_train_loop()
        self.define_test_loop()

    def define_train_loop(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void train_loop(){
    int i;
    for(i = 0; i < TRAIN; i++){
        struct Task *task = (struct Task *)calloc(1,sizeof(struct Task));
        task -> data = load_data_next_line(train_data);
        task -> label = load_labels_next_line(train_labels);
        encode_train_task(task);
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
    int i;
    for(i = 0; i < TEST; i++){
        struct Task *task = (struct Task *)calloc(1,sizeof(struct Task));
        task -> data = load_data_next_line(test_data);
        task -> label = load_labels_next_line(test_labels);
        encode_test_task(task);
    }
    return CORRECT_PREDICTIONS/(float)TEST;
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
    prepare_to_load_data(argv);
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