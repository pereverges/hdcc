import math

class IntermediateRepresentation:

    def __init__(self, name, classes, dimensions, vars, weight_var, encoding, embeddings, debug, encoding_fun,
                 encoding_fun_call):
        self.name = name
        self.classes = classes
        self.dimensions = dimensions
        self.vars = vars
        self.weight_var = weight_var
        self.encoding = encoding
        self.embeddings = embeddings
        self.debug = debug
        self.encoding_fun = encoding_fun
        self.encoding_fun_call = encoding_fun_call
        self.type = 'POOL'
        #self.train_size = 6238
        self.train_size = 60000
        #self.test_size = 1559
        self.test_size = 10000
        self.vector_size = 128
        self.num_threads = 4


    # ------------------- DEFINE MAKEFILE ------------------- #

    def _makefile(self):
        doto = '.o'
        with open('Makefile', 'w') as file:
            file.write('CC=gcc' + '\n')
            file.write('CFLAGS=-I.' + '\n')
            file.write(self.name + ': ' + self.name + doto + '\n')
            file.write('\t$(CC) -o ' + self.name + ' ' + self.name + doto + ' -lm -O3\n')

    # ------------------- AUXILIARY FUNCTIONS ------------------- #

    def random_number(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
float get_rand(float low_bound, float high_bound){
    return (float) ((float)rand() / (float)RAND_MAX) * (high_bound - low_bound) + low_bound;
}
                    '''
            )

    def random_vector(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
float *random_vector(int size, int dimensions, float low_bound, float high_bound){
   float *arr = (float *)malloc(size * dimensions * sizeof(float));
   int i, j;
   for (i = 0; i < size; i++){
      for (j = 0; j < dimensions; j++){
         *(arr + i*dimensions + j) = get_rand(low_bound, high_bound);
      }
   }
   return arr;
}
                    '''
            )

    def pthreads_vec_random_vector(self):
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
                ''')

    def uniform_sample_generator(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
float uniform_sample_generator(float low_bound, float high_bound){
  uint32_t* jsr;
  uint32_t seed = rand();
  jsr = &seed;
  uint32_t jsr_input;
  float value;
  jsr_input = *jsr;

  *jsr = ( *jsr ^ ( *jsr <<   13 ) );
  *jsr = ( *jsr ^ ( *jsr >>   17 ) );
  *jsr = ( *jsr ^ ( *jsr <<    5 ) );

  value = fmod ( 0.5
    + ( float ) ( jsr_input + *jsr ) / 65536.0 / 65536.0, 1.0 ) * (high_bound - low_bound) + low_bound;

  return value;
}                    
                '''
            )

    def uniform_generator(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
double *uniform_generator(int size, int dimensions, double low_bound, double high_bound){
   double *arr = (double *)malloc(size * dimensions * sizeof(double));
   int i, j;

   for (i = 0; i < size; i++){
      for (j = 0; j < dimensions; j++){
         *(arr + i*dimensions + j) = uniform_sample_generator(low_bound, high_bound);
      }
   }
   return arr;
}          
                '''
            )

    # ------------------- HDC FUNCTIONS ------------------- #

    def random_hv(self):
        """ Creates random hypervector """
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int *random_hv(int size){
   int *arr = (int *)malloc(size * DIMENSIONS * sizeof(int));
   int i, j;
   for (i = 0; i < size; i++){
      for (j = 0; j < DIMENSIONS; j++){
         *(arr + i*DIMENSIONS + j) = rand() % 100 > P ? 1 : -1;
      }
   }
   return arr;
}
            '''
            )

    def pthreads_vec_random_hv(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *random_hv(int size){
   f4si *arr = calloc(size * DIMENSIONS, sizeof(float));
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
                ''')

    def level_hv(self):
        """Creates level hypervector"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int *level_hv(int levels){
    int levels_per_span = levels-1;
    int span = 1;
    int *span_hv = random_hv(span+1);
    float *threshold_hv = random_vector(span,DIMENSIONS,0,1);
    int *hv = (int *)malloc(levels * DIMENSIONS * sizeof(int));

    int i, j;
    for(i = 0; i < levels; i++){
        float t = 1 - ((float)i / levels_per_span);
        for(j = 0; j < DIMENSIONS; j++){
            if((t > *(threshold_hv + j) || i == 0) && i != levels-1){
                *(hv + i*DIMENSIONS + j) = *(span_hv + j);
            } else {
                *(hv + i*DIMENSIONS + j) = *(span_hv + DIMENSIONS + j);
            }
        }
    }
    //free(span_hv);
    //free(threshold_hv);
    return hv;
}
                '''
            )

    def pthreads_vec_level_hv(self):
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
                ''')

    def bind(self):
        """Binds a set of hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int *bind(int *a, int *b, int size){
    int i, j;
    int *arr = (int *)malloc(size * DIMENSIONS * sizeof(int));
    for(i = 0; i < size; ++i){
        for(j = 0; j < DIMENSIONS; j++){
            *(arr + (DIMENSIONS*i) + j ) = *(a + (DIMENSIONS * i) + j) * *(b + (DIMENSIONS *i) + j);
        }
    }
    return arr;
}
                '''
            )

    def bundle(self):
        """Bundles two hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int *bundle(int *a, int *b){
    int i;
    int *arr = (int *)malloc(DIMENSIONS * sizeof(int));
    for(i = 0; i < DIMENSIONS; i++){
        *(arr + i) = *(a + i) + *(b + i);
    }
    return arr;
}
                '''
            )

    def multiset_bind(self):
        """Bundles two hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int *multiset_bind(int *a, int *b, float* indices, int size){
    int i, j;
    int *arr = (int *)malloc(size * DIMENSIONS * sizeof(int));
    for(i = 0; i < size; ++i){
        for(j = 0; j < DIMENSIONS; j++){
            *(arr + j ) += *(a + (DIMENSIONS * i) + j) * *(b + (int)*(indices + i) *DIMENSIONS + j);
        }
    }
    return arr;
}
                '''
            )

    def multiset(self):
        """Bundles a set of hypervectors together"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int *multiset(int *a, int size){
    int i, j;
    int *arr = (int *)calloc(size * DIMENSIONS, sizeof(int));
    for(i = 0; i < size; i++){
        for(j = 0; j < DIMENSIONS; ++j){
            *(arr + j) += *(a + i*DIMENSIONS + j);
        }
    }
    free(a);
    return arr;
}
                '''
            )

    def permute(self):
        """Permutes a hypervector"""
        # TODO: Implement
        pass

    # ------------------- CREATE ENCODING ------------------- #

    def forward(self):
        """Dot product between value and weight matrix"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int* forward(int* weights, float* indices, int size){
    int *res = (int *)malloc(size * DIMENSIONS * sizeof(int));
    int i, j;
    for(i = 0; i < size; i++){
       int* indx = (weights + (int)*(indices + i));
       for(j = 0; j < DIMENSIONS; j++){
           *(res + i*DIMENSIONS + j) = *(weights + (int)*(indices + i) *DIMENSIONS + j);
       }
    }
    return res;
}
                '''
            )

    def hard_quantize(self):
        """Set values to be either 1 or -1"""
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

    def generate_encode_function_pthread(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '\n' +
                self.encoding_fun + '''
f4si *encode_function(float* indices){
    SPLIT_SIZE = ceil(INPUT_DIM/NUM_THREADS);
    SIZE = SPLIT_SIZE*NUM_THREADS;
    pthread_t th[NUM_THREADS];
    int i;
    for (i = 0; i < NUM_THREADS; i++) {
        struct arg_struct *args = (struct arg_struct *)calloc(1, sizeof(struct arg_struct));
        args -> split_start = i*SPLIT_SIZE;
        args -> indices = indices;
        if (pthread_create(&th[i], NULL, &encode_fun, args) != 0) {
            perror("Failed to create thread");
        }
    }
    int j;
    f4si *res = calloc(DIMENSIONS,sizeof(int));
    for (i = 0; i < NUM_THREADS; i++) {
        f4si* r;
        if (pthread_join(th[i], (void**) &r) != 0) {
            perror("Failed to join thread");
        }
        for(j = 0; j < NUM_BATCH; j++){
            res[j] += r[j];
        }
        free(r);
    }
    return res;
}            
            ''')

    def generate_encoding_pthread(self):
        """Generates the encoding"""
        # TODO: generalize method, think how to set the dimensions depending of bind, bundle, and forward
        with open(self.name.lower() + '.c', 'a') as file:
            file.write('''
float* encoding(float* x){
    float* enc = (float*)encode_function(x);
    hard_quantize(enc,1);
    return enc;
}
                    '''
                       )

    def generate_encoding(self):
        """Generates the encoding"""
        # TODO: generalize method, think how to set the dimensions depending of bind, bundle, and forward
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '\n' +
                self.encoding_fun + '''
int* encoding(float* x){
    encode_function(x);
    hard_quantize(enc,1);
    return enc;
}
                    '''
            )

    def generate_encoding2(self):
        """Generates the encoding"""
        # TODO: generalize method, think how to set the dimensions depending of bind, bundle, and forward
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int* encoding(float* x){
    int* enc = multiset_bind(ID,''' + self.weight_var + ''',x,INPUT_DIM);
    hard_quantize(enc,1);
    return enc;
}
                        '''
            )

    # ------------------- MATH FUNCTIONS ------------------- #

    def argmax(self):
        """Returns the index of the biggest value in an array"""
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

    def map_range_clamp(self):
        """Sets the output value to be between min_out and max_out given min_in and max_in"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
float** map_range_clamp(float* arr[], int rows, int cols, float in_min, float in_max, float out_min, float out_max, float* res[]){
   int i, j;
   for (i = 0; i < rows; i++){
      for (j = 0; j < cols; j++){
        float map_range = round(out_min + (out_max - out_min) * (*(arr[i] + j) - in_min) / (in_max - in_min));
        *(res[i] + j) = min(max(map_range,out_min),out_max);
      }
      free(arr[i]);
   }
   return res;
}
                '''
            )

    # ------------------- NN FUNCTIONS ------------------- #

    def create_weights(self):
        """Creates an empty array (filled with zeros) for the wight matrix"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
float* weights(){
    float *arr = (float *)calloc(CLASSES * DIMENSIONS, sizeof(float));
    return arr;
}
                '''
            )

    def update_weights(self):
        """Updates the weights, learned from the encoded data"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void update_weight(float* weights, float* encoding, int feature){
    int i;
    for(i = 0; i < DIMENSIONS; i++){
        *(weights + feature*DIMENSIONS + i) += (float)*(encoding + i);
    }
}   
                '''
            )

    def linear(self):
        """Returns and array from the matrix multiplication of the encoding and the weight matrix"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
float* linear(float* m1, float* m2){
    int j, k;
    float *arr = (float *)calloc(CLASSES, sizeof(float));
    for (j = 0; j < DIMENSIONS; ++j) {
      for (k = 0; k < CLASSES; ++k) {
         *(arr + k) += (float)*(m1 + j) * *(m2 + k*DIMENSIONS + j);
      }
   }
    return arr;
}           
                '''
            )

    def norm2(self):
        """Norm2 function"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''          
float norm2(float* weight,int feature){
   float norm = 0.0;
   int i;
   for (i = 0; i < DIMENSIONS; i++){
      norm += *(weight + feature*DIMENSIONS + i) * *(weight + feature*DIMENSIONS + i);
   }
   return sqrt(norm);
}
                '''
            )

    def normalize(self):
        """Normalizes the values of the weight matrix"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void normalize(float* weight){
   float eps = 1e-12;
   int i, j;
   for (i = 0; i < CLASSES; i++){
      float norm = norm2(weight,i);
      for (j = 0; j < DIMENSIONS; j++){
        *(weight + i*DIMENSIONS + j) = *(weight + i*DIMENSIONS + j) / max(norm,eps);
      }
   }
}
                '''
            )

    # ------------------- TRAIN AND TEST ------------------- #

    def train(self):
        """Loop for the training"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void train_loop(float* train[], float* label[], float* classify, int size){
    float *res[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        res[i] = (float *)malloc(INPUT_DIM * sizeof(float));
    }
    map_range_clamp(train,TRAIN,INPUT_DIM,0,1,0,''' + self.weight_var + '_DIM' + '''-1,res);
    int i;
    for(i = 0; i < size; i++){
        float* enc = encoding(res[i]);
        update_weight(classify,enc,*(label[i]));
        free(res[i]);
        free(enc);
    }
    normalize(classify);
}                
                '''
            )

    def test(self):
        """Loop for the testing"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
float test_loop(float* test[], float* label[], float* classify, int size){
    float *res[TEST];
    for (int i = 0; i < TEST; i++){
        res[i] = (float *)calloc(INPUT_DIM, sizeof(float));
    }
    map_range_clamp(test,TEST,INPUT_DIM,0,1,0,''' + self.weight_var + '_DIM' + '''-1,res);
    int i;
    int correct_pred = 0;
    for(i = 0; i < size; i++){
        float* enc = encoding(res[i]);
        float *l = linear(enc,classify);
        int index = argmax(l);
        if((int)index == (int)*(label[i])){
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

    # ------------------- LOAD DATA ------------------- #

    def swap(self):
        """MNIST load data"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
static int32_t swap(int32_t x){
	union { int32_t i; char b[4]; } in, out;
	in.i = x;
	out.b[0] = in.b[3];
	out.b[1] = in.b[2];
	out.b[2] = in.b[1];
	out.b[3] = in.b[0];
	return out.i;
}                
                '''
            )

    def load_images(self):
        """MNIST load_images"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
static uint8_t *load_images(const char *pathname, int *n){
	int32_t meta[4];
	uint8_t *data;
	FILE *file;

	file = fopen(pathname, "r");
	if (!file) {
		fprintf(stderr, "unable to open file\\n");
		return 0;
	}
	if (sizeof (meta) != fread(meta, 1, sizeof (meta), file)) {
		fclose(file);
		fprintf(stderr, "unable to read file\\n");
		return 0;
	}
	if ((0x3080000 != meta[0]) ||
	    (0  >= swap(meta[1])) ||
	    (28 != swap(meta[2])) ||
	    (28 != swap(meta[3]))) {
		fclose(file);
		fprintf(stderr, "invalid file\\n");
		return 0;
	}
	(*n) = swap(meta[1]);
	meta[1] = (*n) * 28 * 28;
	data = (uint8_t *)malloc(meta[1]);
	if (!data) {
		fclose(file);
		fprintf(stderr, "out of memory\\n");
		return 0;
	}
	if ((size_t)meta[1] != fread(data, 1, meta[1], file)) {
		free(data);
		fclose(file);
		fprintf(stderr, "unable to read file\\n");
		return 0;
	}
	fclose(file);
	return data;
}                
                '''
            )

    def load_labels_mnist(self):
        """MNIST load labels"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''      
static uint8_t *load_labels(const char *pathname, int *n){
	int32_t meta[2];
	uint8_t *data;
	FILE *file;

	file = fopen(pathname, "r");
	if (!file) {
		fprintf(stderr, "unable to open file\\n");
		return 0;
	}
	if (sizeof (meta) != fread(meta, 1, sizeof (meta), file)) {
		fclose(file);
		fprintf(stderr, "unable to read file\\n");
		return 0;
	}
	if ((0x1080000 != meta[0]) || (0 >= swap(meta[1]))) {
		fclose(file);
		fprintf(stderr, "invalid file\\n");
		return 0;
	}
	(*n) = swap(meta[1]);
	meta[1] = (*n);
	data = (uint8_t *)malloc(meta[1]);
	if (!data) {
		fclose(file);
		fprintf(stderr, "out of memory\\n");
		return 0;
	}
	if ((size_t)meta[1] != fread(data, 1, meta[1], file)) {
		free(data);
		fclose(file);
		fprintf(stderr, "unable to read file\\n");
		return 0;
	}
	fclose(file);
	return data;
}
                '''
            )

    def to_float(self):
        """Sets values to float"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''  
float* to_float(uint8_t* arr, int rows, int cols){
   float *res = (float *)malloc(rows * cols * sizeof(float));
   int i, j;
   for (i = 0; i < rows; i++) {
      for (j = 0; j < cols; j++) {
         *(res + i*cols + j) = (float) *(arr + i*cols + j)/255;
      }
   }
   return res;
}                
                '''
            )

    def to_int(self):
        """Sets values to int"""
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''  
int* to_int(uint8_t* arr, int rows, int cols){
   int *res = (int *)malloc(rows * cols * sizeof(int));
   int i, j;
   for (i = 0; i < rows; i++) {
      for (j = 0; j < cols; j++) {
         *(res + i*cols + j) = (int) *(arr + i*cols + j);
      }
   }
   return res;
}                
                '''
            )

    def load_dataset_voicehd(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void load_dataset(float* trainx, int* trainy, float* testx, int* testy){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen("data/ISOLET/isolet1+2+3+4.data", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    char* token;
    int count = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        token = strtok(line, ", ");
        for (int i = 0; i < INPUT_DIM; i++){
          *(trainx + count * INPUT_DIM + i) = (float) atof(token);
          token = strtok(NULL, ", ");
        }
        *(trainy + count) = atoi(token);
        count += 1;
    }
    fclose(fp);
    if (line)
        free(line);

    line = NULL;
    len = 0;

    fp = fopen("data/ISOLET/isolet5.data", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    count = 0;

    while ((read = getline(&line, &len, fp)) != -1) {

        token = strtok(line, ", ");

        for (int i = 0; i < INPUT_DIM; i++){
          *(testx + count * INPUT_DIM + i) = atof(token);
          token = strtok(NULL, ", ");
        }
        *(testy + count) = atoi(token);
        count += 1;
    }

    fclose(fp);
    if (line)
        free(line);

}               
                '''
            )

    def _load_data(self):
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
        free(line);
}
            '''
            )

    def _load_labels(self):
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

    def load_data_mnist(self):
        # TODO: Create function to load data in general
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void load_dataset(float** trainx, int** trainy, float** testx, int** testy){
    int train_y_n, train_x_n, test_y_n, test_x_n;
	uint8_t *train_y, *train_x, *test_y, *test_x;

    train_x = load_images("data/MNIST/raw/train-images-idx3-ubyte", &train_x_n);
	test_x = load_images("data/MNIST/raw/t10k-images-idx3-ubyte", &test_x_n);
	train_y = load_labels("data/MNIST/raw/train-labels-idx1-ubyte", &train_y_n);
	test_y = load_labels("data/MNIST/raw/t10k-labels-idx1-ubyte", &test_y_n);

    *trainx = to_float(train_x, TRAIN,INPUT_DIM);
    *trainy = to_int(train_y,TRAIN,1);

    *testx = to_float(test_x, TEST,INPUT_DIM);
    *testy = to_int(test_y,TEST,1);
}
                '''
            )

    # ------------------- DEFINE HEADERS ------------------- #

    def define_embeddings(self):
        embedding = ''
        for i in self.embeddings:
            if i[0] == 'LEVEL':
                if i[1] != self.weight_var:
                    embedding += ("\n    " + str(i[1].upper() + " = level_hv(" + 'INPUT_DIM' + ");"))
                else:
                    embedding += ("\n    " + str(i[1].upper() + " = level_hv(" + str(i[1]) + '_DIM' + ");"))
            if i[0] == 'RANDOM':
                if i[1] != self.weight_var:
                    embedding += ("\n    " + str(i[1].upper() + " = random_hv(" + 'INPUT_DIM' + ");"))
                else:
                    embedding += ("\n    " + str(i[1].upper() + " = random_hv(" + str(i[1]) + '_DIM' + ");"))
        return embedding

    def define_headers(self):
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

    def define_headers_pthreads(self):
        with open(self.name.lower() + '.c', 'w') as file:
            file.write('/*********** ' + self.name + ' ***********/\n')
            file.write('#include <stdio.h>\n')
            file.write('#include <stdlib.h>\n')
            file.write('#include <stdint.h>\n')
            file.write('#include <string.h>\n')
            file.write('#include <math.h>\n')
            file.write('#include <pthread.h>\n')
            if self.debug:
                file.write('#include <time.h>\n')
            file.write('#ifndef max\n')
            file.write('#define max(a,b) (((a) > (b)) ? (a) : (b))\n')
            file.write('#endif\n')
            file.write('#ifndef min\n')
            file.write('#define min(a,b) (((a) < (b)) ? (a) : (b))\n')
            file.write('#endif\n')

    def define_constants_pthreads(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write('/*********** CONSTANTS ***********/\n')
            file.write('int TRAIN;\n')
            file.write('int TEST;\n')
            file.write('int SIZE = ' + str(1) + ';\n')
            file.write('int P = ' + str(50) + ';\n')
            file.write('int DIMENSIONS = ' + str(self.dimensions) + ';\n')
            file.write('int CLASSES = ' + str(self.classes) + ';\n')
            file.write('int S = 256;\n')
            file.write('typedef float f4si __attribute__ ((vector_size (256)));\n')
            file.write('int BATCH;\n')
            file.write('int NUM_BATCH;\n')
            file.write('int NUM_THREADS = 4;\n')
            file.write('int SPLIT_SIZE;\n')
            file.write('int SIZE;\n')
            for i in self.embeddings:
                file.write('f4si* ' + str(i[1]) + ';\n')
                if i[1] != self.weight_var:
                    file.write('int INPUT_DIM = ' + str(i[2]) + ';\n')
                else:
                    file.write('int ' + str(i[1]) + '_DIM = ' + str(i[2]) + ';\n')
            file.write('\n')
            file.write('struct arg_struct {\n')
            file.write('    int split_start;\n')
            file.write('    float* indices;\n')
            file.write('};\n')

    def define_constants(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write('/*********** CONSTANTS ***********/\n')
            file.write('int TRAIN;\n')  # ' + str(6238) + ';\n')
            # file.write('int TRAIN = ' + str(60000) + ';\n')
            file.write('int TEST;\n')  # = ' + str(1559) + ';\n')
            # file.write('int TEST = ' + str(10000) + ';\n')
            file.write('int SIZE = ' + str(1) + ';\n')
            file.write('int P = ' + str(50) + ';\n')
            file.write('int DIMENSIONS = ' + str(self.dimensions) + ';\n')
            file.write('int CLASSES = ' + str(self.classes) + ';\n')
            for i in self.embeddings:
                file.write('int* ' + str(i[1]) + ';\n')
                if i[1] != self.weight_var:
                    file.write('int INPUT_DIM = ' + str(i[2]) + ';\n')
                else:
                    file.write('int ' + str(i[1]) + '_DIM = ' + str(i[2]) + ';\n')

    def define_globals(self):
        self.define_constants()

    # ------------------- MAIN ------------------- #

    def load_auxiliary(self):
        self.random_number()
        self.pthreads_vec_random_vector()
        # self.uniform_sample_generator()
        # self.uniform_generator()

    def run(self):
        self.makefile_thread_pool()
        self.define_header()
        self.define_dataset_loaders()
        self.define_math_functions()
        self.define_hdc_functions()
        self.define_train_and_test()
        self.general_main()

    def run2(self):
        self.makefile()
        self.define_headers_pthreads()
        self.define_constants_pthreads()
        self.load_auxiliary()

        self.load_data()
        self.load_labels()

        self.pthreads_vec_random_hv()
        self.pthreads_vec_level_hv()

        self.create_weights()
        self.update_weights()
        self.linear()
        self.norm2()
        self.normalize()
        self.argmax()
        self.map_range_clamp()
        self.hard_quantize()
        self.generate_encode_function_pthread()
        self.generate_encoding_pthread()

        self.train()
        self.test()

        if self.debug:
            self.general_main_debug_pthread()
        else:
            self.general_main_pthread()

    def run1(self):
        self.makefile()
        self.define_headers()
        self.define_globals()
        self.load_auxiliary()

        # self.swap()
        # self.load_labels()
        # self.load_images()
        # self.to_int()
        # self.to_float()
        # self.load_data_mnist()
        # self.load_dataset_voicehd()
        self.load_data()
        self.load_labels()

        self.random_hv()
        self.level_hv()
        # self.bind()
        # self.bundle()
        # self.multiset()

        self.create_weights()
        self.update_weights()
        self.linear()
        self.norm2()
        self.normalize()
        self.argmax()
        # self.forward()
        self.map_range_clamp()
        self.hard_quantize()
        self.generate_encoding()

        self.train()
        self.test()

        if self.debug:
            self.debug_main()
        else:
            self.general_main()

    def mnist(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
    int main() {
    
    float *WEIGHT = weights();
    '''
                +
                str(self.define_embeddings())
                +
                '''
    
    float *train_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_data[i] = (float *)malloc(INPUT_DIM * sizeof(float));
    }

    int *train_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_labels[i] = (int *)malloc(1 * sizeof(int));
    }

    float *test_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_data[i] = (float *)malloc(INPUT_DIM * sizeof(float));
    }

    int *test_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_labels[i] = (int *)malloc(1 * sizeof(int));
    }
    
    load_dataset(&trainx, &trainy, &testx, &testy);
    
    train_loop(trainx, trainy, WEIGHT,TRAIN);
    float acc = test_loop(testx,testy,WEIGHT,TEST);
    printf("acc %f ", acc);
    }
                '''
            )

    def voicehd(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int main() {
    srand(42);
    
    float *WEIGHT = weights();
    '''
                +
                str(self.define_embeddings())
                +
                '''
            
        
    float *trainx = (float *)malloc(TRAIN * INPUT_DIM * sizeof(float));
    float *testx = (float *)malloc(TEST * INPUT_DIM * sizeof(float));
    int *trainy = (int *)malloc(TRAIN * 1 * sizeof(int));
    int *testy = (int *)malloc(TEST * 1 * sizeof(int));

    load_dataset(trainx, trainy, testx, testy);

    train_loop(trainx, trainy, WEIGHT,TRAIN);
    float acc = test_loop(testx,testy,WEIGHT,TEST);
    printf("acc %f ", acc);

}                
                '''
            )

    def general_main_pthread(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int main(int argc, char **argv) {
    BATCH = (int) S/sizeof(float);
    NUM_BATCH = (int) ceil(DIMENSIONS/BATCH);

    TRAIN = atoi(argv[1]);
    TEST = atoi(argv[2]);
    float *WEIGHT = weights();
    '''
                +
                str(self.define_embeddings())
                +
                '''
    float *train_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_data[i] = (float *)calloc(INPUT_DIM, sizeof(float));
    }

    float *train_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_labels[i] = (float *)calloc(1, sizeof(int));
    }

    float *test_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_data[i] = (float *)calloc(INPUT_DIM, sizeof(float));
    }

    float *test_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_labels[i] = (float *)calloc(1, sizeof(int));
    }

    load_data(train_data, argv[3]);
    load_data(test_data, argv[5]);
    load_label(train_labels, argv[4]);
    load_label(test_labels, argv[6]);

    train_loop(train_data, train_labels, WEIGHT,TRAIN);
    float acc = test_loop(test_data,test_labels,WEIGHT,TEST);
    printf("Accuracy: %f ", acc);  
}              
                '''
            )

    def general_main_debug_pthread(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int main(int argc, char **argv) {
    BATCH = (int) S/sizeof(float);
    NUM_BATCH = (int) ceil(DIMENSIONS/BATCH);

    TRAIN = atoi(argv[1]);
    TEST = atoi(argv[2]);
    float *WEIGHT = weights();
    '''
                +
                str(self.define_embeddings())
                +
                '''
    float *train_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_data[i] = (float *)calloc(INPUT_DIM, sizeof(float));
    }

    float *train_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_labels[i] = (float *)calloc(1, sizeof(int));
    }

    float *test_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_data[i] = (float *)calloc(INPUT_DIM, sizeof(float));
    }

    float *test_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_labels[i] = (float *)calloc(1, sizeof(int));
    }

    load_data(train_data, argv[3]);
    load_data(test_data, argv[5]);
    load_label(train_labels, argv[4]);
    load_label(test_labels, argv[6]);

    struct timespec begin, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    train_loop(train_data, train_labels, WEIGHT,TRAIN);
    float acc = test_loop(test_data,test_labels,WEIGHT,TEST);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("%d, %f, %f \\n", DIMENSIONS,elapsed, acc);
}              
                '''
            )

    def _general_main(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int main(int argc, char **argv) {
    TRAIN = atoi(argv[1]);
    TEST = atoi(argv[2]);
    float *WEIGHT = weights();
    '''
                +
                str(self.define_embeddings())
                +
                '''
    float *train_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_data[i] = (float *)malloc(INPUT_DIM * sizeof(float));
    }

    int *train_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_labels[i] = (int *)malloc(1 * sizeof(int));
    }

    float *test_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_data[i] = (float *)malloc(INPUT_DIM * sizeof(float));
    }

    int *test_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_labels[i] = (int *)malloc(1 * sizeof(int));
    }
    
    load_data(train_data, argv[3]);
    load_data(test_data, argv[5]);
    load_label(train_labels, argv[4]);
    load_label(test_labels, argv[6]);
    
    train_loop(train_data, train_labels, WEIGHT,TRAIN);
    float acc = test_loop(test_data,test_labels,WEIGHT,TEST);
    printf("acc %f ", acc);  
}              
                '''
            )

    def debug_main(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int main(int argc, char **argv) {
    TRAIN = atoi(argv[1]);
    TEST = atoi(argv[2]);
    float *WEIGHT = weights();
    '''
                +
                str(self.define_embeddings())
                +
                '''
    float *train_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_data[i] = (float *)malloc(INPUT_DIM * sizeof(float));
    }

    int *train_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        train_labels[i] = (int *)malloc(1 * sizeof(int));
    }

    float *test_data[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_data[i] = (float *)malloc(INPUT_DIM * sizeof(float));
    }

    int *test_labels[TRAIN];
    for (int i = 0; i < TRAIN; i++){
        test_labels[i] = (int *)malloc(1 * sizeof(int));
    }

    load_data(train_data, argv[3]);
    load_data(test_data, argv[5]);
    load_label(train_labels, argv[4]);
    load_label(test_labels, argv[6]);

    clock_t t;
    t = clock();
    printf("Start\\n");
    train_loop(train_data, train_labels, WEIGHT,TRAIN);
    float acc = test_loop(test_data,test_labels,WEIGHT,TEST);
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // calculate the elapsed time
    printf("The program took %f seconds to execute\\n", time_taken);
    printf("acc %f \\n", acc);
}              
                '''
            )

    # ------------------- DEFINE MAKEFILE ------------------- #

    def makefile_thread_pool(self):
        dotc = '.c'
        with open('Makefile', 'w') as file:
            file.write('CC=gcc' + '\n')
            file.write('all: thread_pool.c thread_pool.h ' + self.name + dotc + '\n')
            file.write('\t$(CC) thread_pool.c ' + self.name + dotc + ' -lpthread -lm -O3 -o ' + self.name + '\n')

    def makefile(self):
        if self.type == 'POOL':
            self.makefile_thread_pool()

    # ------------------- DEFINE HEADER ------------------- #

    def define_header(self):
        if self.type == 'POOL':
            self.define_header_thread_pool()

    def define_header_thread_pool(self):
        self.define_include_thread_pool()
        self.define_constants_thread_pool()

    def define_include_thread_pool(self):
        with open(self.name.lower() + '.c', 'w') as file:
            file.write('/*********** ' + self.name + ' ***********/\n')
            file.write('#include "thread_pool.h"\n')
            file.write('#include <stdio.h>\n')
            file.write('#include <stdlib.h>\n')
            file.write('#include <stdint.h>\n')
            file.write('#include <string.h>\n')
            file.write('#include <math.h>\n')
            file.write('#include <pthread.h>\n')
            if self.debug:
                file.write('#include <time.h>\n')
            file.write('#ifndef max\n')
            file.write('#define max(a,b) (((a) > (b)) ? (a) : (b))\n')
            file.write('#endif\n')
            file.write('#ifndef min\n')
            file.write('#define min(a,b) (((a) < (b)) ? (a) : (b))\n')
            file.write('#endif\n')

    def define_constants_thread_pool(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write('/*********** CONSTANTS ***********/\n')
            file.write('#define TRAIN ' + str(self.train_size) + '\n')
            file.write('#define TEST ' + str(self.test_size) + '\n')

            file.write('#define DIMENSIONS ' + str(self.dimensions) + '\n')
            file.write('#define CLASSES ' + str(self.classes) + '\n')
            file.write('#define VECTOR_SIZE ' + str(self.vector_size) + '\n')
            file.write('float *WEIGHT;\n')
            file.write('typedef float f4si __attribute__ ((vector_size (' + str(self.vector_size) + ')));\n')

            input_dim = 0
            for i in self.embeddings:
                file.write('f4si* ' + str(i[1]) + ';\n')
                if i[1] != self.weight_var:
                    file.write('#define INPUT_DIM ' + str(i[2]) + '\n')
                    input_dim = i[2]
                else:
                    file.write('#define ' + str(i[1]) + '_DIM ' + str(i[2]) + '\n')

            file.write('#define BATCH ' + str(int(self.vector_size/4)) + '\n')
            file.write('#define NUM_BATCH ' + str(int(self.dimensions/(self.vector_size/4))) + '\n')
            file.write('#define NUM_THREADS ' + str(self.num_threads) + '\n')
            file.write('#define SPLIT_SIZE ' + str(math.floor(input_dim/self.num_threads)) + '\n')
            file.write('#define SIZE ' + str(math.floor(input_dim/self.num_threads)*self.num_threads) + '\n')

            file.write('float *TRAIN_DATA[TRAIN];\n')
            file.write('float *TRAIN_LABELS[TRAIN];\n')
            file.write('float *TEST_DATA[TEST];\n')
            file.write('float *TEST_LABELS[TEST];\n')
            file.write('ThreadPool *pool;\n')

            file.write('struct EncodeTask {\n')
            file.write('    int split_start;\n')
            file.write('    float* indices;\n')
            file.write('    f4si *res;\n')
            file.write('};\n')

    # ------------------- DEFINE DATA LOADERS ------------------- #

    def define_dataset_loaders(self):
        if self.type == 'POOL':
            self.define_dataset_loader_thread_pool()

    def define_dataset_loader_thread_pool(self):
        self.load_data_thread_pool()
        self.load_labels_thread_pool()
        self.load_dataset_thread_pool()

    def load_data_thread_pool(self):
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
        free(line);
}
            '''
            )

    def load_labels_thread_pool(self):
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

    def load_dataset_thread_pool(self):
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

    load_data(TRAIN_DATA, argv[3]);
    load_data(TEST_DATA, argv[5]);
    load_label(TRAIN_LABELS, argv[4]);
    load_label(TEST_LABELS, argv[6]);
}
                '''
            )

    # ------------------- DEFINE MATH FUNCTIONS ------------------- #

    def define_math_functions(self):
        if self.type == 'POOL':
            self.define_math_functions_thread_pool()

    def define_math_functions_thread_pool(self):
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

    # ------------------- DEFINE HDC FUNCTIONS ------------------- #

    def define_hdc_functions(self):
        if self.type == 'POOL':
            self.define_hdc_functions_thread_pool()

    def define_hdc_functions_thread_pool(self):
        self.define_random_hv()
        self.define_level_hv()
        self.define_encoding_function_thread_pool()
        self.define_encoding_thread_pool()

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

    def define_encoding_function_thread_pool(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(self.encoding_fun)

    def define_encoding_thread_pool(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
f4si *encode_function(float* indices){
    struct EncodeTask *task = (struct EncodeTask *)malloc(sizeof(struct EncodeTask));
    f4si *res = calloc(DIMENSIONS,sizeof(int));
    for (int i = 0; i < NUM_THREADS; i++) {
        struct EncodeTask *task = (struct EncodeTask *)malloc(sizeof(struct EncodeTask));
        task -> split_start = i*SPLIT_SIZE;
        task -> indices = indices;
        task -> res = res;
        mt_add_job(pool, &encode_fun, task);
    }
    return res;
}

float* encodings(float* x){
    float* enc = (float*)encode_function(x);
    hard_quantize(enc,1);
    return enc;
}
                ''')

    # ------------------- DEFINE TRAIN AND TEST ------------------- #

    def define_train_and_test(self):
        if self.type == 'POOL':
            self.define_train_and_test_thread_pool()

    def define_train_and_test_thread_pool(self):
        self.define_train_loop_thread_pool()
        self.define_test_loop_thread_pool()

    def define_train_loop_thread_pool(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void train_loop(){
    float *res[TRAIN];
    float *enc[TRAIN];
    int i;
    for (i = 0; i < TRAIN; i++){
        res[i] = (float *)calloc(INPUT_DIM, sizeof(float));
        enc[i] = (float *)calloc(DIMENSIONS, sizeof(float));
    }
    map_range_clamp(TRAIN_DATA,TRAIN,''' + self.weight_var + '''_DIM-1, res);
    for(i = 0; i < TRAIN; i++){
        enc[i] = encodings(res[i]);
    }
    mt_join(pool);
    for(i = 0; i < TRAIN; i++){
        update_weight(enc[i],*(TRAIN_LABELS[i]));
        free(res[i]);
        free(enc[i]);
    }
    normalize();
}
                    '''
                )

    def define_test_loop_thread_pool(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
float test_loop(){
    float *res[TEST];
    float *enc[TEST];
    int i;
    for (i = 0; i < TEST; i++){
        res[i] = (float *)calloc(INPUT_DIM, sizeof(float));
        enc[i] = (float *)calloc(DIMENSIONS, sizeof(float));
    }
    map_range_clamp(TEST_DATA,TEST,''' + self.weight_var + '''_DIM-1, res);
    int correct_pred = 0;
    for(i = 0; i < TEST; i++){
        enc[i] = encodings(res[i]);
    }
    mt_join(pool);
    for(i = 0; i < TEST; i++){
        float *l = linear(enc[i]);
        int index = argmax(l);
        if((int)index == (int)*(TEST_LABELS[i])){
            correct_pred += 1;
        }
        free(l);
        free(res[i]);
        free(enc[i]);
    }
    return correct_pred/(float)TEST;
}
                    '''
                )

    # ------------------- DEFINE TRAIN AND TEST ------------------- #

    def general_main(self):
        if self.type == 'POOL':
            self.general_main_thread_pool()

    def general_main_thread_pool(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int main(int argc, char **argv) {
    '''
    +
    str(self.define_embeddings())
    +
    '''
	pool = mt_create_pool(NUM_THREADS);
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
    printf("%d, %f, %f \\n", DIMENSIONS,elapsed, acc);
                    '''
                )
            else:
                file.write(
                    '''
        printf("%d, %f \\n", DIMENSIONS, acc);
                    '''
                )

            file.write(
                '''
    }              
                '''
                )