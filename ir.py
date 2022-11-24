class IntermediateRepresentation:

    def __init__(self, name, classes, dimensions, vars, weight_var, encoding, embeddings):
        self.name = name
        self.classes = classes
        self.dimensions = dimensions
        self.vars = vars
        self.weight_var = weight_var
        self.encoding = encoding
        self.embeddings = embeddings

    # ------------------- DEFINE MAKEFILE ------------------- #

    def makefile(self):
        doto = '.o'
        with open('Makefile', 'w') as file:
            file.write('CC=gcc' + '\n')
            file.write('CFLAGS=-I.' + '\n')
            file.write(self.name + ': ' + self.name + doto + '\n')
            file.write('\t$(CC) -o ' + self.name + ' ' + self.name + doto + ' -lm\n')

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
    //free(a);
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
void hard_quantize(int *arr, int size){
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

    def generate_encoding(self):
        """Generates the encoding"""
        # TODO: generalize method, think how to set the dimensions depending of bind, bundle, and forward
        with open(self.name.lower() + '.c', 'a') as file:

            file.write(
                '''
int* encoding(float* x){
    int* f = forward(''' + self.weight_var+ ''',x,INPUT_DIM); 
    int* enc = ''' + self.encoding + ''';
    //free(f);
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
float* map_range_clamp(float* arr, int rows, int cols, float in_min, float in_max, float out_min, float out_max){
   float *res = (float *)malloc(rows * cols * sizeof(float));
   int i, j;
   for (i = 0; i < rows; i++){
      for (j = 0; j < cols; j++){
        float map_range = round(out_min + (out_max - out_min) * (*(arr + i*cols + j) - in_min) / (in_max - in_min));
        *(res + i*cols + j) = min(max(map_range,out_min),out_max);
      }
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
void update_weight(float* weights, int* encoding, int feature){
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
float* linear(int* m1, float* m2, int s1){
    int i, j, k;
    float *arr = (float *)calloc(s1 * CLASSES, sizeof(float));
    for (i = 0; i < s1; ++i) {
       for (j = 0; j < CLASSES; ++j) {
          for (k = 0; k < DIMENSIONS; ++k) {
             *(arr + i*s1 + j) += (float)*(m1 + i*DIMENSIONS + k) * *(m2 + j*DIMENSIONS + k);
          }
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
   int i,j;
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
void train_loop(float* train, int* label, float* classify, int size){
    train = map_range_clamp(train,TRAIN,INPUT_DIM,0,1,0,''' + self.weight_var+'_DIM' + '''-1);
    int i;
    for(i = 0; i < size; i++){
        int* enc = encoding((train + i*INPUT_DIM));
        update_weight(classify,enc,*(label + i));
        //free(enc);
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
float test_loop(float* test, int* label, float* classify, int size){
    test = map_range_clamp(test,TEST,INPUT_DIM,0,1,0,''' + self.weight_var+'_DIM' + '''-1);
    int i;
    int correct_pred = 0;
    for(i = 0; i < size; i++){
        int* enc = encoding((test + i*INPUT_DIM));
        float *l = linear(enc,classify,CLASSES);
        //free(enc);
        int index = argmax(l);
        if((int)index == (int)*(label+i)){
            correct_pred += 1;
        }
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


    def load_data(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void load_data(float* data, char* path){
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
          *(data + count * INPUT_DIM + i) = (float) atof(token);
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

    def load_labels(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
void load_label(int* data, char* path){
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
        *(data + count) = atoi(line);
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
            file.write('#ifndef max\n')
            file.write('#define max(a,b) (((a) > (b)) ? (a) : (b))\n')
            file.write('#endif\n')
            file.write('#ifndef min\n')
            file.write('#define min(a,b) (((a) < (b)) ? (a) : (b))\n')
            file.write('#endif\n')

    def define_constants(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write('/*********** CONSTANTS ***********/\n')
            file.write('int TRAIN;\n') # ' + str(6238) + ';\n')
            #file.write('int TRAIN = ' + str(60000) + ';\n')
            file.write('int TEST;\n') #= ' + str(1559) + ';\n')
            #file.write('int TEST = ' + str(10000) + ';\n')
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
        self.random_vector()
        #self.uniform_sample_generator()
        #self.uniform_generator()

    def run(self):
        self.makefile()
        self.define_headers()
        self.define_globals()
        self.load_auxiliary()

        #self.swap()
        #self.load_labels()
        #self.load_images()
        #self.to_int()
        #self.to_float()
        #self.load_data_mnist()
        #self.load_dataset_voicehd()
        self.load_data()
        self.load_labels()

        self.random_hv()
        self.level_hv()
        self.bind()
        self.bundle()
        self.multiset()

        self.create_weights()
        self.update_weights()
        self.linear()
        self.norm2()
        self.normalize()
        self.argmax()
        self.forward()
        self.map_range_clamp()
        self.hard_quantize()
        self.generate_encoding()

        self.train()
        self.test()

        self.general_main()




    def mnist(self):
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
    
    float *trainx, *testx;
    int *trainy, *testy;
    
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

    def general_main(self):
        with open(self.name.lower() + '.c', 'a') as file:
            file.write(
                '''
int main(int argc, char **argv) {
    srand(42);
    TRAIN = atoi(argv[1]);
    TEST = atoi(argv[2]);
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
    
    load_data(trainx, argv[3]);
    load_data(testx, argv[5]);
    load_label(trainy, argv[4]);
    load_label(testy, argv[6]);
    
    train_loop(trainx, trainy, WEIGHT,TRAIN);
    float acc = test_loop(testx,testy,WEIGHT,TEST);
    printf("acc %f ", acc);  
}              
                '''
            )
