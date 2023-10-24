import math
from execution_type import Types
from ir_parallel import ParallelRepresentation
from ir_sequential import SequentialRepresentation

class IntermediateRepresentation:
    def __init__(self, name, classes, dimensions, vars, weight_var, encoding, embeddings, debug, encoding_fun,
                 train_size, test_size, num_threads, vector_size, type, input_dim, high, basic, padding, permutes,\
                 ngram, path, not_multiset, vectorial, performance):
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
        self.permutes = permutes
        self.ngram = ngram
        self.path = path
        self.not_multiset = not_multiset
        self.vectorial = vectorial
        self.performance = performance

    def get_basic_name(self, name):
        temp = len(name)
        for c in name:
            if c.isdigit():
                temp = name.index(c)
                break
        return name[0:temp]

    def run(self):
        if self.type == Types.SEQUENTIAL:
            irSec = SequentialRepresentation(self.name, self.classes, self.dimensions, self.vars, self.weight_var, self.encoding, self.embeddings, self.debug, self.encoding_fun,
                 self.train_size, self.test_size, self.num_threads, self.vector_size, self.type, self.input_dim, self.high, self.basic, self.padding, self.ngram, self.permutes, self.vectorial, self.performance)
            irSec.run_sequential()
        elif self.type == Types.PARALLEL:
            irPar = ParallelRepresentation(self.name, self.classes, self.dimensions, self.vars, self.weight_var,
                                           self.encoding, self.embeddings, self.debug, self.encoding_fun,
                 self.train_size, self.test_size, self.num_threads, self.vector_size, self.type,
                 self.input_dim, self.high, self.basic, self.padding, self.permutes, self.ngram, self.path, self.not_multiset, self.vectorial, self.performance)
            irPar.run_parallel()

            ''''
f4si *bind_aux(f4si *a, f4si *b, int n){
    int i, j;
    f4si *enc = (f4si *)calloc(DIMENSIONS * n, sizeof(int));
    for(i = 0; i < n; ++i){
        for(j = 0; j < NUM_BATCH; j++){
             enc[(NUM_BATCH * i) + j] = a[(NUM_BATCH * i) + j] * b[NUM_BATCH + j];
        }
    }
    free(a);
    free(b);
    return enc;
}

f4si* ngram(f4si* arr, int n){
    int i, j,k;
    f4si * res = calloc(DIMENSIONS*(INPUT_DIM-n-1), sizeof(int));
    f4si * sample = calloc(DIMENSIONS*(INPUT_DIM-n-1), sizeof(int));

    res = permute(arr,n-1,0,INPUT_DIM-(n-1));
    for (i = 1; i < n; i++){
        sample = permute(arr,n-i-1,i,INPUT_DIM-(n-i));
        res = bind_aux(res,sample,INPUT_DIM-n-1);
    }
    //free(arr);
    return multiset(res);
}

void encode_train_task(void* task){
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,SIGNALS_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS, sizeof(int));

    enc = ngram(CHANNELS,1);
    hard_quantize((float*)enc,1);
    update_weight((float*)enc,label);
    free(enc);
    free(indices);
    free(data);
}


f4si *permute_forward(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float last;
    f4si * res = malloc(DIMENSIONS*(fi-ini)* sizeof(int));

    for (i = ini; i < fi; ++i){
      //last = arr[i*NUM_BATCH][0];
      //last = arr[(i*NUM_BATCH)+NUM_BATCH-1][BATCH-1];
      //printf(" last %f",last);
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
                res[(i-ini)*NUM_BATCH+j-1][num] = arr[i*NUM_BATCH+j][k];
            }
         }

      }
    }

    return res;
}


           f4si *permute_backward(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float last;
    f4si * res = malloc(DIMENSIONS*(fi-ini)* sizeof(int));

    for (i = ini; i < fi; ++i){
      //last = arr[i*NUM_BATCH][0];
      //last = arr[(i*NUM_BATCH)+NUM_BATCH-1][BATCH-1];
      //printf(" last %f",last);
      for (j = 0; j < NUM_BATCH; j++){
         for(k = 0; k < BATCH; k++){
            if ((BATCH*j)+k+dd < ((BATCH*NUM_BATCH))){
                if (k+dd >= BATCH){
                    int num = (k+dd) % BATCH;
                    res[(i-ini)*NUM_BATCH+j][k] = arr[i*NUM_BATCH+j+1][num];

                } else {
                    res[(i-ini)*NUM_BATCH+j][k] = arr[i*NUM_BATCH+j][k+dd];
                }
            } else {
                int num = (k+dd) % BATCH;
                res[(i-ini)*NUM_BATCH+j][k] = arr[i*NUM_BATCH+j-1][num];
            }
         }

      }
    }

    return res;
}

            '''











            '''
                  
        f4si* ngram(f4si* arr, int n){
            int i, j,k,a;
            f4si * res = calloc(DIMENSIONS*(SYMBOLS_DIM-(n-1)), sizeof(int));
            f4si * sample = calloc(DIMENSIONS*(SYMBOLS_DIM-(n-1)), sizeof(int));

            res = permute(arr,n-1,0,SYMBOLS_DIM-(n-1));
            for (i = 1; i < n; i++){
                sample = permute(arr,n-i-1,i,SYMBOLS_DIM-(n-1)+i);
                res = bind_aux(res,sample,SYMBOLS_DIM-(n-1));
            }
            return multiset_aux(res,SYMBOLS_DIM-(n-1));
        }


   f4si *forward(f4si *a, float* indices, f4si* enc){
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
            enc[(NUM_BATCH * i) + j] = a[(int)indices[i]* NUM_BATCH + j];
        }
    }
    return enc;
}

void encode_train_task(void* task){
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,SYMBOLS_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS*INPUT_DIM, sizeof(int));
    enc = forward(SYMBOLS,indices,enc);
    enc = ngram(enc,3);
    hard_quantize((float*)enc,1);
    update_weight((float*)enc,label);
    //free(enc);
    //free(indices);
    //free(data);
}

            
            '''



            '''
 f4si *permute1(f4si* arr, int dd, int ini, int fi)
 {

    int k, j, i;
    float n1, p1;
    float n2, p2;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           res[i*NUM_BATCH][0] = res[i*NUM_BATCH+j][0];
           res[i*NUM_BATCH+j][0] = p1;
       } else if (j == 0){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           p1 = res[i*NUM_BATCH+j][0];
       } else {
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           n1= res[i*NUM_BATCH+j][0];
           res[i*NUM_BATCH+j][0] = p1;
           p1 = n1;
       }

      }

    }

    return res;
}

 f4si *permute2(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float n1, p1;
    float n2, p2;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           res[i*NUM_BATCH][0] = res[i*NUM_BATCH+j][0];
           res[i*NUM_BATCH][1] = res[i*NUM_BATCH+j][1];
           res[i*NUM_BATCH+j][0] = p1;
           res[i*NUM_BATCH+j][1] = p2;
       } else if (j == 0){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           p1 = res[i*NUM_BATCH+j][0];
           p2 = res[i*NUM_BATCH+j][1];
       } else {
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           n1= res[i*NUM_BATCH+j][0];
           n2 = res[i*NUM_BATCH+j][1];
           res[i*NUM_BATCH+j][0] = p1;
           res[i*NUM_BATCH+j][1] = p2;
           p1 = n1;
           p2 = n2;

       }

      }

    }

    return res;
}


 f4si *permute3(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float n1, p1;
    float n2, p2;
    float n3, p3;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           res[i*NUM_BATCH][0] = res[i*NUM_BATCH+j][0];
           res[i*NUM_BATCH][1] = res[i*NUM_BATCH+j][1];
           res[i*NUM_BATCH][2] = res[i*NUM_BATCH+j][2];
           res[i*NUM_BATCH+j][0] = p1;
           res[i*NUM_BATCH+j][1] = p2;
           res[i*NUM_BATCH+j][2] = p3;
       } else if (j == 0){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           p1 = res[i*NUM_BATCH+j][0];
           p2 = res[i*NUM_BATCH+j][1];
           p3 = res[i*NUM_BATCH+j][2];
       } else {
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           n1= res[i*NUM_BATCH+j][0];
           n2 = res[i*NUM_BATCH+j][1];
           n3 = res[i*NUM_BATCH+j][2];
           res[i*NUM_BATCH+j][0] = p1;
           res[i*NUM_BATCH+j][1] = p2;
           res[i*NUM_BATCH+j][2] = p3;
           p1 = n1;
           p2 = n2;
           p3 = n3;
       }

      }

    }

    return res;
}

 f4si *bind_aux(f4si *a, f4si *b, int n){
    int i, j;
    f4si *enc = (f4si *)calloc(DIMENSIONS * n, sizeof(int));
    for(i = 0; i < n; ++i){
        for(j = 0; j < NUM_BATCH; j++){
             enc[(NUM_BATCH * i) + j] = a[(NUM_BATCH * i) + j] * b[(NUM_BATCH * i) + j];
        }
    }
    free(b);
    free(a);
    return enc;
}

f4si* multiset_aux(f4si *a, int n, f4si* res){
    int i, j;
    for(i = 0; i < n; i++){
        for(j = 0; j < NUM_BATCH; ++j){
            res[j] += a[(NUM_BATCH * i) + j];
        }
    }
    return res;
}


void print_m(f4si* arr, int size){
   int i, j, k;
   for (i = 0; i < size; i++){
      for (j = 0; j < NUM_BATCH; j++){
         for(k = 0; k < BATCH; k++){
            printf("%f ",arr[i*NUM_BATCH+j][k]);
         }
      }
      printf("\n");
   }
}
f4si *forward(f4si *a, float* indices, f4si* enc){
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
            enc[(NUM_BATCH * i) + j] = a[(int)indices[i]* NUM_BATCH + j];
        }
    }
    return enc;
}


f4si* ngram(f4si* arr, int n){
    int i, j,k,a;
    f4si * res = calloc(DIMENSIONS*(INPUT_DIM-(n-1)), sizeof(int));
    f4si * sample = calloc(DIMENSIONS*(INPUT_DIM-(n-1)), sizeof(int));

    res = permute(arr,n-1,0,INPUT_DIM-(n-1));
    for (i = 1; i < n; i++){
        sample = permute(arr,n-i-1,i,INPUT_DIM-(n-1)+i);
        res = bind_aux(res,sample,INPUT_DIM-(n-1));
    }
    free(arr);
    return multiset_aux(res,INPUT_DIM-(n-1), res);
}
void encode_train_task(void* task){
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,SYMBOLS_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS*INPUT_DIM, sizeof(int));
    enc = forward(SYMBOLS,indices,enc);
    enc = ngram(enc,3);
    hard_quantize((float*)enc,1);
    update_weight((float*)enc,label);
    free(enc);
    free(indices);
    free(data);
}


void encode_test_task(void* task){
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,SYMBOLS_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS*INPUT_DIM, sizeof(int));
    enc = forward(SYMBOLS,indices,enc);

    enc = ngram(enc,3);
    float *l = linear((float*)enc);
    if(argmax(l) == label){
        free(l);
        update_correct_predictions();
    }
    free(indices);
    free(data);
    free(enc);
}












                
f4si *permute1(f4si* arr, int dd, int ini, int fi)
 {

    int k, j, i;
    float n1, p1;
    float n2, p2;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           res[(i-ini)*NUM_BATCH][0] = res[(i-ini)*NUM_BATCH+j][0];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
       } else if (j == 0){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           p1 = res[(i-ini)*NUM_BATCH+j][0];
       } else {
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           n1= res[(i-ini)*NUM_BATCH+j][0];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           p1 = n1;
       }

      }

    }

    return res;
}

 f4si *permute2(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float n1, p1;
    float n2, p2;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           res[(i-ini)*NUM_BATCH][0] = res[(i-ini)*NUM_BATCH+j][0];
           res[(i-ini)*NUM_BATCH][1] = res[(i-ini)*NUM_BATCH+j][1];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           res[(i-ini)*NUM_BATCH+j][1] = p2;
       } else if (j == 0){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           p1 = res[(i-ini)*NUM_BATCH+j][0];
           p2 = res[(i-ini)*NUM_BATCH+j][1];
       } else {
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           n1= res[(i-ini)*NUM_BATCH+j][0];
           n2 = res[(i-ini)*NUM_BATCH+j][1];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           res[(i-ini)*NUM_BATCH+j][1] = p2;
           p1 = n1;
           p2 = n2;

       }

      }

    }

    return res;
}


 f4si *permute3(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float n1, p1;
    float n2, p2;
    float n3, p3;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           res[(i-ini)*NUM_BATCH][0] = res[(i-ini)*NUM_BATCH+j][0];
           res[(i-ini)*NUM_BATCH][1] = res[(i-ini)*NUM_BATCH+j][1];
           res[(i-ini)*NUM_BATCH][2] = res[(i-ini)*NUM_BATCH+j][2];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           res[(i-ini)*NUM_BATCH+j][1] = p2;
           res[(i-ini)*NUM_BATCH+j][2] = p3;
       } else if (j == 0){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           p1 = res[(i-ini)*NUM_BATCH+j][0];
           p2 = res[(i-ini)*NUM_BATCH+j][1];
           p3 = res[(i-ini)*NUM_BATCH+j][2];
       } else {
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           n1= res[(i-ini)*NUM_BATCH+j][0];
           n2 = res[(i-ini)*NUM_BATCH+j][1];
           n3 = res[(i-ini)*NUM_BATCH+j][2];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           res[(i-ini)*NUM_BATCH+j][1] = p2;
           res[(i-ini)*NUM_BATCH+j][2] = p3;
           p1 = n1;
           p2 = n2;
           p3 = n3;
       }

      }

    }

    return res;
}

 f4si *permute0(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
            res[(i-ini)*NUM_BATCH+j] = arr[i*NUM_BATCH+j];
      }

    }

    return res;
}

 f4si *bind_aux(f4si *a, f4si *b, int n){
    int i, j;
    f4si *enc = (f4si *)calloc(DIMENSIONS * n, sizeof(int));
    for(i = 0; i < n; ++i){
        for(j = 0; j < NUM_BATCH; j++){
             enc[(NUM_BATCH * i) + j] = a[(NUM_BATCH * i) + j] * b[(NUM_BATCH * i) + j];
        }
    }
    free(b);
    free(a);
    return enc;
}

f4si* multiset_aux(f4si *a, int n, f4si* res){
    int i, j;
    for(i = 0; i < n; i++){
        for(j = 0; j < NUM_BATCH; ++j){
            res[j] += a[(NUM_BATCH * i) + j];
        }
    }
    return res;
}


void print_m(f4si* arr, int size){
   int i, j, k;
   for (i = 0; i < size; i++){
      for (j = 0; j < NUM_BATCH; j++){
         for(k = 0; k < BATCH; k++){
            printf("%f ",arr[i*NUM_BATCH+j][k]);
         }
      }
      printf("\n");
   }
}
f4si *forward(f4si *a, float* indices, f4si* enc){
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
            if (indices[i] != 0.0){
                enc[(NUM_BATCH * i) + j] = a[(int)indices[i]* NUM_BATCH + j];
            }
        }
    }
    return enc;
}


f4si* ngram(f4si* arr, int n){
    int i, j,k,a;
    f4si * res = calloc(DIMENSIONS*(INPUT_DIM-(n-1)), sizeof(int));
    f4si * sample = calloc(DIMENSIONS*(INPUT_DIM-(n-1)), sizeof(int));

    res = permute2(arr,n-1,0,INPUT_DIM-(n-1));
    for (i = 1; i < n; i++){
        if (n-i-1 == 1){
            sample = permute1(arr,n-i-1,i,INPUT_DIM-(n-1)+i);
        } else if (n-i-1 == 2){
            sample = permute2(arr,n-i-1,i,INPUT_DIM-(n-1)+i);
        } else if (n-i-1 == 3){
            sample = permute3(arr,n-i-1,i,INPUT_DIM-(n-1)+i);
        } else {
            sample = permute0(arr,n-i-1,i,INPUT_DIM-(n-1)+i);
        }

        res = bind_
 f4si *permute1(f4si* arr, int dd, int ini, int fi)
 {

    int k, j, i;
    float n1, p1;
    float n2, p2;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           res[i*NUM_BATCH][0] = res[i*NUM_BATCH+j][0];
           res[i*NUM_BATCH+j][0] = p1;
       } else if (j == 0){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           p1 = res[i*NUM_BATCH+j][0];
       } else {
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           n1= res[i*NUM_BATCH+j][0];
           res[i*NUM_BATCH+j][0] = p1;
           p1 = n1;
       }

      }

    }

    return res;
}

 f4si *permute2(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float n1, p1;
    float n2, p2;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           res[i*NUM_BATCH][0] = res[i*NUM_BATCH+j][0];
           res[i*NUM_BATCH][1] = res[i*NUM_BATCH+j][1];
           res[i*NUM_BATCH+j][0] = p1;
           res[i*NUM_BATCH+j][1] = p2;
       } else if (j == 0){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           p1 = res[i*NUM_BATCH+j][0];
           p2 = res[i*NUM_BATCH+j][1];
       } else {
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           n1= res[i*NUM_BATCH+j][0];
           n2 = res[i*NUM_BATCH+j][1];
           res[i*NUM_BATCH+j][0] = p1;
           res[i*NUM_BATCH+j][1] = p2;
           p1 = n1;
           p2 = n2;

       }

      }

    }

    return res;
}


 f4si *permute3(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float n1, p1;
    float n2, p2;
    float n3, p3;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           res[i*NUM_BATCH][0] = res[i*NUM_BATCH+j][0];
           res[i*NUM_BATCH][1] = res[i*NUM_BATCH+j][1];
           res[i*NUM_BATCH][2] = res[i*NUM_BATCH+j][2];
           res[i*NUM_BATCH+j][0] = p1;
           res[i*NUM_BATCH+j][1] = p2;
           res[i*NUM_BATCH+j][2] = p3;
       } else if (j == 0){
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           p1 = res[i*NUM_BATCH+j][0];
           p2 = res[i*NUM_BATCH+j][1];
           p3 = res[i*NUM_BATCH+j][2];
       } else {
           res[i*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           n1= res[i*NUM_BATCH+j][0];
           n2 = res[i*NUM_BATCH+j][1];
           n3 = res[i*NUM_BATCH+j][2];
           res[i*NUM_BATCH+j][0] = p1;
           res[i*NUM_BATCH+j][1] = p2;
           res[i*NUM_BATCH+j][2] = p3;
           p1 = n1;
           p2 = n2;
           p3 = n3;
       }

      }

    }

    return res;
}

 f4si *bind_aux(f4si *a, f4si *b, int n){
    int i, j;
    f4si *enc = (f4si *)calloc(DIMENSIONS * n, sizeof(int));
    for(i = 0; i < n; ++i){
        for(j = 0; j < NUM_BATCH; j++){
             enc[(NUM_BATCH * i) + j] = a[(NUM_BATCH * i) + j] * b[(NUM_BATCH * i) + j];
        }
    }
    free(b);
    free(a);
    return enc;
}

f4si* multiset_aux(f4si *a, int n, f4si* res){
    int i, j;
    for(i = 0; i < n; i++){
        for(j = 0; j < NUM_BATCH; ++j){
            res[j] += a[(NUM_BATCH * i) + j];
        }
    }
    return res;
}


void print_m(f4si* arr, int size){
   int i, j, k;
   for (i = 0; i < size; i++){
      for (j = 0; j < NUM_BATCH; j++){
         for(k = 0; k < BATCH; k++){
            printf("%f ",arr[i*NUM_BATCH+j][k]);
         }
      }
      printf("\n");
   }
}
f4si *forward(f4si *a, float* indices, f4si* enc){
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
            enc[(NUM_BATCH * i) + j] = a[(int)indices[i]* NUM_BATCH + j];
        }
    }
    return enc;
}


f4si* ngram(f4si* arr, int n){
    int i, j,k,a;
    f4si * res = calloc(DIMENSIONS*(INPUT_DIM-(n-1)), sizeof(int));
    f4si * sample = calloc(DIMENSIONS*(INPUT_DIM-(n-1)), sizeof(int));

    res = permute(arr,n-1,0,INPUT_DIM-(n-1));
    for (i = 1; i < n; i++){
        sample = permute(arr,n-i-1,i,INPUT_DIM-(n-1)+i);
        res = bind_aux(res,sample,INPUT_DIM-(n-1));
    }
    free(arr);
    return multiset_aux(res,INPUT_DIM-(n-1), res);
}
void encode_train_task(void* task){
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,SYMBOLS_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS*INPUT_DIM, sizeof(int));
    enc = forward(SYMBOLS,indices,enc);
    enc = ngram(enc,3);
    hard_quantize((float*)enc,1);
    update_weight((float*)enc,label);
    free(enc);
    free(indices);
    free(data);
}


void encode_test_task(void* task){
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,SYMBOLS_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS*INPUT_DIM, sizeof(int));
    enc = forward(SYMBOLS,indices,enc);

    enc = ngram(enc,3);
    float *l = linear((float*)enc);
    if(argmax(l) == label){
        free(l);
        update_correct_predictions();
    }
    free(indices);
    free(data);
    free(enc);
}












                
f4si *permute1(f4si* arr, int dd, int ini, int fi)
 {

    int k, j, i;
    float n1, p1;
    float n2, p2;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           res[(i-ini)*NUM_BATCH][0] = res[(i-ini)*NUM_BATCH+j][0];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
       } else if (j == 0){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           p1 = res[(i-ini)*NUM_BATCH+j][0];
       } else {
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           n1= res[(i-ini)*NUM_BATCH+j][0];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           p1 = n1;
       }

      }

    }

    return res;
}

 f4si *permute2(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float n1, p1;
    float n2, p2;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           res[(i-ini)*NUM_BATCH][0] = res[(i-ini)*NUM_BATCH+j][0];
           res[(i-ini)*NUM_BATCH][1] = res[(i-ini)*NUM_BATCH+j][1];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           res[(i-ini)*NUM_BATCH+j][1] = p2;
       } else if (j == 0){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           p1 = res[(i-ini)*NUM_BATCH+j][0];
           p2 = res[(i-ini)*NUM_BATCH+j][1];
       } else {
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           n1= res[(i-ini)*NUM_BATCH+j][0];
           n2 = res[(i-ini)*NUM_BATCH+j][1];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           res[(i-ini)*NUM_BATCH+j][1] = p2;
           p1 = n1;
           p2 = n2;

       }

      }

    }

    return res;
}


 f4si *permute3(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float n1, p1;
    float n2, p2;
    float n3, p3;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           res[(i-ini)*NUM_BATCH][0] = res[(i-ini)*NUM_BATCH+j][0];
           res[(i-ini)*NUM_BATCH][1] = res[(i-ini)*NUM_BATCH+j][1];
           res[(i-ini)*NUM_BATCH][2] = res[(i-ini)*NUM_BATCH+j][2];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           res[(i-ini)*NUM_BATCH+j][1] = p2;
           res[(i-ini)*NUM_BATCH+j][2] = p3;
       } else if (j == 0){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           p1 = res[(i-ini)*NUM_BATCH+j][0];
           p2 = res[(i-ini)*NUM_BATCH+j][1];
           p3 = res[(i-ini)*NUM_BATCH+j][2];
       } else {
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           n1= res[(i-ini)*NUM_BATCH+j][0];
           n2 = res[(i-ini)*NUM_BATCH+j][1];
           n3 = res[(i-ini)*NUM_BATCH+j][2];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           res[(i-ini)*NUM_BATCH+j][1] = p2;
           res[(i-ini)*NUM_BATCH+j][2] = p3;
           p1 = n1;
           p2 = n2;
           p3 = n3;
       }

      }

    }

    return res;
}

 f4si *permute0(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
            res[(i-ini)*NUM_BATCH+j] = arr[i*NUM_BATCH+j];
      }

    }

    return res;
}

 f4si *bind_aux(f4si *a, f4si *b, int n){
    int i, j;
    f4si *enc = (f4si *)calloc(DIMENSIONS * n, sizeof(int));
    for(i = 0; i < n; ++i){
        for(j = 0; j < NUM_BATCH; j++){
             enc[(NUM_BATCH * i) + j] = a[(NUM_BATCH * i) + j] * b[(NUM_BATCH * i) + j];
        }
    }
    free(b);
    free(a);
    return enc;
}

f4si* multiset_aux(f4si *a, int n, f4si* res){
    int i, j;
    for(i = 0; i < n; i++){
        for(j = 0; j < NUM_BATCH; ++j){
            res[j] += a[(NUM_BATCH * i) + j];
        }
    }
    return res;
}


void print_m(f4si* arr, int size){
   int i, j, k;
   for (i = 0; i < size; i++){
      for (j = 0; j < NUM_BATCH; j++){
         for(k = 0; k < BATCH; k++){
            printf("%f ",arr[i*NUM_BATCH+j][k]);
         }
      }
      printf("\n");
   }
}
f4si *forward(f4si *a, float* indices, f4si* enc){
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){
            if (indices[i] != 0.0){
                enc[(NUM_BATCH * i) + j] = a[(int)indices[i]* NUM_BATCH + j];
            }
        }
    }
    return enc;
}


f4si* ngram(f4si* arr, int n){
    int i, j,k,a;
    f4si * res = calloc(DIMENSIONS*(INPUT_DIM-(n-1)), sizeof(int));
    f4si * sample = calloc(DIMENSIONS*(INPUT_DIM-(n-1)), sizeof(int));

    res = permute2(arr,n-1,0,INPUT_DIM-(n-1));
    for (i = 1; i < n; i++){
        if (n-i-1 == 1){
            sample = permute1(arr,n-i-1,i,INPUT_DIM-(n-1)+i);
        } else if (n-i-1 == 2){
            sample = permute2(arr,n-i-1,i,INPUT_DIM-(n-1)+i);
        } else if (n-i-1 == 3){
            sample = permute3(arr,n-i-1,i,INPUT_DIM-(n-1)+i);
        } else {
            sample = permute0(arr,n-i-1,i,INPUT_DIM-(n-1)+i);
        }

        res = bind_aux(res,sample,INPUT_DIM-(n-1));
    }
    free(arr);
    return multiset_aux(res,INPUT_DIM-(n-1), res);
}

void encode_train_task(void* task){
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,SYMBOLS_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS*INPUT_DIM, sizeof(int));
    enc = forward(SYMBOLS,indices,enc);
    enc = ngram(enc,3);
    hard_quantize((float*)enc,1);
    update_weight((float*)enc,label);
    free(enc);
    free(indices);
    free(data);
}


void encode_test_task(void* task){
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,SYMBOLS_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS*INPUT_DIM, sizeof(int));
    enc = forward(SYMBOLS,indices,enc);

    enc = ngram(enc,3);
    float *l = linear((float*)enc);
    if(argmax(l) == label){
        free(l);
        update_correct_predictions();
    }
    free(indices);
    free(data);
    free(enc);
}
void train_loop(){
    int i;
    for(i = 0; i < TRAIN; i++){
        struct Task *task = (struct Task *)calloc(1,sizeof(struct Task));
        task -> data = load_data_next_line(train_data);
        task -> label = load_labels_next_line(train_labels);
        mt_add_job(pool, &encode_train_task, task);
    }
    mt_join(pool);
    normalize();
}

float test_loop(){
    int i;
    for(i = 0; i < TEST; i++){
        struct Task *task = (struct Task *)calloc(1,sizeof(struct Task));
        task -> data = load_data_next_line(test_data);
        task -> label = load_labels_next_line(test_labels);
        mt_add_job(pool, &encode_test_task, task);
    }
    mt_join(pool);
    return CORRECT_PREDICTIONS/(float)TEST;
}

int main(int argc, char **argv) {

    SYMBOLS = random_hv(SYMBOLS_DIM);
	pool = mt_create_pool(NUM_THREADS);
    weights();
    prepare_to_load_data(argv);

    struct timespec begin, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    train_loop();
    float acc = test_loop();

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("lang,%d,%f,%f", DIMENSIONS,elapsed, acc);

}





f4si *permute0(f4si* arr, int dd, int ini, int fi)
{

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    __builtin_memcpy_inline(res, arr + ini*NUM_BATCH, DIMENSIONS*(fi-ini)*sizeof(int));
    return res;
}-maltivec




f4si* ngram2(f4si* arr, const int n){
    int i, j, k;
    f4si* res = calloc(DIMENSIONS, sizeof(int));
    f4si aux;
    f4si actual;
    float n1, p1;
    float n2, p2;
    for (i = 0; i < (INPUT_DIM-(n-1)); ++i){
        for (k = 0; k < n; ++k){
            for (j = 0; j < NUM_BATCH; j++){
                if (k == 0){
                    if (j == NUM_BATCH-1){
                       aux = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       aux[0] = p1;
                       aux[1] = p2;
                   } else if (j == 0){
                       aux = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       p1 = aux[0];
                       p2 = aux[1];
                   } else {
                       aux = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       n1 = aux[0];
                       n2 = aux[1];
                       aux[0] = p1;
                       aux[1] = p2;
                       p1 = n1;
                       p2 = n2;
                    }
                    actual = aux; // 2 pos
                } else if (k == n-1){
                    actual = actual * arr[(i+k)*NUM_BATCH+j];
                } else {
                       if (j == NUM_BATCH-1){
                           aux = __builtin_shufflevector(arr[(i+k)*NUM_BATCH+j],arr[(i+k)*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           aux[0] = p1;
                       } else if (j == 0){
                           aux = __builtin_shufflevector(arr[(i+k)*NUM_BATCH+j],arr[(i+k)*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           p1 = aux[0];
                       } else {
                           aux = __builtin_shufflevector(arr[(i+k)*NUM_BATCH+j],arr[(i+k)*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           n1 = aux[0];
                           aux[0] = p1;
                           p1 = n1;
                       }
                    //res[i*NUM_BATCH+j] = res[(i)*NUM_BATCH+j] * shuffle(arr[(i+k)*NUM_BATCH+j], (n-1-k))

                    actual = aux * actual;
                }
                                      res[j] = res[j] + actual;

              }

        }
    }

    free(arr);
    //arr = &res;

    return res;
}
            aux(res,sample,INPUT_DIM-(n-1));
    }
    free(arr);
    return multiset_aux(res,INPUT_DIM-(n-1), res);
}

void encode_train_task(void* task){
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,SYMBOLS_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS*INPUT_DIM, sizeof(int));
    enc = forward(SYMBOLS,indices,enc);
    enc = ngram(enc,3);
    hard_quantize((float*)enc,1);
    update_weight((float*)enc,label);
    free(enc);
    free(indices);
    free(data);
}


void encode_test_task(void* task){
    float* data = ((struct Task*)task) -> data;
    int label = ((struct Task*)task) -> label;
    float* indices = (float *)calloc(INPUT_DIM, sizeof(float));
    map_range_clamp_one(data,SYMBOLS_DIM-1, indices);
    f4si * enc = calloc(DIMENSIONS*INPUT_DIM, sizeof(int));
    enc = forward(SYMBOLS,indices,enc);

    enc = ngram(enc,3);
    float *l = linear((float*)enc);
    if(argmax(l) == label){
        free(l);
        update_correct_predictions();
    }
    free(indices);
    free(data);
    free(enc);
}
void train_loop(){
    int i;
    for(i = 0; i < TRAIN; i++){
        struct Task *task = (struct Task *)calloc(1,sizeof(struct Task));
        task -> data = load_data_next_line(train_data);
        task -> label = load_labels_next_line(train_labels);
        mt_add_job(pool, &encode_train_task, task);
    }
    mt_join(pool);
    normalize();
}

float test_loop(){
    int i;
    for(i = 0; i < TEST; i++){
        struct Task *task = (struct Task *)calloc(1,sizeof(struct Task));
        task -> data = load_data_next_line(test_data);
        task -> label = load_labels_next_line(test_labels);
        mt_add_job(pool, &encode_test_task, task);
    }
    mt_join(pool);
    return CORRECT_PREDICTIONS/(float)TEST;
}

int main(int argc, char **argv) {

    SYMBOLS = random_hv(SYMBOLS_DIM);
	pool = mt_create_pool(NUM_THREADS);
    weights();
    prepare_to_load_data(argv);

    struct timespec begin, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    train_loop();
    float acc = test_loop();

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
    printf("lang,%d,%f,%f", DIMENSIONS,elapsed, acc);

}





f4si *permute0(f4si* arr, int dd, int ini, int fi)
{

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    __builtin_memcpy_inline(res, arr + ini*NUM_BATCH, DIMENSIONS*(fi-ini)*sizeof(int));
    return res;
}-maltivec




f4si* ngram2(f4si* arr, const int n){
    int i, j, k;
    f4si* res = calloc(DIMENSIONS, sizeof(int));
    f4si aux;
    f4si actual;
    float n1, p1;
    float n2, p2;
    for (i = 0; i < (INPUT_DIM-(n-1)); ++i){
        for (k = 0; k < n; ++k){
            for (j = 0; j < NUM_BATCH; j++){
                if (k == 0){
                    if (j == NUM_BATCH-1){
                       aux = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       aux[0] = p1;
                       aux[1] = p2;
                   } else if (j == 0){
                       aux = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       p1 = aux[0];
                       p2 = aux[1];
                   } else {
                       aux = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       n1 = aux[0];
                       n2 = aux[1];
                       aux[0] = p1;
                       aux[1] = p2;
                       p1 = n1;
                       p2 = n2;
                    }
                    actual = aux; // 2 pos
                } else if (k == n-1){
                    actual = actual * arr[(i+k)*NUM_BATCH+j];
                } else {
                       if (j == NUM_BATCH-1){
                           aux = __builtin_shufflevector(arr[(i+k)*NUM_BATCH+j],arr[(i+k)*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           aux[0] = p1;
                       } else if (j == 0){
                           aux = __builtin_shufflevector(arr[(i+k)*NUM_BATCH+j],arr[(i+k)*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           p1 = aux[0];
                       } else {
                           aux = __builtin_shufflevector(arr[(i+k)*NUM_BATCH+j],arr[(i+k)*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           n1 = aux[0];
                           aux[0] = p1;
                           p1 = n1;
                       }
                    //res[i*NUM_BATCH+j] = res[(i)*NUM_BATCH+j] * shuffle(arr[(i+k)*NUM_BATCH+j], (n-1-k))

                    actual = aux * actual;
                }
                                      res[j] = res[j] + actual;

              }

        }
    }

    free(arr);
    //arr = &res;

    return res;
}



f4si* ngram(f4si* arr, float* indices, f4si* enc, const int n){
    int i, j, k;
    f4si * actual = calloc(DIMENSIONS, sizeof(int));
    f4si aux;

    float n1, p1;
    float n2, p2;
    for (i = 0; i < (INPUT_DIM-(n-1)); ++i){
            for (j = 0; j < NUM_BATCH; j++){
                    for (k = 0; k < n; ++k){

                if (k == 0){
                    if (j == NUM_BATCH-1){
                        if (indices[i] != padding){
                            arr[(NUM_BATCH * i) + j] = arr[(int)indices[i]* NUM_BATCH + j];
                        }
                       aux = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       aux[0] = p1;
                       aux[1] = p2;
                   } else if (j == 0){
                       if (indices[i] != padding){
                            arr[(NUM_BATCH * i) + j] = arr[(int)indices[i]* NUM_BATCH + j];
                       }
                       aux = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       p1 = aux[0];
                       p2 = aux[1];
                   } else {
                       if (indices[i] != padding){
                            arr[(NUM_BATCH * i) + j] = arr[(int)indices[i]* NUM_BATCH + j];
                        }
                       aux = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       n1 = aux[0];
                       n2 = aux[1];
                       aux[0] = p1;
                       aux[1] = p2;
                       p1 = n1;
                       p2 = n2;
                   }

                    actual[j] = aux; // 2 pos
                } else if (k == n-1){
                    actual[j] *= arr[(i+k)*NUM_BATCH+j];
                } else {
                       if (j == NUM_BATCH-1){
                           aux = __builtin_shufflevector(arr[(i+k)*NUM_BATCH+j],arr[(i+k)*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           aux[0] = p1;
                       } else if (j == 0){
                           aux = __builtin_shufflevector(arr[(i+k)*NUM_BATCH+j],arr[(i+k)*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           p1 = aux[0];
                       } else {
                           aux = __builtin_shufflevector(arr[(i+k)*NUM_BATCH+j],arr[(i+k)*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           n1 = aux[0];
                           aux[0] = p1;
                           p1 = n1;
                       }
                    //res[i*NUM_BATCH+j] = res[(i)*NUM_BATCH+j] * shuffle(arr[(i+k)*NUM_BATCH+j], (n-1-k))

                    actual[j] *= aux;
                }
              }
              enc[j] += actual[j];

        }
    }
    return enc;
}

f4si *permute1(f4si* arr, int dd, int ini, int fi)
 {

    int k, j, i;
    float n1, p1;
    float n2, p2;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           res[(i-ini)*NUM_BATCH][0] = res[(i-ini)*NUM_BATCH+j][0];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
       } else if (j == 0){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           p1 = res[(i-ini)*NUM_BATCH+j][0];
       } else {
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
           n1= res[(i-ini)*NUM_BATCH+j][0];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           p1 = n1;
       }

      }

    }

    return res;
}

 f4si *permute2(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float n1, p1;
    float n2, p2;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           res[(i-ini)*NUM_BATCH][0] = res[(i-ini)*NUM_BATCH+j][0];
           res[(i-ini)*NUM_BATCH][1] = res[(i-ini)*NUM_BATCH+j][1];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           res[(i-ini)*NUM_BATCH+j][1] = p2;
       } else if (j == 0){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           p1 = res[(i-ini)*NUM_BATCH+j][0];
           p2 = res[(i-ini)*NUM_BATCH+j][1];
       } else {
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
           n1= res[(i-ini)*NUM_BATCH+j][0];
           n2 = res[(i-ini)*NUM_BATCH+j][1];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           res[(i-ini)*NUM_BATCH+j][1] = p2;
           p1 = n1;
           p2 = n2;

       }

      }

    }

    return res;
}


 f4si *permute3(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;
    float n1, p1;
    float n2, p2;
    float n3, p3;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
       if (j == NUM_BATCH-1){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           res[(i-ini)*NUM_BATCH][0] = res[(i-ini)*NUM_BATCH+j][0];
           res[(i-ini)*NUM_BATCH][1] = res[(i-ini)*NUM_BATCH+j][1];
           res[(i-ini)*NUM_BATCH][2] = res[(i-ini)*NUM_BATCH+j][2];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           res[(i-ini)*NUM_BATCH+j][1] = p2;
           res[(i-ini)*NUM_BATCH+j][2] = p3;
       } else if (j == 0){
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           p1 = res[(i-ini)*NUM_BATCH+j][0];
           p2 = res[(i-ini)*NUM_BATCH+j][1];
           p3 = res[(i-ini)*NUM_BATCH+j][2];
       } else {
           res[(i-ini)*NUM_BATCH+j] = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j], 29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28);
           n1= res[(i-ini)*NUM_BATCH+j][0];
           n2 = res[(i-ini)*NUM_BATCH+j][1];
           n3 = res[(i-ini)*NUM_BATCH+j][2];
           res[(i-ini)*NUM_BATCH+j][0] = p1;
           res[(i-ini)*NUM_BATCH+j][1] = p2;
           res[(i-ini)*NUM_BATCH+j][2] = p3;
           p1 = n1;
           p2 = n2;
           p3 = n3;
       }

      }

    }

    return res;
}

 f4si *permute0(f4si* arr, int dd, int ini, int fi)
{

    int k, j, i;

    f4si * res = calloc(DIMENSIONS*(fi-ini), sizeof(int));
    for (i = ini; i < fi; ++i){
      for (j = 0; j < NUM_BATCH; j++){
            res[(i-ini)*NUM_BATCH+j] = arr[i*NUM_BATCH+j];
      }

    }

    return res;
}

 f4si *bind_aux(f4si *a, f4si *b, int n){
    int i, j;
    f4si *enc = (f4si *)calloc(DIMENSIONS * n, sizeof(int));
    for(i = 0; i < n; ++i){
        for(j = 0; j < NUM_BATCH; j++){
             enc[(NUM_BATCH * i) + j] = a[(NUM_BATCH * i) + j] * b[(NUM_BATCH * i) + j];
        }
    }
    free(b);
    free(a);
    return enc;
}

f4si* multiset_aux(f4si *a, int n, f4si* res){
    int i, j;
    for(i = 0; i < n; i++){
        for(j = 0; j < NUM_BATCH; ++j){
            res[j] += a[(NUM_BATCH * i) + j];
        }
    }
    return res;
}

'''

'''
f =
        if self.padding is not None:
            f =           
            if (indices[i] != padding){
                enc[(NUM_BATCH * i) + j] = a[(int)indices[i]* NUM_BATCH + j];
            }
        
        else:
            f = 
            enc[(NUM_BATCH * i) + j] = a[(int)indices[i]* NUM_BATCH + j];
        
'''


'''
f4si *forward(f4si *a, float* indices, f4si* enc){
    int i, j;
    for(i = 0; i < INPUT_DIM; ++i){
        for(j = 0; j < NUM_BATCH; j++){


        }
    }
    return enc;
}

'''


'''


f4si* ngram(f4si* arr, float* indices, f4si* enc, const int n){
    int i, j, k;
    //f4si * res = calloc(DIMENSIONS*(INPUT_DIM-(n-1)), sizeof(int));
    f4si * forward_arr = calloc(DIMENSIONS*(n-1), sizeof(int));
    f4si aux;
    f4si actual;
    float n1, p1;
    float n2, p2;
    for (i = 0; i < (INPUT_DIM-(n-1)); ++i){
            for (j = 0; j < NUM_BATCH; j++){
                for (k = 0; k < n; ++k){
                    if (k == 0){
                        if (j == NUM_BATCH-1){
                            if (indices[i] != padding){
                                for (int m = 0; m < n; m++){
                                    forward_arr[m+NUM_BATCH*j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                                }
                            }
                           aux = __builtin_shufflevector(forward_arr[k+NUM_BATCH*j],forward_arr[k+NUM_BATCH*j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                           aux[0] = p1;
                           aux[1] = p2;
                       } else if (j == 0){
                            if (indices[i] != padding){
                                for (int m = 0; m < n; m++){
                                    forward_arr[m+NUM_BATCH*j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                                }
                            }
                           aux = __builtin_shufflevector(forward_arr[k+NUM_BATCH*j],forward_arr[k+NUM_BATCH*j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                           p1 = aux[0];
                           p2 = aux[1];
                       } else {
                            if (indices[i] != padding){
                                for (int m = 0; m < n; m++){
                                    forward_arr[m+NUM_BATCH*j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                                }
                            }
                           aux = __builtin_shufflevector(forward_arr[k+NUM_BATCH*j],forward_arr[k+NUM_BATCH*j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                           n1 = aux[0];
                           n2 = aux[1];
                           aux[0] = p1;
                           aux[1] = p2;
                           p1 = n1;
                           p2 = n2;
                       }

                        actual = aux; // 2 pos
                    } else if (k == n-1){
                        actual *= forward_arr[k+NUM_BATCH*j];
                    } else {
                           if (j == NUM_BATCH-1){
                               aux = __builtin_shufflevector(forward_arr[k+NUM_BATCH*j],forward_arr[k+NUM_BATCH*j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                               aux[0] = p1;
                           } else if (j == 0){
                               aux = __builtin_shufflevector(forward_arr[k+NUM_BATCH*j],forward_arr[k+NUM_BATCH*j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                               p1 = aux[0];
                           } else {
                               aux = __builtin_shufflevector(forward_arr[k+NUM_BATCH*j],forward_arr[k+NUM_BATCH*j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                               n1 = aux[0];
                               aux[0] = p1;
                               p1 = n1;
                           }
                        //res[i*NUM_BATCH+j] = res[(i)*NUM_BATCH+j] * shuffle(arr[(i+k)*NUM_BATCH+j], (n-1-k))

                        actual *= aux;
                    }
              }
              enc[j] += actual;

        }
    }
    //free(arr);
    return enc;
}


f4si* ngram(f4si* arr, float* indices, f4si* enc, const int n){
    int i, j, k;
    //f4si * res = calloc(DIMENSIONS*(INPUT_DIM-(n-1)), sizeof(int));
    //f4si aux;
    float n1, p1;
    float n2, p2;
    for (i = 0; i < (INPUT_DIM-(n-1)); ++i){
            for (j = 0; j < NUM_BATCH; j++){
                f4si* actual = calloc(BATCH, sizeof(int));
                f4si* aux = calloc(BATCH, sizeof(int));
                f4si * forward_arr = calloc(DIMENSIONS*(n-1), sizeof(int));

                for (k = 0; k < n; ++k){
                    if (k == 0){
                        if (j == NUM_BATCH-1){
                            if (indices[i] != padding){
                                for (int m = 0; m < n; m++){
                                    forward_arr[m+NUM_BATCH*j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                                }
                            }
                           aux[0] = __builtin_shufflevector(forward_arr[k+NUM_BATCH*j],forward_arr[k+NUM_BATCH*j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                           aux[0][0] = p1;
                           aux[0][1] = p2;
                       } else if (j == 0){
                            if (indices[i] != padding){
                                for (int m = 0; m < n; m++){
                                    forward_arr[m+NUM_BATCH*j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                                }
                            }
                           aux[0] = __builtin_shufflevector(forward_arr[k+NUM_BATCH*j],forward_arr[k+NUM_BATCH*j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                           p1 = aux[0][0];
                           p2 = aux[0][1];
                       } else {
                            if (indices[i] != padding){
                                for (int m = 0; m < n; m++){
                                    forward_arr[m+NUM_BATCH*j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                                }
                            }
                           aux[0] = __builtin_shufflevector(forward_arr[k+NUM_BATCH*j],forward_arr[k+NUM_BATCH*j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                           n1 = aux[0][0];
                           n2 = aux[0][1];
                           aux[0][0] = p1;
                           aux[0][1] = p2;
                           p1 = n1;
                           p2 = n2;
                       }
                        actual[0] = aux[0]; // 2 pos
                    } else if (k == n-1){
                        //actual[0] = actual[0] * forward_arr[k+NUM_BATCH*j];
                    } else {
                           if (j == NUM_BATCH-1){
                               aux[0] = __builtin_shufflevector(forward_arr[k+NUM_BATCH*j],forward_arr[k+NUM_BATCH*j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                               aux[0][0] = p1;
                           } else if (j == 0){
                               aux[0] = __builtin_shufflevector(forward_arr[k+NUM_BATCH*j],forward_arr[k+NUM_BATCH*j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                               p1 = aux[0][0];
                           } else {
                               aux[0] = __builtin_shufflevector(forward_arr[k+NUM_BATCH*j],forward_arr[k+NUM_BATCH*j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                               n1 = aux[0][0];
                               aux[0][0] = p1;
                               p1 = n1;
                           }
                        //res[i*NUM_BATCH+j] = res[(i)*NUM_BATCH+j] * shuffle(arr[(i+k)*NUM_BATCH+j], (n-1-k))

                        //actual[0] *= aux;
                    }
              }
              enc[j] = actual[0];
              /*
              if (i == INPUT_DIM-(n-1)-1){
                            for (int h = 0; h < BATCH; h++){
                printf("%f ", actual[h]);

              }
              printf("\n");
              }
              */

             //
        }
    }
    //free(arr);
    return enc;
}



f4si* ngram(f4si* arr, const int n){
    int i, j, k;
    f4si* res = calloc(DIMENSIONS, sizeof(int));
    f4si aux;
    f4si actual;
    float n1, p1;
    float n2, p2;
    for (i = 0; i < (INPUT_DIM-(n-1)); ++i){
        for (j = 0; j < NUM_BATCH; j++){
            for (k = 0; k < n; ++k){
                if (k == 0){
                    if (j == NUM_BATCH-1){
                       aux = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       aux[0] = p1;
                       aux[1] = p2;
                   } else if (j == 0){
                       aux = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       p1 = aux[0];
                       p2 = aux[1];
                   } else {
                       aux = __builtin_shufflevector(arr[i*NUM_BATCH+j],arr[i*NUM_BATCH+j],30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29);
                       n1 = aux[0];
                       n2 = aux[1];
                       aux[0] = p1;
                       aux[1] = p2;
                       p1 = n1;
                       p2 = n2;
                    }
                    actual = aux; // 2 pos
                } else if (k == n-1){
                    actual = actual * arr[(i+k)*NUM_BATCH+j];
                } else {
                       if (j == NUM_BATCH-1){
                           aux = __builtin_shufflevector(arr[(i+k)*NUM_BATCH+j],arr[(i+k)*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           aux[0] = p1;
                       } else if (j == 0){
                           aux = __builtin_shufflevector(arr[(i+k)*NUM_BATCH+j],arr[(i+k)*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           p1 = aux[0];
                       } else {
                           aux = __builtin_shufflevector(arr[(i+k)*NUM_BATCH+j],arr[(i+k)*NUM_BATCH+j],31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
                           n1 = aux[0];
                           aux[0] = p1;
                           p1 = n1;
                       }
                    //res[i*NUM_BATCH+j] = res[(i)*NUM_BATCH+j] * shuffle(arr[(i+k)*NUM_BATCH+j], (n-1-k))

                    actual = aux * actual;
                }

              }

               res[j] = res[j] + actual;



        }
    }

    free(arr);
    //arr = &res;

    return res;
}




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


f4si* ngram(f4si* arr, float* indices, f4si* enc, const int d){
    int i, j, k;
    f4si aux;
    f4si *forward_arr = calloc(DIMENSIONS * d, sizeof(int));
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
                      aux = shuffle(forward_arr, k, j, (k-i-1));
                      for (k = 0; k < d; k++){
                           aux[k] = p[k];
                       }
                   } else if (j == 0){
                       for (int m = 0; m < d; m++){
                           if (indices[i] != padding){
                               forward_arr[(m*NUM_BATCH)+j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                           } else {
                               forward_arr[(m*NUM_BATCH)+j] *= 0;
                           }
                       }
                      aux = shuffle(forward_arr, k, j, (k-i-1));
                       for (k = 0; k < d; k++){
                           p[k] = aux[k];
                       }
                   } else {
                        for (int m = 0; m < d; m++){
                            if (indices[i] != padding){
                                forward_arr[(m*NUM_BATCH)+j] = arr[(int)indices[i+m]* NUM_BATCH + j];
                            } else {
                                forward_arr[(m*NUM_BATCH)+j] *= 0;
                            }
                        }
                      aux = shuffle(forward_arr, k, j, (k-i-1));
                       for (k = 0; k < d; k++){
                           n[k] = aux[k];
                           aux[k] = p[k];
                           p[k] = n[k];
                       }
                    }
                    actual = aux; 
                } else if (k == d-1){
                    actual = actual * forward_arr[k*NUM_BATCH+j];
                } else {
                       if (j == NUM_BATCH-1){
                      aux = shuffle(forward_arr, k, j, (k-i-1));
                          for (k = 0; k < d; k++){
                               aux[k] = p[k];
                           }
                       } else if (j == 0){
                      aux = shuffle(forward_arr, k, j, (k-i-1));
                           for (k = 0; k < d; k++){
                               p[k] = aux[k];
                           }
                       } else {
                      aux = shuffle(forward_arr, k, j, (k-i-1));
                           for (k = 0; k < d; k++){
                               n[k] = aux[k];
                               aux[k] = p[k];
                               p[k] = n[k];
                           }
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