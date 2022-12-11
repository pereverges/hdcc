import os
import sys
import subprocess

out_file = sys.argv[1]

dimensions = [10240]
files = ['mnist', 'voicehd']
train_size = [60000, 6238]
test_size = [10000, 1559]
vector_size = 128
num_threads = 6
type_exec = 'PARALLEL'
repetitions = 1

for index, file in enumerate(files):
    with open(out_file, "a") as output:
        output.write('\n' + file + ': num_threads ' + str(num_threads) + ', vector_size ' + str(vector_size) + '\n')
    for i in range(repetitions):
        for dim in dimensions:
            with open(file + ".hdcc", 'r') as f:
                lines = f.readlines()
                lines = lines[:-7]

            lines.append('.TYPE '+ str(type_exec) +';\n')
            lines.append('.NAME '+ str(file).upper() + str(dim).upper()  +';\n')
            lines.append('.DIMENSIONS '+ str(dim) +';\n')
            lines.append('.TRAIN_SIZE '+ str(train_size[index]) +';\n')
            lines.append('.TEST_SIZE '+ str(test_size[index]) +';\n')
            lines.append('.NUM_THREADS '+ str(num_threads) +';\n')
            lines.append('.VECTOR_SIZE '+ str(vector_size) +';\n')

            with open(file + str(dim) +'.hdcc', 'w') as f:
                print(lines)
                f.writelines(lines)

            os.system('python3 main.py ' + str(file) + str(dim) +'.hdcc')
            with open(out_file, "a") as output:
                print(subprocess.check_output('make'))
                if file == 'mnist':
                    res = subprocess.check_output(["./"+str(file) + str(dim), "data/MNIST/mnist_train_data", "data/MNIST/mnist_train_labels", "data/MNIST/mnist_test_data", "data/MNIST/mnist_test_labels"]).decode(sys.stdout.encoding)
                if file == 'voicehd':
                    res = subprocess.check_output(["./"+str(file) + str(dim), "data/ISOLET/isolet_train_data", "data/ISOLET/isolet_train_labels", "data/ISOLET/isolet_test_data", "data/ISOLET/isolet_test_labels"]).decode(sys.stdout.encoding)
                output.writelines(res)