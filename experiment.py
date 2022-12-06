import os
import sys
import subprocess

dimensions = [64, 128, 512, 1024, 4096, 10240]
files = ['mnist', 'voicehd']
repetitions = 5

for file in files:
    with open("output.txt", "a") as output:
        output.write('\n' + file + '\n')
    for i in range(repetitions):
        for dim in dimensions:
            with open(file + ".hdcc", 'r') as f:
                lines = f.readlines()
                lines = lines[:-2]

            lines.append('.NAME '+ str(file).upper() + str(dim).upper()  +';')
            lines.append('.DIMENSIONS '+ str(dim) +';')

            with open(file + str(dim) +'.hdcc', 'w') as f:
                print(lines)
                f.writelines(lines)

            os.system('python3 main.py ' + str(file) + str(dim) +'.hdcc')
            with open("output.txt", "a") as output:
                print(subprocess.check_output('make'))
                if file == 'mnist':
                    res = subprocess.check_output(["./"+str(file) + str(dim), str(60000), str(10000), "data/MNIST/mnist_train_data", "data/MNIST/mnist_train_labels", "data/MNIST/mnist_test_data", "data/MNIST/mnist_test_labels"]).decode(sys.stdout.encoding)
                if file == 'voicehd':
                    res = subprocess.check_output(["./"+str(file) + str(dim), str(6238), str(1559), "data/ISOLET/isolet_train_data", "data/ISOLET/isolet_train_labels", "data/ISOLET/isolet_test_data", "data/ISOLET/isolet_test_labels"]).decode(sys.stdout.encoding)
                output.writelines(res)