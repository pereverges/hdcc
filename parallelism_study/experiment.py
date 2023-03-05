import os
import sys
import subprocess
from datetime import datetime

import os
from collections import deque
from subprocess import Popen, PIPE

now = '_hdcc_' + datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
folder = 'results/'

out_file = sys.argv[1]

dimensions = [10240]
#files = ['languages']
files = ['languages','mnist']
#train_size = [210032]
train_size = [210032, 60000]
#test_size = [21000]
test_size = [21000, 10000]
vector_size = 128
num_threads = [1]
type_exec = 'SEQUENTIAL'
repetitions = 1

position = 58
ti = '-v'
if 'mac' == out_file:
    position = -4
    ti = '-l'


out_file += '_vector_size'

with open(folder + out_file + now, "+a") as output:
    output.write('Application,Dimensions,Time,Accuracy,Memory,Threads\n')

data_path = '../data'

for index, file in enumerate(files):
    fi = file

    for i in range(repetitions):
        for threads in num_threads:
            for dim in dimensions:
                with open(file + ".hdcc", 'r') as f:
                    lines = f.readlines()
                    lines = lines[:-8]

                lines.append('.TYPE ' + str(type_exec) + ';\n')
                lines.append('.NAME ' + str(fi).upper() + str(dim).upper() + ';\n')
                lines.append('.DIMENSIONS ' + str(dim) + ';\n')
                lines.append('.TRAIN_SIZE ' + str(train_size[index]) + ';\n')
                lines.append('.TEST_SIZE ' + str(test_size[index]) + ';\n')
                lines.append('.NUM_THREADS ' + str(threads) + ';\n')
                lines.append('.VECTOR_SIZE ' + str(vector_size) + ';\n')
                lines.append('.VECTORIAL TRUE;\n')

                with open(file + str(dim) +'.hdcc', 'w') as f:
                    f.writelines(lines)

                os.system('python3 ../src/main.py ' + str(file) + str(dim) +'.hdcc')
                with open(folder+out_file+now, "a") as output:
                    subprocess.check_output('make')
                    DEVNULL = open(os.devnull, 'wb', 0)

                    if fi == 'mnist':
                        res = subprocess.check_output(["./"+str(file) + str(dim),
                                                       data_path + "/MNIST/mnist_train_data",
                                                       data_path + "/MNIST/mnist_train_labels",
                                                       data_path + "/MNIST/mnist_test_data",
                                                       data_path + "/MNIST/mnist_test_labels"]).decode(sys.stdout.encoding)
                        p = subprocess.Popen(["/usr/bin/time", ti, "./" + str(file) + str(dim),
                                           data_path + "/MNIST/mnist_train_data",
                                           data_path + "/MNIST/mnist_train_labels",
                                           data_path + "/MNIST/mnist_test_data",
                                           data_path + "/MNIST/mnist_test_labels"], stdout=DEVNULL, stderr=PIPE)

                    if fi == 'voicehd':
                        res = subprocess.check_output(["./"+str(file) + str(dim),
                                                       data_path + "/ISOLET/isolet_train_data",
                                                       data_path + "/ISOLET/isolet_train_labels",
                                                       data_path + "/ISOLET/isolet_test_data",
                                                       data_path + "/ISOLET/isolet_test_labels"]).decode(sys.stdout.encoding)
                        p = subprocess.Popen(["/usr/bin/time", ti, "./" + str(file) + str(dim),
                                               data_path + "/ISOLET/isolet_train_data",
                                               data_path + "/ISOLET/isolet_train_labels",
                                               data_path + "/ISOLET/isolet_test_data",
                                               data_path + "/ISOLET/isolet_test_labels"], stdout=DEVNULL, stderr=PIPE)

                    if fi == 'emgp':
                        res = subprocess.check_output(["./"+str(file) + str(dim),
                             data_path + "/EMG/patient_1_train_data",
                             data_path + "/EMG/patient_1_train_labels",
                             data_path + "/EMG/patient_1_test_data",
                             data_path + "/EMG/patient_1_test_labels"]).decode(sys.stdout.encoding)

                        p = subprocess.Popen(["/usr/bin/time",ti,"./"+str(file) + str(dim),
                                 data_path + "/EMG/patient_1_train_data",
                                 data_path + "/EMG/patient_1_train_labels",
                                 data_path + "/EMG/patient_1_test_data",
                                 data_path + "/EMG/patient_1_test_labels"], stdout=DEVNULL, stderr=PIPE)

                    if fi == 'emgpp':
                        res = subprocess.check_output(
                            ["./" + str(file) + str(dim),
                             data_path + "/EMG/patient_2_train_data",
                             data_path + "/EMG/patient_2_train_labels",
                             data_path + "/EMG/patient_2_test_data",
                             data_path + "/EMG/patient_2_test_labels"]).decode(sys.stdout.encoding)

                        p = subprocess.Popen(["/usr/bin/time",ti,"./"+str(file) + str(dim),
                                 data_path + "/EMG/patient_2_train_data",
                                 data_path + "/EMG/patient_2_train_labels",
                                 data_path + "/EMG/patient_2_test_data",
                                 data_path + "/EMG/patient_2_test_labels"], stdout=DEVNULL, stderr=PIPE)
                    if fi == 'emgppp':
                        res = subprocess.check_output(
                            ["./" + str(file) + str(dim),
                             data_path + "/EMG/patient_3_train_data",
                             data_path + "/EMG/patient_3_train_labels",
                             data_path + "/EMG/patient_3_test_data",
                             data_path + "/EMG/patient_3_test_labels"]).decode(sys.stdout.encoding)

                        p = subprocess.Popen(["/usr/bin/time",ti,"./"+str(file) + str(dim),
                                 data_path + "/EMG/patient_3_train_data",
                                 data_path + "/EMG/patient_3_train_labels",
                                 data_path + "/EMG/patient_3_test_data",
                                 data_path + "/EMG/patient_3_test_labels"], stdout=DEVNULL, stderr=PIPE)
                    if fi == 'emgpppp':
                        res = subprocess.check_output(
                            ["./" + str(file) + str(dim),
                             data_path + "/EMG/patient_4_train_data",
                             data_path + "/EMG/patient_4_train_labels",
                             data_path + "/EMG/patient_4_test_data",
                             data_path + "/EMG/patient_4_test_labels"]).decode(sys.stdout.encoding)

                        p = subprocess.Popen(["/usr/bin/time",ti,"./"+str(file) + str(dim),
                                 data_path + "/EMG/patient_4_train_data",
                                 data_path + "/EMG/patient_4_train_labels",
                                 data_path + "/EMG/patient_4_test_data",
                                 data_path + "/EMG/patient_4_test_labels"], stdout=DEVNULL, stderr=PIPE)
                    if fi == 'emgppppp':
                        res = subprocess.check_output(
                            ["./" + str(file) + str(dim),
                             data_path + "/EMG/patient_5_train_data",
                             data_path + "/EMG/patient_5_train_labels",
                             data_path + "/EMG/patient_5_test_data",
                             data_path + "/EMG/patient_5_test_labels"]).decode(sys.stdout.encoding)

                        p = subprocess.Popen(["/usr/bin/time",ti,"./"+str(file) + str(dim),
                                 data_path + "/EMG/patient_5_train_data",
                                 data_path + "/EMG/patient_5_train_labels",
                                 data_path + "/EMG/patient_5_test_data",
                                 data_path + "/EMG/patient_5_test_labels"], stdout=DEVNULL, stderr=PIPE)
                    if fi == 'languages':
                        res = subprocess.check_output(
                            ["./" + str(file) + str(dim),
                             data_path + "/LANGUAGES/train_data.txt",
                             data_path + "/LANGUAGES/train_labels.txt",
                             data_path + "/LANGUAGES/test_data.txt",
                             data_path + "/LANGUAGES/test_labels.txt"]).decode(sys.stdout.encoding)

                        p = subprocess.Popen(["/usr/bin/time",ti,"./"+str(file) + str(dim),
                                 data_path + "/LANGUAGES/train_data.txt",
                                 data_path + "/LANGUAGES/train_labels.txt",
                                 data_path + "/LANGUAGES/test_data.txt",
                                 data_path + "/LANGUAGES/test_labels.txt"], stdout=DEVNULL, stderr=PIPE)
                    with p.stderr:
                        q = deque(iter(p.stderr.readline, b''))
                    rc = p.wait()
                    #print(str(b''.join(q).decode().strip().split()[40:60]))
                    if position == 58:
                        print(b''.join(q).decode().strip().split())
                        res += ","+str(int(b''.join(q).decode().strip().split()[position])*1000)
                    else:
                        res += ","+b''.join(q).decode().strip().split()[position]
                    res += ","+str(threads)+"\n"
                    output.writelines(res)
