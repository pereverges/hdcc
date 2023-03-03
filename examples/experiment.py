import os
import sys
import subprocess
from datetime import datetime

import os
from collections import deque
from subprocess import Popen, PIPE

now = '_hdcc_' + datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
folder = '../results/'

out_file = sys.argv[1]

#dimensions = [64, 128, 512, 1024, 4096, 10240]
dimensions = [32]
files = ['languages','emgp','emgpp','emgppp','emgpppp','emgppppp','mnist', 'voicehd']
train_size = [210032, 368, 345, 338, 333, 235, 60000, 6238]
test_size = [21000, 158, 148, 145, 143, 101, 10000, 1559]
vector_size = 128
num_threads = 4
type_exec = ['PARALLEL','SEQUENTIAL']
repetitions = 1
vectorial = ['FALSE','TRUE']

position = 58
ti = '-v'
if 'mac' == out_file:
    position = -4
    ti = '-l'


out_file += '_num_threads_' + str(num_threads) + '_vector_size'

with open(folder + out_file + now, "+a") as output:
    output.write('Application,Dimensions,Time,Accuracy,Memory,Vectorial,Parallel\n')

data_path = '../data'

for index, file in enumerate(files):
    fi = file

    for i in range(repetitions):
        for vec in vectorial:
            for par in type_exec:
                for dim in dimensions:
                    with open(file + ".hdcc", 'r') as f:
                        lines = f.readlines()
                        lines = lines[:-8]

                    lines.append('.TYPE ' + str(par) + ';\n')
                    lines.append('.NAME ' + str(fi).upper() + str(dim).upper() + ';\n')
                    lines.append('.DIMENSIONS ' + str(dim) + ';\n')
                    lines.append('.TRAIN_SIZE ' + str(train_size[index]) + ';\n')
                    lines.append('.TEST_SIZE ' + str(test_size[index]) + ';\n')
                    lines.append('.NUM_THREADS ' + str(num_threads) + ';\n')
                    lines.append('.VECTOR_SIZE ' + str(vector_size) + ';\n')
                    lines.append('.VECTORIAL ' + str(vec) + ';\n')

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

                        if vec == 'TRUE':
                            res += ",true"
                        else:
                            res += ",false"

                        if par == 'PARALLEL':
                            res += ",true\n"
                        else:
                            res += ",false\n"
                        output.writelines(res)
