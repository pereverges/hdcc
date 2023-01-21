import os
import sys
import subprocess
from datetime import datetime

import os
from collections import deque
from subprocess import Popen, PIPE

now = '_hdcc_' + datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
folder = 'experiments/'

out_file = sys.argv[1]

dimensions = [64, 128, 512, 1024, 4096, 10240]
files = ['emgp','emgpp','emgppp','emgpppp','emgppppp','mnist', 'voicehd']
train_size = [368, 345, 338, 333, 235, 60000, 6238]
test_size = [ 158, 148, 145, 143, 101, 10000, 1559]
vector_size = 128
num_threads = 4
type_exec = 'PARALLEL_MEMORY_EFFICIENT'
repetitions = 1

position = 58
ti = '-v'
if 'mac' == out_file:
    position = -4
    ti = '-l'


out_file += '_num_threads_' + str(num_threads) + '_vector_size'

with open(folder + out_file + now, "a") as output:
    output.write('Application,Dimensions,Time,Accuracy,Memory\n')



for index, file in enumerate(files):
    for i in range(repetitions):
        for dim in dimensions:
            with open(file + ".hdcc", 'r') as f:
                lines = f.readlines()
                lines = lines[:-7]

            lines.append('.TYPE ' + str(type_exec) + ';\n')
            lines.append('.NAME ' + str(file).upper() + str(dim).upper() + ';\n')
            lines.append('.DIMENSIONS ' + str(dim) + ';\n')
            lines.append('.TRAIN_SIZE ' + str(train_size[index]) + ';\n')
            lines.append('.TEST_SIZE ' + str(test_size[index]) + ';\n')
            lines.append('.NUM_THREADS ' + str(num_threads) + ';\n')
            lines.append('.VECTOR_SIZE ' + str(vector_size) + ';\n')

            with open(file + str(dim) +'.hdcc', 'w') as f:
                #print(lines)
                f.writelines(lines)

            os.system('python3 main.py ' + str(file) + str(dim) +'.hdcc')
            with open(folder+out_file+now, "a") as output:
                subprocess.check_output('make')
                DEVNULL = open(os.devnull, 'wb', 0)

                if file == 'mnist':
                    res = subprocess.check_output(["./"+str(file) + str(dim),
                                                   "data/MNIST/mnist_train_data",
                                                   "data/MNIST/mnist_train_labels",
                                                   "data/MNIST/mnist_test_data",
                                                   "data/MNIST/mnist_test_labels"]).decode(sys.stdout.encoding)
                    p = subprocess.Popen(["/usr/bin/time", ti, "./" + str(file) + str(dim),
                                       "data/MNIST/mnist_train_data",
                                       "data/MNIST/mnist_train_labels",
                                       "data/MNIST/mnist_test_data",
                                       "data/MNIST/mnist_test_labels"], stdout=DEVNULL, stderr=PIPE)

                if file == 'voicehd':
                    res = subprocess.check_output(["./"+str(file) + str(dim),
                                                   "data/ISOLET/isolet_train_data",
                                                   "data/ISOLET/isolet_train_labels",
                                                   "data/ISOLET/isolet_test_data",
                                                   "data/ISOLET/isolet_test_labels"]).decode(sys.stdout.encoding)
                    p = subprocess.Popen(["/usr/bin/time", ti, "./" + str(file) + str(dim),
                                          "data/ISOLET/isolet_train_data",
                                           "data/ISOLET/isolet_train_labels",
                                           "data/ISOLET/isolet_test_data",
                                           "data/ISOLET/isolet_test_labels"], stdout=DEVNULL, stderr=PIPE)

                if file == 'emgp':
                    res = subprocess.check_output(["./"+str(file) + str(dim),
                         "data/EMG_based_hand_gesture/patient_1_train_data",
                         "data/EMG_based_hand_gesture/patient_1_train_labels",
                         "data/EMG_based_hand_gesture/patient_1_test_data",
                         "data/EMG_based_hand_gesture/patient_1_test_labels"]).decode(sys.stdout.encoding)

                    p = subprocess.Popen(["/usr/bin/time",ti,"./"+str(file) + str(dim),
                             "data/EMG_based_hand_gesture/patient_1_train_data",
                             "data/EMG_based_hand_gesture/patient_1_train_labels",
                             "data/EMG_based_hand_gesture/patient_1_test_data",
                             "data/EMG_based_hand_gesture/patient_1_test_labels"], stdout=DEVNULL, stderr=PIPE)

                if file == 'emgpp':
                    res = subprocess.check_output(
                        ["./" + str(file) + str(dim),
                         "data/EMG_based_hand_gesture/patient_2_train_data",
                         "data/EMG_based_hand_gesture/patient_2_train_labels",
                         "data/EMG_based_hand_gesture/patient_2_test_data",
                         "data/EMG_based_hand_gesture/patient_2_test_labels"]).decode(sys.stdout.encoding)

                    p = subprocess.Popen(["/usr/bin/time",ti,"./"+str(file) + str(dim),
                             "data/EMG_based_hand_gesture/patient_2_train_data",
                             "data/EMG_based_hand_gesture/patient_2_train_labels",
                             "data/EMG_based_hand_gesture/patient_2_test_data",
                             "data/EMG_based_hand_gesture/patient_2_test_labels"], stdout=DEVNULL, stderr=PIPE)
                if file == 'emgppp':
                    res = subprocess.check_output(
                        ["./" + str(file) + str(dim), "data/EMG_based_hand_gesture/patient_3_train_data",
                         "data/EMG_based_hand_gesture/patient_3_train_labels",
                         "data/EMG_based_hand_gesture/patient_3_test_data",
                         "data/EMG_based_hand_gesture/patient_3_test_labels"]).decode(sys.stdout.encoding)

                    p = subprocess.Popen(["/usr/bin/time",ti,"./"+str(file) + str(dim),
                             "data/EMG_based_hand_gesture/patient_3_train_data",
                             "data/EMG_based_hand_gesture/patient_3_train_labels",
                             "data/EMG_based_hand_gesture/patient_3_test_data",
                             "data/EMG_based_hand_gesture/patient_3_test_labels"], stdout=DEVNULL, stderr=PIPE)
                if file == 'emgpppp':
                    res = subprocess.check_output(
                        ["./" + str(file) + str(dim), "data/EMG_based_hand_gesture/patient_4_train_data",
                         "data/EMG_based_hand_gesture/patient_4_train_labels",
                         "data/EMG_based_hand_gesture/patient_4_test_data",
                         "data/EMG_based_hand_gesture/patient_4_test_labels"]).decode(sys.stdout.encoding)

                    p = subprocess.Popen(["/usr/bin/time",ti,"./"+str(file) + str(dim),
                             "data/EMG_based_hand_gesture/patient_4_train_data",
                             "data/EMG_based_hand_gesture/patient_4_train_labels",
                             "data/EMG_based_hand_gesture/patient_4_test_data",
                             "data/EMG_based_hand_gesture/patient_4_test_labels"], stdout=DEVNULL, stderr=PIPE)
                if file == 'emgppppp':
                    res = subprocess.check_output(
                        ["./" + str(file) + str(dim), "data/EMG_based_hand_gesture/patient_5_train_data",
                         "data/EMG_based_hand_gesture/patient_5_train_labels",
                         "data/EMG_based_hand_gesture/patient_5_test_data",
                         "data/EMG_based_hand_gesture/patient_5_test_labels"]).decode(sys.stdout.encoding)

                    p = subprocess.Popen(["/usr/bin/time",ti,"./"+str(file) + str(dim),
                             "data/EMG_based_hand_gesture/patient_5_train_data",
                             "data/EMG_based_hand_gesture/patient_5_train_labels",
                             "data/EMG_based_hand_gesture/patient_5_test_data",
                             "data/EMG_based_hand_gesture/patient_5_test_labels"], stdout=DEVNULL, stderr=PIPE)
                if file == 'languages':
                    res = subprocess.check_output(
                        ["./" + str(file) + str(dim), "data/LANGUAGES/train_data.txt",
                         "data/LANGUAGES/train_labels.txt",
                         "data/LANGUAGES/test_data.txt",
                         "data/LANGUAGES/test_labels.txt"]).decode(sys.stdout.encoding)

                    p = subprocess.Popen(["/usr/bin/time",ti,"./"+str(file) + str(dim),
                             "data/LANGUAGES/train_data.txt",
                             "data/LANGUAGES/train_labels.txt",
                             "data/LANGUAGES/test_data.txt",
                             "data/LANGUAGES/test_labels.txt"], stdout=DEVNULL, stderr=PIPE)
                with p.stderr:
                    q = deque(iter(p.stderr.readline, b''))
                rc = p.wait()
                #print(str(b''.join(q).decode().strip().split()[40:60]))
                if position == 58:
                    res += ","+str(int(b''.join(q).decode().strip().split()[position])*1000)+"\n"    
                else:
                    res += ","+b''.join(q).decode().strip().split()[position]+"\n"
                output.writelines(res)
