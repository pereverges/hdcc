import sys
import subprocess
from datetime import datetime
import os
from collections import deque
from subprocess import Popen, PIPE

now = '_torchhd_' + datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
folder = '../results/'

out_file = sys.argv[1]

dimensions = [64, 128, 512, 1024, 4096, 10240]
#files = ['languages','emg', 'voicehd', 'mnist']
files = ['emg', 'voicehd', 'mnist']

patients = ['0','1','2','3','4']
repetitions = 5

with open(folder + out_file + now, "a") as output:
    output.write('Application,Dimensions,Time,Accuracy,Memory\n')

for file in files:
    for i in range(repetitions):
        for dim in dimensions:
            with open(folder+out_file+now, "a") as output:
                if file == 'emg':
                    for patient in patients:
                        res = subprocess.check_output(['python3', file + '.py', patient, str(dim)]).decode(sys.stdout.encoding)
                        try:
                            subprocess.check_output(["python","/home/pverges/.local/lib/python3.9/site-packages/mprof.py","run","--multiprocess","--include-children","python3",file + '.py', patient, str(dim)])
                        except:
                            print('Error')
                        DEVNULL = open(os.devnull, 'wb', 0)
                        memor = subprocess.check_output(["python","/home/pverges/.local/lib/python3.9/site-packages/mprof.py", 'peak']).decode(sys.stdout.encoding).split()[-6]
                        if memor == 'last':
                            memor = subprocess.check_output(["python","/home/pverges/.local/lib/python3.9/site-packages/mprof.py", 'peak']).decode(sys.stdout.encoding).split()[-2]
                        res += ',' + str(float(memor) * 1000000) + '\n'

                        output.writelines(res)
                else:
                    res = subprocess.check_output(['python3', file + '.py', str(dim)]).decode(sys.stdout.encoding)
                    try:
                        subprocess.check_output(["python","/home/pverges/.local/lib/python3.9/site-packages/mprof.py","run","--multiprocess","--include-children","python3",file + '.py', str(dim)])
                    except:
                        print('Error')
                    DEVNULL = open(os.devnull, 'wb', 0)
                    res += ','+str(float(subprocess.check_output(["python","/home/pverges/.local/lib/python3.9/site-packages/mprof.py", 'peak']).decode(sys.stdout.encoding).split()[-2])*1000000)+'\n'

                    output.writelines(res)
