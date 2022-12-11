import sys
import subprocess
from datetime import datetime

now = '_torchhd_' + datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
folder = 'experiments/'

out_file = sys.argv[1]

dimensions = [64, 128, 512, 1024, 4096, 10240]
files = ['mnist', 'voicehd']
repetitions = 1

with open(folder + out_file + now, "a") as output:
    output.write('Application,Dimensions,Time,Accuracy\n')

for file in files:
    for i in range(repetitions):
        for dim in dimensions:
            with open(folder+out_file+now, "a") as output:
                res = subprocess.check_output(['python3', file + '.py', str(dim)]).decode(sys.stdout.encoding)
                output.writelines(res)
