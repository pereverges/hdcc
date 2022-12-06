import os
import sys
import subprocess

dimensions = [64, 128, 512, 1024, 4096, 10240]
files = ['mnist', 'voicehd']
repetitions = 5

for file in files:
    with open("output_torchhd.txt", "a") as output:
        output.write('\n' + file + '\n')
    for i in range(repetitions):
        for dim in dimensions:
            with open("output.txt", "a") as output:
                res = subprocess.check_output(['python3', file + '.py', str(dim)]).decode(sys.stdout.encoding)
                output.writelines(res)
