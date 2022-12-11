import os
import sys
import subprocess

out_file = sys.argv[1]

dimensions = [64, 128, 512, 1024, 4096, 10240]
files = ['mnist', 'voicehd']
repetitions = 1

for file in files:
    with open(out_file, "a") as output:
        output.write('\n' + file + '\n')
    for i in range(repetitions):
        for dim in dimensions:
            with open(out_file, "a") as output:
                res = subprocess.check_output(['python3', file + '.py', str(dim)]).decode(sys.stdout.encoding)
                output.writelines(res)
