#!/usr/bin/python

import argparse
import sys
import math
import random
import string
import subprocess
import time

## seeds inputs (hard coded for now)
file_list = ["seeds/t0.jpg", "seeds/t1.jpg", "seeds/t2.jpg", "seeds/t3.jpg", "seeds/t4.jpg", "seeds/t5.jpg"]
## Run DNNs (hard coded for now)
apps = ["./run_template_mnist.py"]
## fuzzing iterations (hard coded for now)
num_tests = 1000

num_crashes = 0
for i in range(num_tests):
    file_choice = random.choice(file_list)
    buf = bytearray(open(file_choice, 'rb').read())
    numwrites = 1 # to keep a minimum change (hard coded for now)
    for j in range(numwrites):
        rbyte = random.randrange(256)
        rn = random.randrange(len(buf))
        buf[rn] = rbyte
        
    fuzz_output = 'mutant-iter{0}'.format(i)
    f = open('mutants/' + fuzz_output, 'wb')
    f.write(buf)
    f.close()

    commandline = [apps[0], '--model', 'mnist2.h5', '--origins', file_choice, '--mutants', 'mutants/'+fuzz_output]
    process = subprocess.Popen(commandline)

    time.sleep(4)
    crashed = process.poll()
    if not crashed:
        process.terminate()
    else:
        num_crashes += 1
        output = open("advs.list", 'a')
        output.write("Adv# {0}: command {1}\n".format(num_crashes, commandline))
        output.close()
        adv_output = 'advs/' + fuzz_output
        f = open(adv_output, 'wb')
        f.write(buf)
        f.close()
