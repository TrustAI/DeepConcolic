import argparse
import sys
import os
import cv2
import math
import random
import string
import subprocess
import time
from utils import *
from variables import *

## to be refined
apps = ['./src/run_template.py']

def deepconcolic_fuzz(test_object, outs, model_name, stime, file_list,
                      num_tests = 1000, num_processes = 1):
  #report_args = { 'save_input_func': test_object.save_input_func,
  #                'inp_ub': test_object.inp_ub,
  #                'outs': outs}
  
  if not os.path.isdir(outs):
      os.system('mkdir -p {0}'.format(outs))
  mutant_path = outs + '/mutants'
  if not os.path.isdir(mutant_path):
      os.system('mkdir -p {0}'.format(mutant_path))
  adv_path = outs + '/advs'
  if not os.path.isdir(adv_path):
      os.system('mkdir -p {0}'.format(adv_path))

  data = test_object.raw_data.data
  if data.shape[1] == 28: # todo: this is mnist hacking
    img_rows, img_cols, img_channels = data.shape[1], data.shape[2], 1
  else:
    img_rows, img_cols, img_channels = data.shape[1], data.shape[2], data.shape[3]

  num_crashes = 0
  for i in range(num_tests):
      processes = []
      fuzz_outputs = []
      for j in range(0, num_processes):
          file_choice = random.choice(file_list)
          buf = bytearray(open(file_choice, 'rb').read())
          numwrites = 1 # to keep a minimum change (hard coded for now)
          for j in range(numwrites):
              rbyte = random.randrange(256)
              rn = random.randrange(len(buf))
              buf[rn] = rbyte
              
          fuzz_output = mutant_path + '/mutant-iter{0}-p{1}'.format(i, j)
          fuzz_outputs.append(fuzz_output)
          f = open(fuzz_output, 'wb')
          f.write(buf)
          f.close()
  
          commandline = ['python', apps[0], '--model', model_name, '--origins', file_choice, '--mutants', fuzz_output, '--input-rows', str(img_rows), '--input-cols', str(img_cols), '--input-channels', str(img_channels)]
          process = subprocess.Popen(commandline)
          processes.append(process)
  
      time.sleep(stime) # (hard coded for now)
      for j in range(0, num_processes):
          process = processes[j]
          fuzz_output = fuzz_outputs[j]
          crashed = process.poll()
          print ('>>>>>', crashed)
          if crashed == SIG_NORMAL:
              process.terminate()
          elif crashed == SIG_COV:
              ## TODO coverage guided; add fuzz_output into the queue
              print (">>>> add fuzz_output into the queue")
              process.terminate()
          elif crashed == SIG_ADV:
              num_crashes += 1
              output = open(outs+"/advs.list", 'a')
              output.write("Adv# {0}: command {1}\n".format(num_crashes, commandline))
              output.close()
              adv_output = adv_path+'/' + fuzz_output.split('/')[-1]
              f = open(adv_output, 'wb')
              f.write(buf)
              f.close()
          else: pass

  #print (report_args)
