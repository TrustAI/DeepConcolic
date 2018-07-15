import argparse
import sys
from datetime import datetime

import keras
#from keras.datasets import mnist
from keras.models import *
#from keras.layers import * 
#from keras import *

from ap_lp import *
from utils import *

class effective_layert:
  def __init__(self, layer_index, current_layer, is_conv=False):
    self.layer_index=layer_index
    self.activations=[]
    self.is_conv=is_conv
    self.current_layer=current_layer
    self.fk=1.0
    sp=current_layer.output.shape
    if is_conv:
      #self.cover_map=np.ones((1, sp[1], sp[2], sp[3]))
      self.cover_map=np.zeros((1, sp[1], sp[2], sp[3]), dtype=bool)
    else:
      #self.cover_map=np.ones((1, sp[1]))
      self.cover_map=np.zeros((1, sp[1]), dtype=bool)
    print 'Created an effective layer: [is_conv {0}] [cover_map {1}]'.format(is_conv, self.cover_map.shape)

    
def main():
  parser=argparse.ArgumentParser(
          description='To encode an DNN given a reference input' )

  parser.add_argument('model', action='store', nargs='+', help='The input neural network model (.h5)')
  parser.add_argument("-i", "--inputs", dest="inputs",
                    help="the reference input seeds directory", metavar="DIR")
  #parser.add_argument("-m", "--max",
  #                  action="store_false", dest="Max", default=True,
  #                  help="to max/min ")


  args=parser.parse_args()
  inputs= args.inputs
  print inputs, type(inputs)
  #Max=args.Max
  #print Max, type(Max)

  if inputs is None:
    print 'No inputs are given...'
    sys.exit(0)

  inputFiles=[] 
  if inputs[-1]!='/':
    inputs+='/'
  for f in os.listdir(inputs):
    if not os.path.isdir(inputs+f):
      inputFiles.append(inputs+f)
  print inputFiles
  model = load_model(args.model[0])

  ### input size
  inps=model.layers[0].input.shape
  row, column, channel=inps[1], inps[2], inps[3]
  N=row*column*channel

  ### to get 'x's and 'd's from input files
  xs=[]
  ds=[]
  labels=[]

  for fname in inputFiles:
    ##  each input file f shall contain 3 lines
    ####  the first line contains N elements for 'x_0'
    ####  the second line contains N elements for 'd'
    ####  the third line is the target output index
    x0=np.zeros((1,N))
    d=np.zeros((1,N))
    lines = [line.rstrip('\n') for line in open(fname)]
    s=lines[0].split()
    for i in range(0, N):
      x0[0][i]=float(s[i])
    s=lines[1].split()
    for i in range(0, N):
      d[0][i]=float(s[i])
    x0=np.reshape(x0, (row, column, channel))
    d=np.reshape(d, (row, column, channel))
    xs.append(x0)
    ds.append(d)
    label=int(lines[2])
    labels.append(label)

  #### configuration phase
  layer_functions=[]
  effective_layers=[]

  for l in range(0, len(model.layers)):
    layer=model.layers[l]
    name=layer.name

    get_current_layer_output = K.function([layer.input], [layer.output])
    layer_functions.append(get_current_layer_output)

    if is_conv_layer(layer) or is_dense_layer(layer):
      effective_layers.append(effective_layert(layer_index=l, current_layer=layer, is_conv=is_conv_layer(layer)))

  ## a list of (min, max) to be printed into files
  lp_results=[]

  for i in range(0, len(xs)):
    activations=eval(layer_functions, xs[i]) 
    v_min=AP(model, activations, xs[i], labels[i], ds[i], False)
    v_max=AP(model, activations, xs[i], labels[i], ds[i], True)
    if not os.path.isdir(inputs+'results'):
      os.system('mkdir -p {0}'.format(inputs+'results'))
    with open(inputs+'results/'+inputFiles[i].split('/')[1], 'w') as the_file:
      the_file.write('{0}\n'.format(v_min))
      the_file.write('{0}\n'.format(v_max))

if __name__=="__main__":
  main()
  
