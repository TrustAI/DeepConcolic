from typing import *
from utils import *

assert False

# ---

def ssc_setup(test_object, outs):
  # print('\n== MC/DC (ssc) coverage for neural networks ==\n')
  results, _ = setup_output_files (outs, 'ssc_report')

  print('\n== Total layers: {0} ==\n'.format(len(get_layer_functions(test_object.dnn))))
  cover_layers = get_cover_layers (test_object.dnn, constr = cover_layert,
                                   layer_indices = test_object.layer_indices,
                                   exclude_direct_input_succ = True)
  print('\n== Cover-able layers: {0} ==\n'.format(len(cover_layers)))

  for i in range(0, len(cover_layers)):
    cover_layers[i].initialize_ubs()
    cover_layers[i].initialize_ssc_map((test_object.layer_indices, test_object.feature_indices))

  #print ("to compute the ubs")
  activations=None
  if not test_object.training_data is None:
    for x in test_object.training_data:
      x_acts = test_object.eval_batch (np.array([x]), allow_input_layer = True)
      for i in range(1, len(cover_layers)):
        #print (type(x_acts[cover_layers[i].layer_index][0]))
        #print (type(cover_layers[i].ubs))
        cover_layers[i].ubs=np.maximum(cover_layers[i].ubs, x_acts[cover_layers[i].layer_index][0])
  #print ("done")
  #  tot_size=len(test_object.training_data)
  #  batches=np.array_split(test_object.training_data[0:tot_size], tot_size//10 + 1)
  #  for i in range(0, len(batches)):
  #    batch=batches[i]
  #    sub_acts=eval_batch(layer_functions, batch, is_input_layer(test_object.dnn.layers[0]))
  #    if i==0:
  #      activations=sub_acts
  #    else:
  #      for j in range(0, len(activations)):
  #        activations[j]=np.concatenate((activations[j], sub_acts[j]), axis=0)

  return results, cover_layers, activations
