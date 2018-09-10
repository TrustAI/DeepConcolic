
import numpy as np
from utils import *

def sort_pixels(dnn, layer_functions, image, nc_layer, pos, gran=2):
  sort_list=np.linspace(0, 1, gran)
  image_batch = np.kron(np.ones((gran, 1, 1, 1)), image)
  images=[]
  (row, col, chl) = image.shape
  dim_row, dim_col=row, col
  if row>DIM: dim_row=DIM
  if col>DIM: dim_col=DIM
  selected_rows=np.random.choice(row, dim_row)
  selected_cols=np.random.choice(col, dim_col)
  for i in selected_rows:
    for j in selected_cols:
      new_image_batch = image_batch.copy()
      for g in range(0, gran):
         new_image_batch[g, i, j, :] = sort_list[g]
      images.append(new_image_batch)
  images=np.asarray(images)
  
  images = images.reshape(dim_row * dim_col * gran, row, col, chl)

  activations = eval_batch(layer_functions, images)

  target_list = activations[nc_layer.layer_index]
  osp=activations[nc_layer.layer_index].shape
  index=np.unravel_index(pos, osp)
  target_change=None
  if nc_layer.is_conv:
    target_change = target_list[:, index[1], index[2], index[3]].reshape(-1, gran).transpose()
  else:
    target_change = target_list[:, index[1]].reshape(-1, gran).transpose()

  min_indices = np.argmax(target_change, axis=0)
  min_values = np.amax(target_change, axis=0)
  min_idx_values = min_indices.astype('float32') / (gran - 1)

  [x, y] = np.meshgrid(np.arange(dim_row), np.arange(dim_col))
  x = x.flatten('F')  # to flatten in column-major order
  y = y.flatten('F')  # to flatten in column-major order

  target_list = np.hstack((np.split(x, len(x)),
                                     np.split(y, len(y)),
                                     np.split(min_values, len(min_values)),
                                     np.split(min_idx_values, len(min_idx_values))))
  sorted_map = target_list[(target_list[:, 2]).argsort()]
  sorted_map = np.flip(sorted_map, 0)
  for i in range(0, len(sorted_map)):
    sorted_map[i][0]=selected_rows[int(sorted_map[i][0])]
    sorted_map[i][1]=selected_cols[int(sorted_map[i][1])]

  return sorted_map

def accumulate(dnn, layer_functions, image, nc_layer, pos, sorted_pixels, mani_range):
  images=[]
  mani_image=image.copy()
  for i in range(0, mani_range):
    pixel_row = sorted_pixels[i, 0].astype('int')
    pixel_col = sorted_pixels[i, 1].astype('int')
    pixel_value = sorted_pixels[i, 3]
    mani_image[pixel_row][pixel_col] = pixel_value
    images.append(mani_image.copy())

  images = np.asarray(images)
  (row, col, chl) = image.shape
  activations = eval_batch(layer_functions, images.reshape(len(images), row, col, chl))

  osp=activations[nc_layer.layer_index].shape
  index=np.unravel_index(pos, osp)
  nc_acts=None
  if nc_layer.is_conv:
    nc_acts = activations[nc_layer.layer_index][:, index[1], index[2], index[3]]
  else:
    nc_acts = activations[nc_layer.layer_index][:, index[1]]

  adversarial_images = images[nc_acts> 0, :, :]

  if adversarial_images.any():
    success_flag=True
    idx_first=np.amin((nc_acts>0).nonzero(), axis=1)
  else:
    success_flag=False
    idx_first=np.nan

  return adversarial_images, idx_first, success_flag

def refine_act_image(dnn, layer_functions, image, nc_layer, pos, sorted_pixels, act_image_first, idx_first):
  (row, col, chl) = image.shape
  refined_act_image=act_image_first.copy()
  total_idx=0
  idx_range=np.arange(idx_first)
  while True:
    length=len(idx_range)
    #print ('idx_first: ', idx_first)
    for i in range(0, idx_first[0]):
      pixel_row = sorted_pixels[i, 0].astype('int')
      pixel_col = sorted_pixels[i, 1].astype('int')
      refined_act_image[pixel_row, pixel_col] = image[pixel_row, pixel_col]
      activations = eval_batch(layer_functions, refined_act_image.reshape(1, row, col, chl))
      osp=activations[nc_layer.layer_index].shape
      index=np.unravel_index(pos, osp)
      refined_activation=None
      if nc_layer.is_conv:
        refined_activation = activations[nc_layer.layer_index][0][index[1]][index[2]][index[3]]
      else:
        refined_activation = activations[nc_layer.layer_index][0][index[1]]
      if refined_activation < 0:  # == label:
        refined_act_image[pixel_row, pixel_col] = sorted_pixels[i, 3]
      else:
        total_idx = total_idx + 1
        idx_range = idx_range[~(idx_range == i)]

    if len(idx_range) == length:
      break



  return refined_act_image



