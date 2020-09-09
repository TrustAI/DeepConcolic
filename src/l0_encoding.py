import numpy as np
from utils import *

class L0Analyzer:
  """
  Custom analyzer for generating new inputs, based on the L0 norm.
  """

  def __init__(self, input_shape, eval_batch, gran = 2):
    self.shape = input_shape
    self.dim_row = min(input_shape[0], DIM)
    self.dim_col = min(input_shape[1], DIM)
    [x, y] = np.meshgrid(np.arange(self.dim_row), np.arange(self.dim_col))
    xflat = x.flatten('F')          # to flatten in column-major order
    yflat = y.flatten('F')          # to flatten in column-major order
    self.xflat = np.split(xflat, len(xflat))
    self.yflat = np.split(yflat, len(yflat))
    self.gran = gran
    self.sort_size = self.dim_row * self.dim_col * self.gran
    self.eval_batch = eval_batch
    super().__init__()


  def eval_change(self, images, n, target):
    nc_layer, pos = target.layer, target.position
    row, col, chl = self.shape
    activations = self.eval_batch (images.reshape(n, row, col, chl))
    activations = activations [nc_layer.layer_index]
    return (activations[:, pos[1], pos[2], pos[3]] if nc_layer.is_conv else
            activations[:, pos[1]])


  def sort_pixels(self, image, nc_target):
    row, col, chl = self.shape

    sort_list = np.linspace(0, 1, self.gran)
    image_batch = np.kron(np.ones((self.gran, 1, 1, 1)), image)

    selected_rows = np.random.choice(row, self.dim_row)
    selected_cols = np.random.choice(col, self.dim_col)
    images = []
    for i in selected_rows:
      for j in selected_cols:
        new_image_batch = image_batch.copy()
        for g in range(0, self.gran):
          new_image_batch[g, i, j, :] = sort_list[g]
        images.append(new_image_batch)

    images = np.asarray (images)
    target_change = (self.eval_change (images, self.sort_size, nc_target)
                     .reshape(-1, self.gran).transpose())

    min_indices = np.argmax(target_change, axis=0)
    min_values = np.amax(target_change, axis=0)
    min_idx_values = min_indices.astype('float32') / (self.gran - 1)
    
    target_list = np.hstack((self.xflat, self.yflat,
                             np.split(min_values, len(min_values)),
                             np.split(min_idx_values, len(min_idx_values))))
  
    sorted_map = target_list[(target_list[:, 2]).argsort()]
    sorted_map = np.flip(sorted_map, 0)
    for i in range(0, len(sorted_map)):
      sorted_map[i][0]=selected_rows[int(sorted_map[i][0])]
      sorted_map[i][1]=selected_cols[int(sorted_map[i][1])]
  
    return sorted_map


  def accumulate(self, image, nc_target, sorted_pixels, mani_range):
    row, col, chl = self.shape

    images = []
    mani_image = image.copy()
    for i in range(0, mani_range):
      pixel_row = sorted_pixels[i, 0].astype('int')
      pixel_col = sorted_pixels[i, 1].astype('int')
      pixel_value = sorted_pixels[i, 3]
      mani_image[pixel_row][pixel_col] = pixel_value
      images.append (mani_image.copy ())

    images = np.asarray(images)
    nc_acts = self.eval_change (images, len (images), nc_target)
  
    adversarial_images = images[nc_acts > 0, :, :]
    if adversarial_images.any():
      success_flag=True
      idx_first=np.amin((nc_acts>0).nonzero(), axis=1)
    else:
      success_flag=False
      idx_first=np.nan
  
    return adversarial_images, idx_first, success_flag


  def refine_act_image(self, image, nc_target, sorted_pixels, act_image_first, idx_first):
    row, col, chl = self.shape

    refined_act_image = act_image_first.copy()
    total_idx = 0
    idx_range = np.arange(idx_first)
    while True:
      length = len(idx_range)
      #print ('idx_first: ', idx_first)
      for i in range(0, idx_first[0]):
        pixel_row = sorted_pixels[i, 0].astype('int')
        pixel_col = sorted_pixels[i, 1].astype('int')
        refined_act_image[pixel_row, pixel_col] = image[pixel_row, pixel_col]
        
        refined_activation = self.eval_change (refined_act_image, 1, nc_target)

        if refined_activation < 0:  # == label:
          refined_act_image[pixel_row, pixel_col] = sorted_pixels[i, 3]
        else:
          total_idx = total_idx + 1
          idx_range = idx_range[~(idx_range == i)]
  
      if len(idx_range) == length:
        break
  
    return refined_act_image



# ---
