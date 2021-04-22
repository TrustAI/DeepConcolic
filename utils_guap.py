'''Some helper functions for PyTorch
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.1)



def flow_st(images, flows):

    batch_size,_,H,W = images.size()

    device = images.device


    # basic grid: tensor with shape (2, H, W) with value indicating the
    # pixel shift in the x-axis or y-axis dimension with respect to the
    # original images for the pixel (2, H, W) in the output images,
    # before applying the flow transforms
    grid_single = torch.stack(
                torch.meshgrid(torch.arange(0,H), torch.arange(0,W))
            ).float()


    grid = grid_single.repeat(batch_size, 1, 1, 1)#100,2,28,28

    images = images.permute(0,2,3,1) #100, 28,28,1

    grid = grid.to(device)

    grid_new = grid + flows
    # assert 0

    sampling_grid_x = torch.clamp(
        grid_new[:, 1], 0., (W - 1.)
            )
    sampling_grid_y = torch.clamp(
        grid_new[:, 0], 0., (H - 1.)
    )
    
    # now we need to interpolate

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a square around the point of interest
    x0 = torch.floor(sampling_grid_x).long()
    x1 = x0 + 1
    y0 = torch.floor(sampling_grid_y).long()
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate image boundaries
    # - 2 for x0 and y0 helps avoiding black borders
    # (forces to interpolate between different points)
    x0 = torch.clamp(x0, 0, W - 2)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 2)
    y1 = torch.clamp(y1, 0, H - 1)


    b =torch.arange(0, batch_size).view(batch_size, 1, 1).repeat(1, H, W).to(device)
    # assert 0 
    Ia = images[b, y0, x0].float()
    Ib = images[b, y1, x0].float()
    Ic = images[b, y0, x1].float()
    Id = images[b, y1, x1].float()


    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
    wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
    wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
    wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)

    # add dimension for addition
    wa = wa.unsqueeze(3)
    wb = wb.unsqueeze(3)
    wc = wc.unsqueeze(3)
    wd = wd.unsqueeze(3)

    # compute output
    perturbed_image = wa * Ia+ wb * Ib+ wc * Ic+wd * Id
 

    perturbed_image = perturbed_image.permute(0,3,1,2)

    return perturbed_image


class Loss_flow(nn.Module):
    def __init__(self, neighbours=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])):
        super(Loss_flow, self).__init__()

    def forward(self, flows):
    

        paddings = (1, 1, 1, 1,0, 0, 0, 0)
        padded_flows = F.pad(flows,paddings, "constant", 0)


        # #rook
        shifted_flowsr = torch.stack([
                    padded_flows[:, :, 2:, 1:-1],  # bottom mid
                    padded_flows[:, :, 1:-1, :-2],  # mid left
                    padded_flows[:, :, :-2, 1:-1],   # top mid
                    padded_flows[:, :, 1:-1, 2:],  # mid right
                    ],-1) 

        flowsr = flows.unsqueeze(-1).repeat(1,1,1,1,4)
        _,h,w,_ = flowsr[:,0].shape


        loss0 = torch.norm((flowsr[:,0] - shifted_flowsr[:,0]).view(-1,4), p = 2, dim=(0), keepdim=True) ** 2
        loss1 = torch.norm((flowsr[:,1] - shifted_flowsr[:,1]).view(-1,4), p = 2, dim=(0), keepdim=True) ** 2
        
        return torch.max(torch.sqrt((loss0+loss1)/(h*w)))

def cal_l2dist(X1,X2):
    list_bhat = []
    list_hdist = []
    list_ssim = []
    list_l2 = []
    batch,nc,_,_ = X1.shape

    for i in range (batch):     
        img1 = X1[i].unsqueeze(0)
        img2 = X2[i].unsqueeze(0)    
        x1 = img1.mul(255).clamp(0, 255).permute(0,2,3,1).to('cpu', torch.uint8).numpy()
        x2 = img2.mul(255).clamp(0, 255).permute(0,2,3,1).to('cpu', torch.uint8).numpy()
        list_l2.append(np.sqrt(np.sum( (x1 - x2)**2 )))
    return np.mean(list_l2)

def norm_ip(img):
    min = float(img.min())
    max = float(img.max())
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)
    return img


