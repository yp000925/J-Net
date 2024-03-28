import torch
import torch.nn as nn


def FDMAE_LOSS(x,y):
    _,_,h,w = x.shape
    window = torch.outer(torch.hann_window(h), torch.hann_window(w))
    window = torch.fft.ifftshift(window).unsqueeze(0).unsqueeze(0).to(x.device)
    x = torch.fft.fft2(x)
    y = torch.fft.fft2(y)
    fdmae_loss = nn.L1Loss(reduction='mean')(x*window, y*window)
    return fdmae_loss

def TV_LOSS(x):
    b,c,h,w = x.shape
    grad_x = x[:,:,1:,:]-x[:,:,:-1,:]
    grad_y = x[:,:,:,1:]-x[:,:,:,:-1]
    tv = (grad_x.abs().sum()+grad_y.abs().sum())/(b*c*h*w)
    return tv


if  __name__ == "__main__":
    x = torch.randn((1,3,256,256))
    y = torch.randn((1,3,256,256))
    loss = FDMAE_LOSS(x,y)
    print(loss)

