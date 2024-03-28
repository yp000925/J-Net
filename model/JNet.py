import torch
import torch.nn as nn
from utils.model_utils import *
from utils.general import *
from utils.dataset_utils import *
seed = 11
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class JNet(nn.Module):
    def __init__(self, config, **kwargs):
        super(JNet, self).__init__()
        self.model_W = create_model(config['MODEL_W'])
        self.model_U = create_model(config['MODEL_U'])
        self.num_params = network_parameters(self.model_W) + network_parameters(self.model_U)
        self.prop_kernel = config['MEASUREMENT']['prop_kernel']
        # self.depth_max = self.prop_kernel['distance']*2
        # self.depth_min = self.prop_kernel['z_min']
        # self.depth_max = self.prop_kernel['z_max']
        self.depth_min = self.prop_kernel['distance']*0.9
        self.depth_max = self.prop_kernel['distance']*1.1
        self.operator = None
        self.amp_pred = None
        self.phase_pred = None
        self.device = 'cpu'

    def forward(self, x):
        self.device = x.device
        depth_ratio_pred = self.model_U(x)
        self.amp_pred = self.model_W(x)[:,0,:,:].unsqueeze(1)# use relu activation
        self.phase_pred = self.model_W(x)[:,1,:,:].unsqueeze(1) # use sigmoid activation
        # self.phase_pred = self.rescale_phase(self.phase_pred, range=[0, torch.pi])
        # self.amp_pred = torch.ones_like(self.phase_pred)
        self.complex_map = torch.exp(2j*torch.pi*self.phase_pred)*self.amp_pred
        self.update_kernel(depth_ratio_pred)
        out = self.operator.forward(self.complex_map)
        return out

    def rescale_phase(self, phase, range=[-1, 1]):
        return (phase - phase.min()) / (phase.max() - phase.min()) * (range[1] - range[0]) + range[0]

    def update_kernel(self, depth_ratio, verbose=False):
        self.prop_kernel['distance'] = self.depth_min+(self.depth_max-self.depth_min)*depth_ratio
        self.operator = DH_operator(**self.prop_kernel,device =self.device)
        if verbose:
            print("The depth ratio is {}".format(depth_ratio))
            print("The updated depth is {}".format(self.prop_kernel['distance']))
            print("The z_max is {}".format(self.depth_max))
        return


    def summary(self):
        print("The number of parameters in model: Depth {} and Map {}".format(network_parameters(self.model_W),
                                                                   network_parameters(self.model_U)))


if __name__ == '__main__':
    from utils.model_utils import network_parameters
    import argparse
    import yaml

    height = 512
    width = 512
    x = torch.randn((1, 1, height, width))  # .cuda()
    gt = torch.randn((1, 1, height, width))  # .cuda()

    parser = argparse.ArgumentParser()
    pth = "/Users/zhangyunping/PycharmProjects/PATrans/configs/AdaptDH_C.yaml"
    with open(pth) as f:
        model_config = yaml.load(f,Loader=yaml.FullLoader)
    model = JNet(model_config)
    model.summary()
    y = model(x)
    # loss = nn.MSELoss()(y, gt)
    # loss.backward()