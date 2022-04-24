from torch import nn, Tensor
import torch
import random
from tqdm import trange

class MaxModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.conv_id = 0
        self.maxes = []
        self.mins = []
        
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self.get_max)
                self.maxes.append(0.)
                self.mins.append(0.)
        
    def get_max(self, module, input_value, output):
        max_val = torch.max(output).item()
        min_val = torch.min(output).item()
        self.maxes[self.conv_id] = max(max_val, self.maxes[self.conv_id])
        self.mins[self.conv_id] = min(min_val, self.mins[self.conv_id])
        self.conv_id += 1
    
    def reset_conv_id(self):
        self.conv_id = 0
                        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def get_maxes(self):
        return self.maxes
    
    def get_mins(self):
        return self.mins

def get_range(net_max, dataset, n=500):
    net_max.eval()
    new_net_max = MaxModel(net_max)

    sample_inds = random.sample(range(0, len(dataset)), n)

    print("Getting max...")
    for i in trange(n):
        img = torch.unsqueeze(dataset[sample_inds[i]]['image'], 0)
        new_net_max(img)
        new_net_max.reset_conv_id()

    return new_net_max.get_maxes(), new_net_max.get_mins()

def get_range_img(net_max, img):
    net_max.eval()
    new_net_max = MaxModel(net_max)
    img = torch.unsqueeze(img, 0)
    new_net_max(img)
    
    return new_net_max.get_maxes(), new_net_max.get_mins()