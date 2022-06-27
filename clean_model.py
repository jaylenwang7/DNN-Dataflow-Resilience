from torch import nn, Tensor
import torch
from helpers import num_nonzeros

class CleanModel(nn.Module):
    def __init__(self, model: nn.Module, process_FC=False):
        super().__init__()
        self.model = model
        self.clean_outputs = []
        self.weight_zeros = []
        self.output_zeros = []
        self.input_zeros = []
        self.process_FC = process_FC
        self.target_id = -1
        self.layer_ind = 0

        # Register a hook for each layer
        num_layer = 0
        for layer in self.model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.register_forward_hook(self.run)
                weight_el = layer.weight.numel()
                weight_zeros = num_nonzeros(layer.weight)
                self.weight_zeros.append((weight_zeros, weight_el))
                num_layer += 1
    
    def run(self, module, input_value, output):
        # if at target layer, then record the output and zeros
        if self.layer_ind == self.target_id or self.target_id == -1:
            self.clean_outputs.append(output.detach().clone())
            self.input_zeros.append((num_nonzeros(input_value[0]), input_value[0].numel()))
            self.output_zeros.append((num_nonzeros(output), output.numel()))
        else:
            self.clean_outputs.append(None)
            self.input_zeros.append(None)
            self.output_zeros.append(None)
        self.layer_ind += 1
    
    def set_target_id(self, target_id=-1):
        self.target_id = target_id
    
    def reset(self):
        self.clean_outputs = []
        self.output_zeros = []
        self.input_zeros = []
        self.layer_ind = 0
        self.target_id = -1
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def get_clean_outputs(self):
        return self.clean_outputs

    def get_clean_output(self, layer_id):
        return self.clean_outputs[layer_id]
    
    def get_nonzero(self, layer_id):
        return (self.output_zeros[layer_id], self.input_zeros[layer_id], self.weight_zeros[layer_id])
    
    def get_nonzeros(self):
        return (self.output_zeros, self.input_zeros, self.weight_zeros)

def run_clean(clean_net, test_img, layer_id=-1):
    clean_net.reset()
    clean_net.set_target_id(layer_id)
    with torch.no_grad():
        clean_out = clean_net(test_img)

    if layer_id == -1:
        return clean_out, clean_net.get_clean_outputs(), clean_net.get_nonzeros()
    else:
        return clean_out, clean_net.get_clean_output(layer_id), clean_net.get_nonzero(layer_id)