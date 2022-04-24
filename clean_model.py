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

        # Register a hook for each layer
        num_conv = 0
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self.run)
                weight_el = layer.weight.numel()
                weight_zeros = num_nonzeros(layer.weight)
                self.weight_zeros.append((weight_zeros, weight_el))
                num_conv += 1
    
    def run(self, module, input_value, output):
        self.clean_outputs.append(output.detach().clone())
        self.input_zeros.append((num_nonzeros(input_value[0]), input_value[0].numel()))
        self.output_zeros.append((num_nonzeros(output), output.numel()))
    
    def reset(self):
        self.clean_outputs = []
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def get_clean_outputs(self):
        return self.clean_outputs

    def get_clean_output(self, conv_id):
        return self.clean_outputs[conv_id]
    
    def get_nonzero(self, conv_id):
        return (self.output_zeros[conv_id], self.input_zeros[conv_id], self.weight_zeros[conv_id])
    
    def get_nonzeros(self):
        return (self.output_zeros, self.input_zeros, self.weight_zeros)

def run_clean(clean_net, test_img, conv_id=-1):
    clean_net.reset()
    with torch.no_grad():
        clean_out = clean_net(test_img)

    if conv_id == -1:
        return clean_net.get_clean_outputs(), clean_net.get_nonzeros()
    else:
        return clean_net.get_clean_output(conv_id), clean_net.get_nonzero(conv_id)