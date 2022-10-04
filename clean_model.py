from torch import nn, Tensor
import torch
from helpers import num_nonzeros

class CleanModel(nn.Module):
    """Wrapper of the PyTorch nn.Module that is used to get the clean outputs of layers and other info about the native model
    without any injections.
    """
    
    def __init__(self, model: nn.Module, process_FC: bool=False, device=None):
        """Initializes a CleanModel instance.

        Args:
            model (nn.Module): The network that this wrapper wraps around.
            process_FC (bool, optional): Sets whether to process FC layers (CONV layers always processed). Defaults to False.
        """
           
        super().__init__()
        self.model = model
        self.clean_outputs = []
        self.weight_zeros = []
        self.output_zeros = []
        self.input_zeros = []
        self.process_FC = process_FC
        self.target_id = -1
        self.layer_ind = 0
        self.device = "cpu"
        self.set_device(device)

        # Register a hook for each layer
        num_layer = 0
        for layer in self.model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.register_forward_hook(self.run)
                weight_el = layer.weight.numel()
                weight_zeros = num_nonzeros(layer.weight, dims=None)
                self.weight_zeros.append((weight_zeros, weight_el))
                num_layer += 1
                
    def set_device(self, device):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
    
    def run(self, module, input_value, output) -> None:
        # if at target layer, then record the output and zeros
        if self.layer_ind == self.target_id or self.target_id == -1:
            self.clean_outputs.append(output.detach().clone())
            self.input_zeros.append((num_nonzeros(input_value[0], dims=list(range(1, len(input_value[0].size())))), input_value[0][0].numel()))
            self.output_zeros.append((num_nonzeros(output, dims=list(range(1, len(output.size())))), output[0].numel()))
        else:
            self.clean_outputs.append(None)
            self.input_zeros.append(None)
            self.output_zeros.append(None)
        self.layer_ind += 1
    
    def set_target_id(self, target_id: int=-1) -> None:
        self.target_id = target_id
    
    def reset(self) -> None:
        self.clean_outputs = []
        self.output_zeros = []
        self.input_zeros = []
        self.layer_ind = 0
        self.target_id = -1
        
    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        return self.model(x)
    
    def get_clean_outputs(self):
        return self.clean_outputs

    def get_clean_output(self, layer_id: int):
        return self.clean_outputs[layer_id]
    
    def get_nonzero(self, layer_id: int):
        return (self.output_zeros[layer_id], self.input_zeros[layer_id], self.weight_zeros[layer_id])
    
    def get_nonzeros(self):
        return (self.output_zeros, self.input_zeros, self.weight_zeros)

    def run_clean(self, img, layer_id: int=-1):
        self.reset()
        self.set_target_id(layer_id)
        with torch.no_grad():
            clean_out = self.forward(img)

        if layer_id == -1:
            return clean_out, self.get_clean_outputs(), self.get_nonzeros()
        else:
            return clean_out, self.get_clean_output(layer_id), self.get_nonzero(layer_id)