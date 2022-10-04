from xxlimited import Str
import torch
from torch import nn, Tensor
import copy
import bitflip
from typing import Union

# object to inject into a single layer
class InjectModel(nn.Module):
    def __init__(self, model: nn.Module, layer_id, d_type='i', device=None):
        super().__init__()
        
        # initialize all params
        self.layer_id = layer_id        # id of the layer to be injected into
        self.model = model              # holds a copy of the model 
        self.layer = 0                  # holds a copy of the layer
        self.sites = 0
        self.outputs = []
        self.inj_coord = 0
        self.mode = -1
        self.bit = -1
        self.change_to = 1000.
        self.pre_values = []
        self.post_values = []
        self.max_vals = []
        self.min_vals = []
        self.range_max = False
        self.layer_ind = 0
        self.all_outs = False
        self.out_max = 0
        self.out_min = 0
        self.is_FC = False
        self.FC_size = -1
        self.device = ""
        
        # set what type of injection based on user input
        self.set_d_type(d_type)
        # set device to run on
        self.set_device(device)

        # register a hook for each layer
        with torch.no_grad():
            num_layer = 0
            for layer in self.model.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    if num_layer == layer_id:
                        if isinstance(layer, (nn.Linear)):
                            self.is_FC = True
                        self.layer = copy.deepcopy(layer).to(self.device)
                        self.layer.weight.requires_grad = False
                        layer.register_forward_hook(self.inject)
                    else:
                        layer.register_forward_hook(self.compare)
                    num_layer += 1
            self.num_layer = num_layer
        
    def set_device(self, device):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
            
    def get_is_FC(self):
        return self.is_FC
    
    def set_FC_size(self, FC_size):
        assert(self.is_FC)
        self.FC_size = FC_size
        
    def get_FC_size(self):
        assert(self.is_FC)
        return self.FC_size
    
    # set the maximum value for ranging
    # if max_vals == -1, this disables ranging
    def set_range(self, max_vals=[], min_vals=[]):
        # this turns off max ranging
        if not max_vals and not min_vals:
            self.max_vals = []
            self.min_vals = []
            self.range_max = False
            return
        
        # set range maxing on - and then set the right values
        self.range_max = True
        if not min_vals:
            self.max_vals = max_vals
            self.min_vals = [None]*self.num_layer
        elif not max_vals:
            self.min_vals = min_vals
            self.max_vals = [None]*self.num_layer
        else:
            self.min_vals = min_vals
            self.max_vals = max_vals
        # make sure that there's a max/min value for each layer
        assert(len(self.min_vals) == self.num_layer)
        assert(len(self.max_vals) == self.num_layer)
        
    def bitflip_value(self, value):
        # for data recording purposes - keep the original value
        self.pre_values.append(value.item())
        
        # change the value depending on the mode
        value = value.to(self.device)
        if self.mode == 0:
            value = bitflip.flip_bit(value, self.bit)
        elif self.mode == 1:
            value = bitflip.flip_random_bit(value)
        elif self.mode == 2:
            value = torch.as_tensor(self.change_to)
        else:
            assert(False)
        # record changed value
        self.post_values.append(value.item())
        return value
    
    # a hook function that will perform HW injection (given some SW error model)
    def inject(self, module, input_value, output):
        # in here you want to:
        #   1. if injecting into input - (bit flip) one of the data elements
        #   2. run a faulty Conv2d with the injected data (either input or weight)
        #   3. overwrite the output with the given sites from the error output (from 2)
        #   4. if using Ranger - clamp the new injected values within the range
        #   5. add a copy of the output to the list of conv outputs
        
        # 1 ===========
        input_tensor = input_value[0].clone().detach().to(self.device)
        batch_size = input_tensor.shape[0]
        
        # if injecting into input - need to do this online during the hook
        if self.d_type == 'i':
            for i in range(batch_size):
                inject_val = input_tensor[i][self.inj_coord]
                input_tensor[i][self.inj_coord] = self.bitflip_value(inject_val)
        
        if self.d_type != 'o':
            # 2 ===========
            faulty_output = self.layer(input_tensor)
            
            # 3 =========== 
            # if the list of sites is not empty
            if self.sites:
                for i in range(batch_size):
                    for site in self.sites:
                        output[i][site] = faulty_output[i][site]
            else:
                # if empty list is given - then just directly copy (don't pick any sites)
                output.copy_(faulty_output)
        else:
            for i in range(batch_size):
                inject_val = output[i][self.inj_coord]
                output[i][self.inj_coord] = self.bitflip_value(inject_val)
        
        # 4 ===========
        if self.range_max:
            max_val = self.max_vals[self.layer_id]
            min_val = self.min_vals[self.layer_id]
            
            # uncomment below for clamp checking
            # clamped_output = torch.clamp(output, min=min_val, max=max_val)
            # num_diff = compare_outputs(output, clamped_output)
            # print("NUM_DIFF: " + str(num_diff))
            # print("MAX VAL: " + str(max_val))
            # print("MIN VAL: " + str(min_val))
            # assert(False)
            
            output.copy_(torch.clamp(output, min=min_val, max=max_val))
        
        # add the resulting max and min, to see if the clamp worked, for debugging purposes
        output_dims = list(range(1, len(output.size())))
        self.maxes = torch.amax(output, dim=output_dims).to("cpu").tolist()
        self.mins = torch.amin(output, dim=output_dims).to("cpu").tolist()
            
        # 5 ===========
        if self.layer_ind == self.layer_id or self.all_outs:
            self.outputs.append(copy.deepcopy(output))
        else:
            self.outputs.append(None)
        
        self.layer_ind += 1
    
    # hook function for layers not being injected into
    # this for: 1) doing ranging, 2) for data collection
    def compare(self, module, input_value, output):
        if self.range_max:
            max_val = self.max_vals[self.layer_ind]
            min_val = self.min_vals[self.layer_ind]
            output.copy_(torch.clamp(output, min=min_val, max=max_val))
            
        if self.layer_ind == self.layer_id or self.all_outs:
            self.outputs.append(copy.deepcopy(output))
        else:
            self.outputs.append(None)
            
        self.layer_ind += 1
    
    # set the mode of 
    def set_mode(self, mode: str, change_to: int=1000., bit: Union[int, range]=-1):
        if mode == "bit":
            assert(bit != -1)
            self.bit = bit
            self.mode = 0
        elif mode == "rand_bit":
            assert(False and "not implemented/deprecated - use bit mode with a range instead")
            self.mode = 1
        elif mode == "change_to":
            self.mode = 2
            self.change_to = change_to
        else:
            assert(False)
            
    def set_d_type(self, d_type: str):
        assert(d_type in ['i', 'w', 'o'])
        self.d_type = d_type
            
    def set_sites(self, sites):
        self.sites = sites
    
    def get_outputs(self):
        return self.outputs

    def get_output(self, layer_id=-1):
        if layer_id == -1:
            layer_id = self.layer_id
        return self.outputs[layer_id]
    
    # called before each hook call
    # only reset values for things that change between injection calls
    def reset(self):
        if self.d_type == 'w' and not self.pre_values == []:
            self.reset_weight()
        self.outputs = []
        self.pre_values = []
        self.post_values = []
        self.inj_coord = 0
        self.sites = []
        self.mode = -1
        self.bit = -1
        self.change_to = 1000.
        self.layer_ind = 0
        self.out_max = 0
        self.out_min = 0
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def reset_weight(self):
        self.layer.weight[self.inj_coord] = self.pre_value
    
    # inject into a weight offline - so not as part of the hook function
    def inject_weight(self, inj_coord):
        assert(self.d_type == 'w')
        # if the weight has been injected, reset the weights to the original value
        if not self.pre_value == []:
            self.reset_weight()
        
        # update inj_coord
        self.inj_coord = inj_coord
        with torch.no_grad():
            # get the clean value
            inject_val = self.layer.weight[self.inj_coord]
            self.layer.weight[self.inj_coord] = self.bitflip_value(inject_val)
    
    def get_weight(self):
        return self.layer.weight.detach().clone()
    
    def run_hook(self, test_img, inj_coord, sites, mode="change_to", change_to=1000., bit=-1):
        self.reset()
        if not self.is_FC:
            # make sure the output indices have right dim
            if sites and self.d_type != 'o':
                assert(len(sites[0]) == 3)
            # make sure inj_coord has right dim
            if self.d_type == 'i':
                assert(len(inj_coord) == 3)
            elif self.d_type == 'w':
                assert(len(inj_coord) == 4)
            else:
                assert(len(inj_coord) == 3)
        else:
            assert(self.FC_size != -1)
            out_in_size = self.FC_size - 1
            # make sure the output indices have right dim
            if sites and self.d_type != 'o':
                assert(len(sites[0]) == out_in_size)
            # make sure inj_coord has right dim
            if self.d_type == 'i':
                assert(len(inj_coord) == out_in_size)
            elif self.d_type == 'w':
                assert(len(inj_coord) == 2)
            else:
                assert(len(inj_coord) == out_in_size)
            
        self.inj_coord = inj_coord
        self.set_sites(sites)
        self.set_mode(mode, change_to, bit)
        
        if self.d_type == 'w':
            self.inject_weight(self.inj_coord)
        
        test_img = test_img.to(self.device)
        with torch.no_grad():
            out = self.forward(test_img)

        return (out, self.pre_values, self.post_values)
    
    def get_maxmin(self):
        return self.maxes, self.mins