import torch
import torch
from torch import nn, Tensor
import copy
import bitflip
from helpers import *

# object to inject into a single conv layer
class InjectConvLayer(nn.Module):
    def __init__(self, model: nn.Module, conv_id, inj_loc='i'):
        super().__init__()
        
        # initialize all params
        self.conv_id = conv_id
        self.model = model
        self.conv = 0
        self.sites = 0
        self.outputs = []
        self.inj_coord = 0
        self.mode = -1
        self.bit = -1
        self.change_to = 1000.
        self.pre_value = []
        self.post_value = []
        self.max_vals = []
        self.min_vals = []
        self.range_max = False
        self.conv_ind = 0
        self.all_outs = False
        
        # set what type of injection based on user input
        self.inj_loc = inj_loc

        # register a hook for each layer
        with torch.no_grad():
            num_conv = 0
            for layer in self.model.modules():
                if isinstance(layer, nn.Conv2d):
                    if num_conv == conv_id:
                        self.conv = copy.deepcopy(layer)
                        self.conv.weight.requires_grad = False
                        layer.register_forward_hook(self.inject)
                    else:
                        layer.register_forward_hook(self.compare)
                    num_conv += 1
            self.num_conv = num_conv
    
    # set the maximum value for ranging
    # if max_vals == -1, this disables ranging
    def set_range(self, max_vals=[], min_vals=[]):
        # this turns off max ranging
        if not max_vals and not min_vals:
            self.max_vals = []
            self.min_vals = []
            self.range_max = False
            return
        
        # set it - and indicate using ranger
        if not self.min_vals:
            self.max_vals = max_vals
            self.min_vals = [None]*self.num_conv
        elif not self.max_vals:
            self.min_vals = min_vals
            self.max_vals = [None]*self.num_conv
        else:
            self.min_vals = min_vals
            self.max_vals = max_vals
            self.range_max = True
        assert(len(self.min_vals) == self.num_conv)
        assert(len(self.max_vals) == self.num_conv)
    
    # a hook function that will perform HW injection (given some SW error model)
    def inject(self, module, input_value, output):
        # in here you want to:
        #   1. if injecting into input - (bit flip) one of the data elements
        #   2. run a faulty Conv2d with the injected data (either input or weight)
        #   3. overwrite the output with the given sites from the error output (from 2)
        #   4. if using Ranger - clamp the new injected values within the range
        #   5. add a copy of the output to the list of conv outputs
        
        # 1 ===========
        input_tensor = input_value[0].clone().detach()
        
        # if injecting into input - need to do this online during the hook
        if self.inj_loc == 'i':
            inject_val = 0
            inject_val = input_tensor[0][self.inj_coord]
            
            # for data recording purposes - get the original value
            self.pre_value = inject_val.detach().clone()
            
            # change the value depending on the mode
            if self.mode == 0:
                try:
                    inject_val = bitflip.flip_bit(inject_val, self.bit)
                except:
                    print(inject_val)
                    print(self.bit)
                    assert(False)
            elif self.mode == 1:
                inject_val = bitflip.flip_random_bit(inject_val)
            elif self.mode == 2:
                inject_val = torch.as_tensor(self.change_to)
            else:
                assert(False)
            # record changed value
            self.post_value = inject_val
            
            # inject the new value into the input
            input_tensor[0][self.inj_coord] = inject_val
        
        # 2 ===========
        faulty_output = self.conv(input_tensor)
        
        # 3 =========== 
        # if the list of sites is not empty
        if self.sites:
            for site in self.sites:
                ind = (0, site[0], site[1], site[2])
                try:
                    output[ind] = faulty_output[ind]
                except:
                    print(ind)
                    print(self.inj_coord)
                    assert(False)
        else:
            # if empty list is given - then just directly copy (don't pick any sites)
            output.copy_(faulty_output)
        
        # 4 ===========
        if self.range_max:
            max_val = self.max_vals[self.conv_id]
            min_val = self.min_vals[self.conv_id]
            
            # uncomment below for clamp checking
            # clamped_output = torch.clamp(output, min=min_val, max=max_val)
            # num_diff = compare_outputs(output, clamped_output)
            # print("NUM_DIFF: " + str(num_diff))
            # print("MAX VAL: " + str(max_val))
            # print("MIN VAL: " + str(min_val))
            # assert(False)
            
            output.copy_(torch.clamp(output, min=min_val, max=max_val))
            
        # 5 ===========
        if self.conv_ind == self.conv_id or self.all_outs:
            self.outputs.append(copy.deepcopy(output))
        else:
            self.outputs.append(None)
        
        self.conv_ind += 1
    
    # hook function for layers not being injected into
    # this for: 1) doing ranging, 2) for data collection
    def compare(self, module, input_value, output):
        if self.range_max:
            max_val = self.max_vals[self.conv_ind]
            min_val = self.min_vals[self.conv_ind]
            output.copy_(torch.clamp(output, min=min_val, max=max_val))
            
        if self.conv_ind == self.conv_id or self.all_outs:
            self.outputs.append(copy.deepcopy(output))
        else:
            self.outputs.append(None)
            
        self.conv_ind += 1
    
    # set the mode of 
    def set_mode(self, mode, change_to=1000., bit=-1):
        if mode == "bit":
            assert(bit != -1)
            self.bit = bit
            self.mode = 0
        elif mode == "rand_bit":
            self.mode = 1
        elif mode == "change_to":
            self.mode = 2
            self.change_to = change_to
        else:
            assert(False)
            
    def set_loc(self, loc):
        assert(loc in ['i', 'w', 'o'])
        if self.inj_loc == 'w' and not self.pre_value == []:
            self.reset_weight()
        self.inj_loc = loc
            
    def set_sites(self, sites):
        self.sites = sites
    
    def get_outputs(self):
        return self.outputs

    def get_output(self, conv_id=-1):
        if conv_id == -1:
            conv_id = self.conv_id
        return self.outputs[conv_id]
    
    def reset(self):
        if self.inj_loc == 'w' and not self.pre_value == []:
            self.reset_weight()
        self.outputs = []
        self.pre_value = []
        self.post_value = []
        self.inj_coord = 0
        self.sites = []
        self.mode = -1
        self.bit = -1
        self.change_to = 1000.
        self.conv_ind = 0
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def reset_weight(self):
        self.conv.weight[self.inj_coord] = self.pre_value
    
    # inject into a weight offline - so not as part of the hook function
    def inject_weight(self, inj_coord):
        assert(self.inj_loc == 'w')
        # if the weight has been injected, reset the weights to the original value
        if not self.pre_value == []:
            self.reset_weight()
        
        # update inj_coord
        self.inj_coord = inj_coord
        with torch.no_grad():
            # get the clean value
            inject_val = self.conv.weight[self.inj_coord]
            # print("inject val: " + str(inject_val))
            self.pre_value = inject_val.detach().clone()
            
            # get the new injected value depending on mode
            if self.mode == 0:
                inject_val = bitflip.flip_bit(inject_val, self.bit)
            elif self.mode == 1:
                inject_val = bitflip.flip_bit(inject_val)
            elif self.mode == 2:
                inject_val = torch.as_tensor(self.change_to)
            else:
                assert(False)
            self.post_value = inject_val
            
            # replace value within the weights
            self.conv.weight[self.inj_coord] = inject_val
    
    def get_weight(self):
        return self.conv.weight.detach().clone()
    
    def run_hook(self, test_img, inj_coord, sites, mode="change_to", change_to=1000., bit=-1):
        self.reset()
        self.inj_coord = inj_coord
        self.set_sites(sites)
        self.set_mode(mode, change_to, bit)
        
        if self.inj_loc == 'w':
            self.inject_weight(self.inj_coord)
            
        with torch.no_grad():
            out = self.forward(test_img)

        return (out, self.pre_value.item(), self.post_value.item())