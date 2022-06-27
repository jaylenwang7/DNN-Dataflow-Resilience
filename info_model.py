from torch import nn, Tensor
import torch
from prettytable import PrettyTable
from dataset import get_dataset

class LayerInfo():
    def __init__(self, layer, inp, outp):
        self.wshape = tuple(layer.weight.shape)
        self.oshape = tuple(outp.shape)
        self.ishape = tuple(inp[0].shape)
        
    def get_shapes(self):
        pass
    
    def get_vars(self):
        pass
    
    def get_padding(self):
        pass
    
    def get_stride(self):
        pass
    
class ConvInfo(LayerInfo):
    def __init__(self, layer, inp, outp):
        super().__init__(layer, inp, outp)
        self.padding = layer.padding
        self.stride = layer.stride
        
    def get_shapes(self):
        return [self.wshape, self.oshape, self.ishape, self.padding, self.stride]
    
    def get_padding(self):
        return self.padding
    
    def get_stride(self):
        return self.stride
    
    def get_vars(self):
        # need to return in order (m, c, s, r, q, p, h, w)
        shapes = self.get_shapes()
        return list(shapes[0]) + list(shapes[1][2:]) + list(shapes[2][2:])
        
class FCInfo(LayerInfo):
    def __init__(self, layer, inp, outp):
        super().__init__(layer, inp, outp)
        
    def get_shapes(self):
        return [self.wshape, self.oshape, self.ishape]
    
    def get_vars(self):
        # need to return in order (m, c, s, r, q, p, h, w) - only m,c are not 1
        shapes = self.get_shapes()
        if len(shapes[1]) == 2:
            return list(shapes[0]) + [1, 1, 1, 1, 1, 1]
        elif len(shapes[1]) == 3:
            extra_dim = [shapes[1][1]]
            return list(shapes[0]) + [1, 1, 1] + extra_dim + [1] + extra_dim
        else:
            assert(False and "shape of FC layer not recognized")
    
    # use hard-coded values for FC
    def get_padding(self):
        return (0, 0)
    def get_stride(self):
        return (1, 1)
    
    def get_input_size(self):
        shapes = self.get_shapes()
        return len(shapes[1])

# model wrapper to print layer sizes
class InfoModel(nn.Module):
    def __init__(self, model: nn.Module, process_FC=False):
        super().__init__()
        self.model = model
        self.layer_info = []
        self.conv_info = []
        self.FC_info = []
        self.process_FC = process_FC
        
        # Register a hook for each layer
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self.hook_conv_info)
            elif isinstance(layer, nn.Linear) and self.process_FC:
                layer.register_forward_hook(self.hook_FC_info)
    
    # hook function - appends a conv layer's data to lists
    def hook_conv_info(self, layer, input_, output):
        this_conv_info = ConvInfo(layer, input_, output)
        self.conv_info.append(this_conv_info)
        self.layer_info.append(this_conv_info)
    
    # hook function - appends a FC layer's data to lists
    def hook_FC_info(self, layer, input_, output):
        this_FC_info = FCInfo(layer, input_, output)
        self.FC_info.append(this_FC_info)
        self.layer_info.append(this_FC_info)

    # forward function - takes in a normal image form the dataset
    def forward(self, x: Tensor) -> Tensor:
        x = torch.unsqueeze(x, 0)
        return self.model(x)
    
    def get_vars(self, info):
        vars = []
        for i in info:
            vars.append(i.get_vars())
        return vars
    
    # return the conv_info object as well as accumulates the var sizes in order:
    # m, c, s, r, q, p, h, w
    def get_info(self):
        return self.layer_info, self.get_vars(self.layer_info)
    
    def get_conv_info(self):
        return self.conv_info, self.get_vars(self.conv_info)
    
    def get_FC_info(self):
        return self.FC_info, self.get_vars(self.FC_info)
    
def get_layer_info(get_net: callable, img, with_FC=True):
    verb_net = get_net()
    vnet = InfoModel(verb_net, process_FC=with_FC)
    vnet(img)
    layers_info, var_sizes = vnet.get_info()
    num_layers = len(layers_info)
    # vnet.conv_info has the form [weight, output, input, pad, stride] for each layer
    # weight (m, c, s, r), output (1, m, q, p), input (1, c, h, w)
    # var_sizes will have the form [m, c, s, r, q, p, h, w]
    paddings = []
    strides = []
    FC_types = []
    for i in range(num_layers):
        layer_info = layers_info[i]
        paddings.append(layer_info.get_padding())
        strides.append(layer_info.get_stride())
        if isinstance(layer_info, FCInfo):
            FC_types.append(layer_info.get_input_size())
        else:
            FC_types.append(-1)
        
    return num_layers, var_sizes, paddings, strides, FC_types

def print_layer_sizes(net, net_name='', do_print=True, with_FC=True, return_FC=True, return_inc=False):
    # instantiate a table
    table = PrettyTable()
    if net_name:
        table.title = "Layer Information for " + net_name
    table.field_names = ["layer #", "type", "weight", "output", "input", "padding", "stride", "layer id"]
    
    # get dataset to use; TODO: take in a function instead
    dataset = get_dataset()
    
    # get verbose model info and lyaer info
    vnet = InfoModel(net, process_FC=with_FC)
    vnet(dataset[0]['image'])
    layer_info, _ = vnet.get_info()
    
    # loop through layers and collect data while printing
    # layer_num = number of the layer (not necessarily unique)
    # layer_id = number of last unique layer seen (counts unique layers)
    # curr_id = layer_id of the currently processed layer
    layer_num = layer_id = curr_id = 0
    layer_id = -1
    layer_inc = -1
    inc_dict = {}
    layer_dict = {}
    layer_ids = []
    
    # loop through all layer_infos
    for linfo in layer_info:
        layer_inc += 1
        # get the shape of the layer
        vinfo = tuple(linfo.get_shapes())
        # if not seen shape before (new layer shape)
        if vinfo not in layer_dict:
            # increment layer_id since new
            layer_id += 1
            # add the shape to the dict
            layer_dict[vinfo] = layer_id
            inc_dict[layer_id] = layer_inc
            # change curr_id
            curr_id = layer_id
        # else seen before
        else:
            # get the ID of the layer
            curr_id = layer_dict[vinfo]
        row = []
        row.append(str(layer_num))
        
        if isinstance(linfo, ConvInfo):
            row.append("CONV")
            layer_ids.append(curr_id)
        elif isinstance(linfo, FCInfo):
            row.append("FC")
            if return_FC:
                layer_ids.append(curr_id)
        else:
            assert(False)
            
        for s in vinfo:
            row.append(str(s))
            
        if isinstance(linfo, FCInfo):
            row += ["N/A", "N/A"]
            
        row.append(curr_id)
        table.add_row(row)
        layer_num += 1
        
    if do_print:
        print(table)
    
    if return_inc:
        return [inc_dict[layer_id] for layer_id in layer_ids]
    else:
        return layer_ids