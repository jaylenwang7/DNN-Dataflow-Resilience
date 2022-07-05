import torchvision.models as models
import torch
from os.path import exists
import os
import numpy as np
import pickle
from loop import *
from info_model import *
from parser import parse_files
from pathlib import Path
import timm

def get_alexnet():
    net = models.alexnet(pretrained=True)
    net.eval()
    return net

def get_resnet18():
    net = models.resnet18(pretrained=True)
    net.eval()
    return net

def get_mobilenet_v3_small():
    net = models.mobilenet_v3_small(pretrained=True)
    net.eval()
    return net

def get_convnext_tiny():
    net = models.convnext_tiny(pretrained=True)
    net.eval()
    return net

def get_convnext_small():
    net = models.convnext_tiny(pretrained=True)
    net.eval()
    return net

def get_efficientnet_b0():
    net = models.efficientnet_b0(pretrained=True)
    net.eval()
    return net

def get_googlenet():
    net = models.googlenet(pretrained=True)
    net.eval()
    return net

def get_squeezenet():
    net = models.squeezenet(pretrained=True)
    net.eval()
    return net

def get_vit():
    net = timm.create_model("vit_base_patch16_224", pretrained=True)
    net.eval()
    return net

def get_deit_tiny():
    # net = timm.create_model("vit_small_patch16_224_dino", pretrained=True)
    net = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    net.eval()
    return net


# given a clean network (net), some set of imgs (given by img_inds)
# of the dataset - return the correct classification rates
def get_baseline(net, img_inds, dataset):
    net.eval()
    correct = 0
    total = 0
    classifications = []
    for ind in img_inds:
        img = torch.unsqueeze(dataset[ind]['image'], 0)
        res = net(img)
        _, max_inds = torch.topk(res, 5, 1)
        max_inds = torch.squeeze(max_inds)
        max_inds = max_inds.numpy()
        classifications.append(max_inds)
        if max_inds[0] == dataset[ind]['label']:
            correct += 1
        total += 1

    return correct, total, classifications

# compares two numpy arrays and return number of indices for which they differ
def compare_outputs(output1, output2):
    assert(output1.shape == output2.shape)
    matching = output1 == output2
    diff = np.where(matching == False)
    num_diffs = diff[0].size

    return num_diffs

# compares two numpy arrays and return number of indices for which they differ
def compare_outputs_range(output1, output2):
    assert(output1.shape == output2.shape)
    matching = output1 == output2
    diff = np.where(matching == False)
    num_diffs = diff[0].size
    ranges = []
    for d in diff:
        ranges.append((np.min(d), np.max(d)+1))

    return num_diffs, ranges

def num_nonzeros(output):
    num_z = torch.count_nonzero(output)
    return int(num_z.item())

def get_new_filename(filename, extension='csv'):
    file_num = -1
    candidate_filename = filename
    while exists(filename + "." + extension):
        file_num += 1
        filename = candidate_filename + "-" + str(file_num)
        
    return filename + "." + extension, file_num

def pickle_object(obj, filename: str):
    with open(filename, 'ab') as f:
        pickle.dump(obj, f)
        
def get_pickle(filename:str):
    with open(filename, 'rb') as f:
        pickle.load(f)
        
def get_loops(get_net, dir, sizes, paddings, strides, to_parse='**/*.map.txt', d_type='i', print_out=False, layers=[], ids_to_skip=[]):
    net = get_net()
    # parse the files in the given dir and get info
    loops, divs, names = parse_files(dir, to_parse)
    # get the memory names from the first layer
    sample_id = list(loops.keys())[0]
    out_names = [names[sample_id][div] for div in divs[sample_id][d_type]]
    # use print_layer_sizes to get the layer_ids for each layer
    layer_ids = print_layer_sizes(net, do_print=False, return_FC=True, return_inc=True)
    
    if not layers:
        layers = range(len(layer_ids))
    
    # TODO: you can reuse loop objects for layers of same size - just need to reset things
    out_loops = []
    # loop through each layer
    for i in range(len(layer_ids)):
        # get layer_id for the current layer and create loop object
        layer_id = layer_ids[i]
        # if not in desired layers, just append None
        if i not in layers or layer_id in ids_to_skip:
            out_loops.append(None)
            continue
        
        # create the loop using the layer_id
        div = divs[layer_id][d_type]
        if d_type == 'o':
            div = [0]
        new_loop = Loop(loops[layer_id], 
                        div, 
                        d_type=d_type, 
                        input_strides=strides[i], 
                        sizes=sizes[i], 
                        paddings=paddings[i])
        # add to list of loop objects
        out_loops.append(new_loop)
    
    # print out each loop if desired
    if print_out:
        for i in range(len(out_loops)):
            print("Layer " + str(i) + ":")
            print(out_loops[i])

    return out_loops, out_names

def delete_files(dir, filename):
    files = Path(dir).glob("**/" + filename)
    for file in files:
        if exists(file):
            os.remove(file)
            
def check_inj_coord(inj_coord, stride, w):
    return inj_coord % stride < w

def check_inj_ind(inj_ind, strides, ws):
    for i in range(len(inj_ind)):
        if not check_inj_coord(inj_ind[i], strides[i], ws[i]):
            return False
    return True

def get_str_num(in_string, after=""):
    in_string = str(in_string)
    if after:
        in_string = in_string.split(after)[1]
    num = ""
    prev_dig = False
    for s in in_string:
        if s.isdigit():
            num += s
            prev_dig = True
        else:
            if prev_dig:
                break
    if num.isdigit():
        return int(num)
    else:
        return None