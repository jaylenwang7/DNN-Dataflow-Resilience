import torchvision.models as models
import torch
from os.path import exists
import os
import numpy as np
import pickle
from loop import *
from info_model import *
from parser import *
from pathlib import Path

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
    file_num = 0
    candidate_filename = filename
    while exists(filename + "." + extension):
        filename = candidate_filename + str(file_num)
        file_num += 1
        
    return filename + "." + extension

def pickle_object(obj, filename: str):
    with open(filename, 'ab') as f:
        pickle.dump(obj, f)
        
def get_pickle(filename:str):
    with open(filename, 'rb') as f:
        pickle.load(f)
        
def get_loops(get_net, dir, sizes, paddings, strides, to_parse='**/*.map.txt', d_type='i'):
    net = get_net()
    loops, divs, names = parse_files(dir, to_parse)
    layer_ids = print_layer_sizes(net, do_print=False)
    
    # TODO: you can reuse loop objects for layers of same size - just need to reset things
    out_loops = []
    for i in range(len(layer_ids)):
        layer_id = layer_ids[i] - 1
        new_loop = Loop(loops[layer_id], divs[layer_id][d_type], d_type=d_type, input_strides=strides[i], 
                        sizes=sizes[i], paddings=paddings[i])
        out_loops.append(new_loop)
        
    return out_loops, names

def delete_files(dir, filename):
    files = Path(dir).glob("**/" + filename)
    for file in files:
        if exists(file):
            os.remove(file)