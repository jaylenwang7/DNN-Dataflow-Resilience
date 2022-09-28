import torchvision.models as models
import torch
from os.path import exists
import os
import numpy as np
import pickle
from info_model import *
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
    net = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    net.eval()
    return net

# compares two numpy arrays and return number of indices for which they differ
def compare_outputs(output1, output2):
    assert(output1.shape == output2.shape)
    matching = output1 == output2
    diff = np.where(matching.to("cpu") == False)
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

def num_nonzeros(output, dims=None):
    num_z = torch.count_nonzero(output, dim=dims)
    return num_z.to("cpu").tolist()

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