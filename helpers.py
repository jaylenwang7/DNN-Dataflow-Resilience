import torchvision.models as models
import torch
from os.path import exists
import numpy as np

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