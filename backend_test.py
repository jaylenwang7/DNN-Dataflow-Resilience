from loop import *
from datetime import datetime
from pathlib import Path

'''
This file is for testing the backend - so just looking at sites that are produced and checking correctness
'''

def pretty_print(sites, window, timed=False, filename='test.txt', ):
    num_s = 0
    with open(filename, 'a') as f:
        if timed:
            all_sites = sites[0][0][0]
        else:
            all_sites = sites[0][0]
            
        all_sites_ = set(all_sites)
        f.write("og len: " + str(len(all_sites)) + ", set len: " + str(len(all_sites_)) + "\n")
        f.write("og window: " + str(window) + "\n")
        for s in sites:
            f.write(str(num_s) + " ----------" + "\n")
            num_s += 1
            for p in s:
                if timed:
                    for g in p:
                        f.write("**  " + str(g) + "\n")
                else:
                    f.write("** " + str(p) + "\n")
        f.write("============================================================\n")
        f.write("============================================================\n")

def run_test(vars, dividers, inj_site, filename, spatial=True, d_type='i', stride=[1, 1], 
             range_check=False, sizes=[], serial=False, prune=False):
    dir = "tests/"
    p = Path(dir)
    p.mkdir(parents=True, exist_ok=True)
    filename = "tests/" + filename + "-" + d_type
    if spatial:
        filename = filename + "-spatial"
    if serial:
        filename += "-serial"
    filename = filename + ".txt"

    with open(filename, 'a') as f:
        f.write(str(datetime.now()) + "\n")
        
    if sizes:
        var_sizes = list(sizes[0]) + list(sizes[1][2:]) + list(sizes[2][2:])
    else:
        var_sizes = []

    injection = Loop(vars, dividers, d_type=d_type, input_strides=stride, out_file=filename, serial=serial, sizes=var_sizes)
    _, sites = injection.inject_full(inj_site)
    og_window = injection.original_window
    if spatial and not serial:
        sites = injection.insert_spatial()
        if prune:
            sites = injection.prune_sites(sites)
        pretty_print(sites, og_window, timed=True, filename=filename)
        if range_check:
            # convert given sizes - given by [weight, output, input, padding, stride]
            check_sites(sites[0][0][0], inj_site, var_sizes, strides=stride, padding=sizes[3], d_type=d_type)
    else:
        # print(sites)
        # sites = injection.all_out_sites
        pretty_print(sites, og_window, filename=filename)

def test_nvdla(spatial=True):
    nvdla_vars_2 = [('m', 12), ('m', 16, True), ('c', 64), ('q', 8), ('p', 8), ('c', 1), 
              ('s', 5), ('r', 5), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 2, 10]
    inj_site = (5, 6, 6)
    run_test(nvdla_vars_2, mem_dividers, inj_site, 'test_nvdla', spatial)
    
def test_nvdla_no_spatial(spatial=True):
    nvdla_vars_2 = [('m', 12), ('m', 16), ('c', 64), ('q', 8), ('p', 8), ('c', 1), 
              ('s', 5), ('r', 5), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 2, 10]
    inj_site = (5, 6, 6)
    run_test(nvdla_vars_2, mem_dividers, inj_site, 'test_nvdla_no_spatial', spatial)

def test_nvdla_small(spatial=True):
    nvdla_vars_2 = [('m', 5), ('m', 4, True), ('p', 8), ('r', 5), ('p', 4), ('r', 1)]
    mem_dividers = [0, 1, 5]
    inj_site = (5, 0, 6)
    run_test(nvdla_vars_2, mem_dividers, inj_site, 'test_nvdla_small', spatial)

def test_eyeriss(spatial=True):
    eyeriss_vars = [('q', 55), ('c', 3), ('p', 55), ('m', 12, True), ('s', 11, True), ('r', 11), ('m', 8), ('m', 1)]
    mem_dividers = [0, 2, 5]
    inj_site = (0, 20, 20)
    # number of total output sites should be:
    #   m*r*s = 1452
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss', spatial)

def test_eyeriss_s(spatial=True):
    eyeriss_vars = [('q', 55), ('p', 55), ('s', 11, True), ('r', 11)]
    mem_dividers = [0, 1, 3]
    inj_site = (0, 20, 20)
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_s', spatial)

def test_eyeriss_m(spatial=True):
    eyeriss_vars = [('p', 20), ('m', 3, True), ('r', 11), ('m', 4), ('m', 1)]
    mem_dividers = [0, 1, 2]
    inj_site = (0, 0, 20)
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_m', spatial)

def test_nvdla_small_weight(spatial=True):
    nvdla_vars_2 = [('m', 5), ('m', 4, True), ('p', 8), ('r', 5), ('p', 4), ('r', 1)]
    mem_dividers = [0, 2, 5]
    inj_site = (0, 0, 3)
    run_test(nvdla_vars_2, mem_dividers, inj_site, 'test_nvdla_small_weight', spatial, 'w')

def test_eyeriss_s_weight(spatial=True):
    eyeriss_vars_2 = [('q', 55), ('p', 55), ('s', 11, True), ('r', 11)]
    mem_dividers = [0, 3]
    inj_site = (0, 1, 0)
    run_test(eyeriss_vars_2, mem_dividers, inj_site, 'test_eyeriss_s_weight', spatial, 'w')

def test_eyeriss_weight(spatial=True):
    eyeriss_vars_2 = [('q', 55), ('c', 3), ('p', 55), ('m', 12, True), ('s', 11, True), ('r', 11), ('m', 8), ('m', 1)]
    mem_dividers = [0, 5]
    inj_site = (0, 5, 5)
    run_test(eyeriss_vars_2, mem_dividers, inj_site, 'test_eyeriss_weight', spatial, 'w')

def test_weight_spatial(spatial=True):
    weight_vars = [('m', 1), ('r', 2, True), ('p', 3), ('r', 1)]
    mem_dividers = [1, 2, 3]
    inj_site = (0, 0, 1)
    run_test(weight_vars, mem_dividers, inj_site, 'test_weight_spatial', spatial, 'w')

def test_weight(spatial=True):
    weight_vars = [('m', 1), ('r', 2), ('p', 3), ('r', 1)]
    mem_dividers = [1, 2, 3]
    inj_site = (0, 0, 0)
    run_test(weight_vars, mem_dividers, inj_site, 'test_weight', spatial, 'w')

# tests small network with spatial in r with weight injection
def test_weight_spatial_v2(spatial=True):
    weight_vars = [('m', 4), ('r', 2, True), ('p', 3), ('r', 1)]
    mem_dividers = [1, 2, 3]
    inj_site = (3, 0, 1)
    run_test(weight_vars, mem_dividers, inj_site, 'test_weight_spatial_v2', spatial, 'w')

# tests small network with spatial in m with weight injection
def test_weight_spatial_v3(spatial=True):
    weight_vars = [('m', 4, True), ('r', 2), ('p', 3), ('r', 1)]
    mem_dividers = [1, 2, 3]
    inj_site = (2, 0, 1)
    run_test(weight_vars, mem_dividers, inj_site, 'test_weight_spatial_v3', spatial, 'w')

# tests small network with spatial in m with weight injection
def test_weight_spatial_v4(spatial=True):
    weight_vars = [('m', 4, True), ('p', 3), ('r', 2), ('r', 1)]
    mem_dividers = [1, 2, 3]
    inj_site = (2, 0, 1)
    run_test(weight_vars, mem_dividers, inj_site, 'test_weight_spatial_v4', spatial, 'w')

# tests small network with spatial in r with weight injection
def test_weight_spatial_v5(spatial=True):
    weight_vars = [('m', 4), ('r', 4, True), ('p', 8), ('r', 1)]
    mem_dividers = [1, 2, 3]
    inj_site = (3, 0, 3)
    run_test(weight_vars, mem_dividers, inj_site, 'test_weight_spatial_v5', spatial, 'w')


'''
Test stride
'''
def test_stride(spatial=True):
    stride_vars = [('m', 4, True), ('r', 2), ('p', 3), ('r', 1)]
    mem_dividers = [1, 2, 3]
    strides = [1, 2]
    inj_site = (2, 0, 1)
    run_test(stride_vars, mem_dividers, inj_site, 'test_stride', spatial, 'w', strides)

def test_stride_v2(spatial=True):
    stride_vars = [('r', 3), ('p', 8), ('r', 1)]
    mem_dividers = [0, 1]
    strides = [1, 4]
    inj_site = (0, 0, 4)
    run_test(stride_vars, mem_dividers, inj_site, 'test_stride_v2', spatial, 'i', strides)

def test_stride_v3(spatial=True):
    eyeriss_vars = [('p', 4), ('r', 4), ('s', 1), ('r', 1)]
    mem_dividers = [0, 1, 2]
    stride = [1, 2]
    inj_site = (0, 0, 7)
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_stride_v3', spatial, d_type='i', stride=stride)

def test_stride_v4(spatial=True):
    stride_vars = [('s', 3), ('q', 8), ('s', 1)]
    mem_dividers = [0, 1]
    strides = [4, 1]
    inj_site = (0, 4, 0)
    run_test(stride_vars, mem_dividers, inj_site, 'test_stride_v4', spatial, 'i', strides)
    
def test_stride_v5(spatial=True):
    stride_vars = [('p', 55), ('r', 11), ('r', 1)]
    mem_dividers = [0]

    inj_site = (0, 0, 20)
    strides = [4, 4]
    
    run_test(stride_vars, mem_dividers, inj_site, 'test_stride_v5', spatial, 'i', strides, serial=False)
    
def test_stride_v6(spatial=True):
    stride_vars = [('p', 5), ('p', 11), ('r', 11), ('r', 1)]
    mem_dividers = [0]

    inj_site = (0, 0, 20)
    strides = [1, 1]
    
    run_test(stride_vars, mem_dividers, inj_site, 'test_stride_v6', spatial, 'i', strides, serial=False)
    

'''
Tests small eyeriss examples or stride
'''
def test_eyeriss_small(spatial=True):
    eyeriss_vars = [('q', 4), ('p', 4), ('s', 4, True), ('r', 4), ('s', 1), ('r', 1)]
    mem_dividers = [0, 2, 3]
    stride = [2, 2]
    inj_site = (0, 3, 7)
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_small', spatial, d_type='i', stride=stride)

def test_eyeriss_small_v2(spatial=True):
    eyeriss_vars = [('q', 4), ('p', 4), ('s', 12, True), ('r', 4), ('s', 1), ('r', 1)]
    mem_dividers = [0, 2, 3]
    stride = [2, 2]
    inj_site = (0, 8, 7)
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_small_v2', spatial, d_type='i', stride=stride)

def test_eyeriss_stride(spatial=True):
    eyeriss_vars = [('q', 55), ('c', 3), ('p', 55), ('m', 12, True), ('s', 11, True), ('r', 11), ('m', 8), ('r', 1)]
    mem_dividers = [0, 2, 5]
    strides = [4, 4]
    inj_site = [2, 31, 24]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_stride', spatial, 'i', strides)


'''
Following tests are for layers of alexnet with eyeriss
for alexnet:
   [weight,               output,             input,              padding,    stride]
0: [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
1: [(192, 64, 5, 5),     (1, 192, 27, 27),   (1, 64, 27, 27),    (2, 2),     (1, 1)]
2: [(384, 192, 3, 3),    (1, 384, 13, 13),   (1, 192, 13, 13),   (1, 1),     (1, 1)]
3: [(256, 384, 3, 3),    (1, 256, 13, 13),   (1, 384, 13, 13),   (1, 1),     (1, 1)]
4: [(256, 256, 3, 3),    (1, 256, 13, 13),   (1, 256, 13, 13),   (1, 1),     (1, 1)]
'''
def test_eyeriss_alexnet0(spatial=True):
    eyeriss_vars = [('q', 5), ('c', 3), ('m', 4), ('p', 55), ('q', 11, True), ('s', 11, True),
                     ('r', 11), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers = [0, 2, 6]
    inj_site = (1, 143, 102)
    strides = [4, 4]
    sizes = [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_alexnet0', spatial, 'i', strides, range_check=True, sizes=sizes, serial=False)
    
def test_eyeriss_alexnet0_weight(spatial=True):
    eyeriss_vars = [('q', 5), ('c', 3), ('m', 4), ('p', 55), ('q', 11, True), ('s', 11, True),
                     ('r', 11), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers = [0, 6]
    inj_site = (54, 0, 6, 0)
    strides = [4, 4]
    sizes = [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_alexnet0', spatial, 'w', strides, range_check=True, sizes=sizes, serial=False)

def test_eyeriss_alexnet1(spatial=True):
    eyeriss_vars = [('q', 9), ('c', 16), ('m', 3), ('p', 27), ('m', 4, True), ('q', 3, True), ('c', 2, True),
                     ('s', 5, True), ('r', 5), ('c', 2), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers = [0, 2, 8]
    inj_site = (35, 18, 22)
    strides = [1, 1]
    sizes = [(192, 64, 5, 5),     (1, 192, 27, 27),   (1, 64, 27, 27),    (2, 2),     (1, 1)]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_alexnet1', spatial, 'i', strides, range_check=True, sizes=sizes)
    
def test_eyeriss_alexnet1_weight(spatial=True):
    eyeriss_vars = [('q', 9), ('c', 16), ('m', 3), ('p', 27), ('m', 4, True), ('q', 3, True), ('c', 2, True),
                     ('s', 5, True), ('r', 5), ('c', 2), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers = [0, 6]
    inj_site = (171, 53, 0, 2)
    strides = [1, 1]
    sizes = [(192, 64, 5, 5),     (1, 192, 27, 27),   (1, 64, 27, 27),    (2, 2),     (1, 1)]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_alexnet1', spatial, 'w', strides, range_check=True, sizes=sizes)

def test_eyeriss_alexnet2(spatial=True):
    eyeriss_vars = [('m', 3), ('c', 12), ('m', 8), ('p', 13), ('q', 13, True), ('c', 4, True),
                     ('s', 3, True), ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers = [0, 2, 7]
    inj_site = (111, 10, 7)
    strides = [1, 1]
    sizes = [(384, 192, 3, 3),    (1, 384, 13, 13),   (1, 192, 13, 13),   (1, 1),     (1, 1)]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_alexnet2', spatial, 'i', strides, range_check=True, sizes=sizes)
    
def test_eyeriss_alexnet2_weight(spatial=True):
    eyeriss_vars = [('m', 3), ('c', 12), ('m', 8), ('p', 13), ('q', 13, True), ('c', 4, True),
                     ('s', 3, True), ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers = [0, 6]
    inj_site = (305, 165, 2, 1)
    strides = [1, 1]
    sizes = [(384, 192, 3, 3),    (1, 384, 13, 13),   (1, 192, 13, 13),   (1, 1),     (1, 1)]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_alexnet2', spatial, 'w', strides, range_check=True, sizes=sizes)

def test_eyeriss_alexnet3(spatial=True):
    eyeriss_vars = [('c', 24), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers = [0, 2, 7]
    inj_site = (86, 10, 11)
    strides = [1, 1]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_alexnet3', spatial, 'i', strides)

def test_eyeriss_alexnet4(spatial=True):
    eyeriss_vars = [('c', 24), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers = [0, 2, 6]
    inj_site = (222, 5, 6)
    strides = [1, 1]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_alexnet4', spatial, 'i', strides)
    
'''
Following tests are for misc for eyeriss
'''
    
def test_eyeriss_alexnet0_no_spatial(spatial=True):
    eyeriss_vars = [('q', 5), ('c', 3), ('m', 4), ('p', 55), ('q', 11), ('s', 11), ('r', 11), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers = [0, 2, 6]

    inj_site = (1, 143, 102)
    strides = [4, 4]
    sizes = [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_alexnet0_no_spatial', spatial, 'i', strides, range_check=True, sizes=sizes, serial=False)
    
def test_eyeriss_alexnet0_p(spatial=True):
    # eyeriss_vars = [('q', 5), ('c', 3), ('m', 4), ('p', 55), ('q', 11), ('s', 11), ('r', 11), ('m', 16), ('s', 1), ('r', 1)]
    eyeriss_vars = [('c', 3), ('m', 4), ('p', 55), ('r', 11), ('m', 16), ('r', 1)]
    mem_dividers = [0]

    inj_site = (1, 0, 102)
    strides = [4, 4]
    sizes = [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_alexnet0_no_spatial', spatial, 'i', strides, range_check=True, sizes=sizes, serial=False)
    
def test_eyeriss_alexnet0_q(spatial=True):
    eyeriss_vars = [('q', 5), ('c', 3), ('m', 4), ('q', 11), ('s', 11), ('m', 16), ('s', 1)]
    mem_dividers = [0]

    inj_site = (1, 20, 0)
    strides = [4, 4]
    
    sizes = [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_alexnet0_no_spatial', spatial, 'i', strides, range_check=True, sizes=sizes, serial=False)
    
def test_eyeriss_alexnet0_qp(spatial=True):
    eyeriss_vars = [('p', 55), ('r', 11), ('r', 1)]
    mem_dividers = [0]

    inj_site = (0, 0, 20)
    strides = [4, 4]
    sizes = [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_alexnet0_no_spatial', spatial, 'i', strides, range_check=True, sizes=sizes, serial=False)
    
def test_eyeriss_alexnet4_no_spatial(spatial=True):
    eyeriss_vars = [('c', 24), ('m', 16), ('p', 13), ('q', 13), ('c', 4), ('s', 3),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers = [0, 2, 6]
    # mem_dividers_1w = [0, 6]
    inj_site = (222, 5, 6)
    strides = [1, 1]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_alexnet4', spatial, 'i', strides)

def test_eyeriss_weights_none(spatial=True):
    eyeriss_vars_1 = [('q', 5), ('c', 3), ('m', 4), ('p', 55), ('q', 11, True), ('s', 11, True),
                     ('r', 11), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_1 = [0, 6]
    inj_site = (21, 1, 7, 4)
    strides = [4, 4]
    run_test(eyeriss_vars_1, mem_dividers_1, inj_site, "test_eyeriss_weights_none", spatial, 'w', strides)

def test_q_spatial_weight(spatial=True):
    eyeriss_vars_1 = [('q', 9), ('m', 3), ('p', 27), ('m', 4, True), ('q', 3, True),
                     ('s', 5, True), ('r', 5), ('s', 1), ('r', 1)]
    mem_dividers_1 = [0, 6]
    inj_site = (21, 0, 1, 2)
    strides = [1, 1]
    run_test(eyeriss_vars_1, mem_dividers_1, inj_site, "test_q_spatial", spatial, 'w', strides)

def test_q_spatial_input(spatial=True):
    eyeriss_vars_1 = [('q', 9), ('m', 3), ('p', 27), ('m', 4, True), ('q', 3, True),
                     ('s', 5, True), ('r', 5), ('s', 1), ('r', 1)]
    mem_dividers_1 = [0, 1, 6]
    inj_site = (0, 17, 20)
    strides = [1, 1]
    # size of DRAM number sites should be:
    # 25*12=300
    run_test(eyeriss_vars_1, mem_dividers_1, inj_site, "test_q_spatial", spatial, 'i', strides)
    
def test_input_q(spatial=True):
    vars = [('q', 2), ('p', 4), ('q', 2, True), ('s', 3), ('r', 3), ('s', 1), ('r', 1)]
    inj_site = (0, 4, 4)
    mem_dividers_1 = [0]
    strides = [1, 1]
    run_test(vars, mem_dividers_1, inj_site, "test_input_q", spatial, 'i', strides)
    
def test_spatial_q(spatial=True):
    vars = [('q', 4, True), ('s', 3), ('s', 1)]
    inj_site = (0, 4, 0)
    mem_dividers_1 = [0]
    strides = [1, 1]
    run_test(vars, mem_dividers_1, inj_site, "test_spatial_q", spatial, 'i', strides)
    
def test_spatial_q_weight(spatial=True):
    vars = [('q', 4, True), ('s', 3), ('s', 1)]
    inj_site = (0, 0, 1, 0)
    mem_dividers_1 = [0, 1, 2]
    strides = [1, 1]
    run_test(vars, mem_dividers_1, inj_site, "test_spatial_q", spatial, 'w', strides)
    
def test_spatial_s(spatial=True):
    vars = [('q', 4), ('s', 3, True), ('s', 1)]
    inj_site = (0, 4, 0)
    mem_dividers_1 = [0]
    strides = [1, 1]
    run_test(vars, mem_dividers_1, inj_site, "test_spatial_s", spatial, 'i', strides)
    
def test_spatial_s_weight(spatial=True):
    vars = [('q', 4), ('s', 3, True), ('s', 1)]
    inj_site = (0, 0, 1, 0)
    mem_dividers_1 = [0]
    strides = [1, 1]
    run_test(vars, mem_dividers_1, inj_site, "test_spatial_s", spatial, 'w', strides)
    
'''
Following tests are for layers of alexnet with NVDLA
for alexnet:
   [weight,               output,             input,              padding,    stride]
0: [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
1: [(192, 64, 5, 5),     (1, 192, 27, 27),   (1, 64, 27, 27),    (2, 2),     (1, 1)]
2: [(384, 192, 3, 3),    (1, 384, 13, 13),   (1, 192, 13, 13),   (1, 1),     (1, 1)]
3: [(256, 384, 3, 3),    (1, 256, 13, 13),   (1, 384, 13, 13),   (1, 1),     (1, 1)]
4: [(256, 256, 3, 3),    (1, 256, 13, 13),   (1, 256, 13, 13),   (1, 1),     (1, 1)]
'''
def test_nvdla_alexnet0(spatial=True):
    nvdla_vars = [('m', 4), ('m', 16, True), ('c', 3), ('q', 11), ('p', 11), ('c', 1), 
                    ('s', 11), ('r', 11), ('q', 5), ('p', 5), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    inj_site = (1, 143, 102)
    strides = [4, 4]
    sizes = [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
    run_test(nvdla_vars, mem_dividers, inj_site, 'test_nvdla_alexnet0', spatial, 'i', strides, range_check=True, sizes=sizes, serial=False)
    
def test_nvdla_alexnet0_weight(spatial=True):
    nvdla_vars = [('m', 4), ('m', 16, True), ('c', 3), ('q', 11), ('p', 11), ('c', 1), 
                    ('s', 11), ('r', 11), ('q', 5), ('p', 5), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    inj_site = (1, 143, 102)
    strides = [4, 4]
    sizes = [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
    run_test(nvdla_vars, mem_dividers, inj_site, 'test_nvdla_alexnet0', spatial, 'w', strides, range_check=True, sizes=sizes, serial=False)
    
def test_nvdla_alexnet1(spatial=True):
    nvdla_vars = [('m', 12), ('m', 16, True), ('c', 64), ('q', 8), ('p', 8), ('c', 1), 
                  ('s', 5), ('r', 5), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    inj_site = (35, 18, 22)
    strides = [1, 1]
    sizes = [(192, 64, 5, 5),     (1, 192, 27, 27),   (1, 64, 27, 27),    (2, 2),     (1, 1)]
    run_test(nvdla_vars, mem_dividers, inj_site, 'test_nvdla_alexnet1', spatial, 'i', strides, range_check=True, sizes=sizes, serial=False)
    
def test_nvdla_alexnet1_weight(spatial=True):
    nvdla_vars = [('m', 12), ('m', 16, True), ('c', 64), ('q', 8), ('p', 8), ('c', 1), 
                  ('s', 5), ('r', 5), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    inj_site = (13, 35, 1, 7)
    strides = [1, 1]
    sizes = [(192, 64, 5, 5),     (1, 192, 27, 27),   (1, 64, 27, 27),    (2, 2),     (1, 1)]
    run_test(nvdla_vars, mem_dividers, inj_site, 'test_nvdla_alexnet1', spatial, 'w', strides, range_check=True, sizes=sizes, serial=False, prune=True)
    
def test_spatial_m0(spatial=True):
    vars = [('m', 3), ('m', 6, True), ('m', 1)]
    dividers = [0, 1, 2]
    inj_site = (0, 0, 0)
    strides = [1, 1]
    run_test(vars, dividers, inj_site, 'test_spatial_m0', spatial, 'i', strides)
    
def test_spatial_m1(spatial=True):
    vars = [('m', 3), ('m', 2, True), ('m', 3)]
    dividers = [0, 1, 2]
    inj_site = (0, 0, 0)
    strides = [1, 1]
    run_test(vars, dividers, inj_site, 'test_spatial_m1', spatial, 'i', strides)
    
def test_spatial_m2(spatial=True):
    vars = [('m', 2), ('m', 2), ('m', 2, True), ('m', 3)]
    dividers = [0, 1, 2, 3]
    inj_site = (0, 0, 0)
    strides = [1, 1]
    run_test(vars, dividers, inj_site, 'test_spatial_m2', spatial, 'i', strides)
    
def test_eyeriss_resnet18_0(spatial=True):
    eyeriss_vars = [('q', 8), ('c', 4), ('m', 2), ('p', 56), ('m', 2, True), ('q', 7, True), ('q', 1), ('c', 4, True), ('s', 3, True), ('q', 1), ('r', 3), ('c', 4), ('m', 16)]
    mem_dividers = [0, 2, 8]
    inj_site = (2, 49, 38)
    strides = [1, 1]
    sizes = [(64, 64, 3, 3), (1, 64, 56, 56), (1, 64, 56, 56), (1, 1), (1, 1)]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_resnet18_0', spatial, 'i', strides, range_check=True, sizes=sizes, serial=False)
    
def test_eyeriss_resnet18_6(spatial=True):
    eyeriss_vars = [('c', 16), ('m', 16), ('p', 14), ('q', 14, True), ('q', 1), ('c', 4, True), ('s', 3, True), ('q', 1), ('r', 3), ('c', 4), ('m', 16)]
    mem_dividers = [0, 1, 7]
    inj_site = (13, 16, 20)
    strides = [1, 1]
    sizes = [(128, 128, 3, 3), (1, 128, 28, 28), (1, 128, 28, 28), (1, 1), (1, 1)]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_resnet18_6', spatial, 'i', strides, range_check=True, sizes=sizes, serial=False)
    
def test_eyeriss_resnet18_7(spatial=True):
    eyeriss_vars = [('q', 2), ('c', 4), ('m', 4), ('p', 14), ('m', 2, True), ('q', 7, True), ('q', 1), ('m', 2, True), ('c', 4, True), ('q', 1), ('c', 8), ('m', 16)]
    mem_dividers = [0, 2, 9]
    inj_site = (32, 10, 30)
    strides = [2, 2]
    sizes = [(128, 64, 1, 1), (1, 128, 28, 28), (1, 64, 56, 56), (0, 0), (2, 2)]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_resnet18_7', spatial, 'i', strides, range_check=True, sizes=sizes, serial=False)
    
def test_eyeriss_resnet18_13(spatial=True):
    eyeriss_vars = [('q', 2), ('c', 4), ('m', 4), ('p', 14), ('m', 2, True), ('q', 7, True), ('q', 1), ('m', 2, True), ('c', 4, True), ('q', 1), ('c', 8), ('m', 16)]
    mem_dividers = [0, 2, 9]
    inj_site = (32, 10, 30)
    strides = [1, 1]
    sizes = [(256, 256, 3, 3), (1, 256, 14, 14), (1, 256, 14, 14), (1, 1), (1, 1) ]
    run_test(eyeriss_vars, mem_dividers, inj_site, 'test_eyeriss_resnet18_7', spatial, 'i', strides, range_check=True, sizes=sizes, serial=False)
    
def test_weight1_stride2(spatial=True):
    vars = [('q', 2), ('q', 3, True), ('s', 1)]
    mem_dividers = [0]
    inj_site = (0, 4, 0)
    strides = [2, 2]
    run_test(vars, mem_dividers, inj_site, 'test_weight1_stride2', spatial, 'i', strides)

if __name__=="__main__":

    # test_eyeriss_alexnet0()
    # test_eyeriss_alexnet1()
    # test_eyeriss_alexnet2()
    # test_eyeriss_alexnet3()
    # test_eyeriss_alexnet4()
    # test_eyeriss_alexnet0_weight()
    # test_eyeriss_alexnet1_weight()
    # test_eyeriss_alexnet2_weight()
    
    # test_nvdla_alexnet0()
    # test_nvdla_alexnet1()
    # test_nvdla_alexnet0_weight()
    # test_nvdla_alexnet1_weight()
    
    # test_eyeriss_resnet18_0()
    # test_eyeriss_resnet18_6()
    # test_eyeriss_resnet18_7()
    
    test_weight1_stride2()
    
    # test_spatial_q()
    pass
