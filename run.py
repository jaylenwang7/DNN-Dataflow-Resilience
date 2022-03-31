from parser import *
from dataset import get_dataset
from helpers import *
from info_model import *
import loop as Loop
from model_injection import model_injection
from plotter import plotter

# if you want standard set of img indices to use across multiple experiments
# set of 100 images
sample_img_inds =           [46682, 7174, 25266, 20546, 21432, 11278, 34091, 43676, 36650, 35665, 
                            1230, 5002, 43288, 40393, 7420, 32524, 34305, 18479, 7068, 10317, 
                            40909, 408, 36743, 13065, 29275, 19160, 39110, 48822, 32220, 43096, 
                            1219, 3221, 27341, 25203, 8125, 42698, 14530, 3210, 44292, 48187, 
                            15912, 17289, 46640, 12198, 11693, 9411, 27420, 41892, 36781, 40879, 
                            16616, 33695, 19354, 14178, 2492, 2783, 45270, 25813, 17976, 17925, 
                            1394, 25675, 42908, 22083, 15782, 46222, 45715, 4099, 6151, 46450, 
                            33650, 18904, 11190, 36895, 49823, 16408, 40557, 9920, 30480, 9598, 
                            17606, 14870, 38048, 26602, 8300, 2585, 33886, 20564, 1501, 7154, 
                            36094, 25587, 5677, 46361, 2653, 12684, 837, 31840, 24381, 38293]

sample_inj_inds_input =    [[(2, 184, 132), (1, 34, 10), (0, 117, 55), (0, 67, 196), (1, 86, 124), (2, 35, 40), (1, 178, 192), (1, 110, 140)], 
                            [(28, 12, 15), (34, 16, 13), (8, 16, 18), (41, 23, 20), (45, 21, 16), (52, 12, 16), (25, 6, 19), (38, 10, 17)], 
                            [(150, 0, 4), (144, 2, 11), (12, 2, 11), (154, 6, 1), (19, 7, 4), (135, 4, 10), (64, 8, 3), (66, 11, 5)], 
                            [(86, 10, 11), (190, 7, 8), (222, 3, 1), (291, 3, 2), (58, 6, 2), (232, 3, 3), (121, 5, 3), (236, 3, 11)], 
                            [(126, 1, 1), (23, 0, 8), (142, 0, 11), (49, 1, 7), (15, 11, 0), (159, 2, 8), (83, 0, 2), (154, 1, 1)]]

sample_inj_inds_weight =   [[(54, 0, 6, 0), (20, 2, 5, 5), (4, 1, 0, 1), (55, 1, 0, 5), (12, 2, 7, 3), (53, 2, 5, 8), (23, 2, 9, 6), (8, 2, 7, 3)], 
                            [(13, 35, 1, 0), (41, 14, 4, 4), (148, 61, 0, 0), (2, 45, 3, 1), (148, 5, 1, 4), (116, 21, 3, 3), (171, 53, 0, 2), (171, 32, 4, 2)],
                            [(305, 165, 2, 1), (362, 60, 0, 2), (119, 82, 1, 2), (303, 50, 1, 0), (272, 164, 1, 1), (285, 179, 1, 0), (186, 78, 1, 1), (125, 173, 1, 0)], 
                            [(5, 248, 2, 1), (83, 255, 1, 1), (23, 173, 2, 2), (170, 32, 0, 0), (100, 36, 0, 2), (211, 206, 2, 0), (177, 252, 2, 0), (65, 74, 0, 1)], 
                            [(252, 121, 0, 2), (133, 243, 2, 1), (6, 41, 2, 0), (234, 218, 1, 1), (35, 70, 1, 1), (46, 254, 0, 2), (166, 254, 2, 2), (186, 154, 2, 1)]]


def run_eyeriss_inputs():
    # get the dataset
    dataset = get_dataset()
    
    num_layers, var_sizes, paddings, strides = get_conv_info(get_alexnet, dataset[0]['image'])

    # get the loop objects
    loops = []

    # layer 1
    eyeriss_vars_1 = [('q', 5), ('c', 3), ('m', 4), ('p', 55), ('q', 11, True), ('s', 11, True),
                      ('r', 11), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_1i = [0, 2, 6]
    mem_dividers_1w = [0, 6]
    eyeriss_injection_1 = Loop.loop(eyeriss_vars_1, mem_dividers_1i, d_type='i', sizes=var_sizes[0], input_strides=[4, 4])
    loops.append(eyeriss_injection_1)

    # layer 2
    eyeriss_vars_2 = [('q', 9), ('c', 16), ('m', 3), ('p', 27), ('m', 4, True), ('q', 3, True), ('c', 2, True),
                     ('s', 5, True), ('r', 5), ('c', 2), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_2i = [0, 2, 8]
    mem_dividers_2w = [0, 8]
    eyeriss_injection_2 = Loop.loop(eyeriss_vars_2, mem_dividers_2i, d_type='i', sizes=var_sizes[1], input_strides=[1, 1])
    loops.append(eyeriss_injection_2)

    # layer 3
    eyeriss_vars_3 = [('m', 3), ('c', 12), ('m', 8), ('p', 13), ('q', 13, True), ('c', 4, True),
                     ('s', 3, True), ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_3i = [0, 2, 7]
    mem_dividers_3w = [0, 6]
    eyeriss_injection_3 = Loop.loop(eyeriss_vars_3, mem_dividers_3i, d_type='i', sizes=var_sizes[2], input_strides=[1, 1])
    loops.append(eyeriss_injection_3)

    # layer 4
    eyeriss_vars_4 = [('c', 24), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_4i = [0, 2, 6]
    mem_dividers_4w = [0, 6]
    eyeriss_injection_4 = Loop.loop(eyeriss_vars_4, mem_dividers_4i, d_type='i', sizes=var_sizes[3], input_strides=[1, 1])
    loops.append(eyeriss_injection_4)

    # layer 5
    eyeriss_vars_5 = [('c', 16), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_5i = [0, 2, 6]
    mem_dividers_5w = [0, 6]
    eyeriss_injection_5 = Loop.loop(eyeriss_vars_5, mem_dividers_5i, d_type='i', sizes=var_sizes[4], input_strides=[1, 1])
    loops.append(eyeriss_injection_5)

    VAL_MAX = [65.44505310058594, 153.95208740234375, 238.4318389892578, 137.9015350341797, 79.4622573852539]
    img_inds = [46682, 7174, 25266, 20546, 21432, 11278, 34091, 43676, 36650, 35665, 1230, 5002, 43288, 40393, 7420, 32524, 34305, 18479, 7068, 10317, 40909, 408, 36743, 13065, 29275, 19160, 39110, 48822, 32220, 43096, 1219, 3221, 27341, 25203, 8125, 42698, 14530, 3210, 44292, 48187, 15912, 17289, 46640, 12198, 11693, 9411, 27420, 41892, 36781, 40879, 16616, 33695, 19354, 14178, 2492, 2783, 45270, 25813, 17976, 17925, 1394, 25675, 42908, 22083, 15782, 46222, 45715, 4099, 6151, 46450, 33650, 18904, 11190, 36895, 49823, 16408, 40557, 9920, 30480, 9598, 17606, 14870, 38048, 26602, 8300, 2585, 33886, 20564, 1501, 7154, 36094, 25587, 5677, 46361, 2653, 12684, 837, 31840, 24381, 38293]
    
    debug = True
    mod_inj = model_injection(get_alexnet, dataset, 'alexnet', 'eyeriss', loops, maxes=VAL_MAX, overwrite=False, debug=debug)
    mod_inj.full_inject(mode="bit", bit=5, img_inds=img_inds, debug=debug, inj_sites=[])

def run_eyeriss_weights():
    # get the dataset and network
    dataset = get_dataset()
    alexnet = get_alexnet()

    # get the loop objects
    loops = []
    d_type = 'w'
    
    num_layers, var_sizes, paddings, strides = get_conv_info(get_alexnet, dataset[0]['image'])

    # layer 1
    eyeriss_vars_1 = [('q', 5), ('c', 3), ('m', 4), ('p', 55), ('q', 11, True), ('s', 11, True),
                     ('r', 11), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_1 = [0, 6]
    eyeriss_injection_1 = Loop.loop(eyeriss_vars_1, mem_dividers_1, d_type=d_type, sizes=var_sizes[0], input_strides=[4, 4])
    loops.append(eyeriss_injection_1)

    # layer 2
    eyeriss_vars_2 = [('q', 9), ('c', 16), ('m', 3), ('p', 27), ('m', 4, True), ('q', 3, True), ('c', 2, True),
                     ('s', 5, True), ('r', 5), ('c', 2), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_2 = [0, 8]
    eyeriss_injection_2 = Loop.loop(eyeriss_vars_2, mem_dividers_2, d_type=d_type, sizes=var_sizes[1], input_strides=[1, 1])
    loops.append(eyeriss_injection_2)

    # layer 3
    eyeriss_vars_3 = [('m', 3), ('c', 12), ('m', 8), ('p', 13), ('q', 13, True), ('c', 4, True),
                     ('s', 3, True), ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_3 = [0, 7]
    eyeriss_injection_3 = Loop.loop(eyeriss_vars_3, mem_dividers_3, d_type=d_type, sizes=var_sizes[2], input_strides=[1, 1])
    loops.append(eyeriss_injection_3)

    # layer 4
    eyeriss_vars_4 = [('c', 24), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_4 = [0, 6]
    eyeriss_injection_4 = Loop.loop(eyeriss_vars_4, mem_dividers_4, d_type=d_type, sizes=var_sizes[3], input_strides=[1, 1])
    loops.append(eyeriss_injection_4)

    # layer 5
    eyeriss_vars_5 = [('c', 16), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_5 = [0, 6]
    eyeriss_injection_5 = Loop.loop(eyeriss_vars_5, mem_dividers_5, d_type=d_type, sizes=var_sizes[4], input_strides=[1, 1])
    loops.append(eyeriss_injection_5)

    VAL_MAX = [65.44505310058594, 153.95208740234375, 238.4318389892578, 137.9015350341797, 79.4622573852539]
    
    debug = True
    mod_inj = model_injection(get_alexnet, dataset, 'alexnet', 'eyeriss', loops, 
                                  d_type=d_type, maxes=VAL_MAX, file_addon="", debug=debug)
    mod_inj.full_inject(mode="bit", bit=5, img_inds=[], debug=debug, inj_sites=[])

def run_nvdla_inputs():
    # get the dataset and network
    dataset = get_dataset()
    
    num_layers, var_sizes, paddings, strides = get_conv_info(get_alexnet, dataset[0]['image'])

    # get the loop objects
    loops = []

    # layer 1
    # 1: [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
    nvdla_vars_1 = [('m', 4), ('m', 16, True), ('c', 3), ('q', 11), ('p', 11), ('c', 1), 
                    ('s', 11), ('r', 11), ('q', 5), ('p', 5), ('r', 1), ('s', 1)]
    mem_dividers_1 = [0, 1, 10]
    nvdla_injection_1 = Loop.loop(nvdla_vars_1, mem_dividers_1, d_type='i', sizes=var_sizes[0], paddings=paddings[0], input_strides=[4, 4])
    loops.append(nvdla_injection_1)

    # layer 2
    # 2: [(192, 64, 5, 5),     (1, 192, 27, 27),   (1, 64, 27, 27),    (2, 2),     (1, 1)]
    nvdla_vars_2 = [('m', 12), ('m', 16, True), ('c', 64), ('q', 8), ('p', 8), ('c', 1), 
                    ('s', 5), ('r', 5), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_2 = [0, 1, 10]
    nvdla_injection_2 = Loop.loop(nvdla_vars_2, mem_dividers_2, d_type='i', sizes=var_sizes[1], paddings=paddings[1], input_strides=[1, 1])
    loops.append(nvdla_injection_2)

    # layer 3
    # 3: [(384, 192, 3, 3),    (1, 384, 13, 13),   (1, 192, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_3 = [('m', 24), ('m', 16, True), ('c', 192), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_3 = [0, 1, 10]
    nvdla_injection_3 = Loop.loop(nvdla_vars_3, mem_dividers_3, d_type='i', sizes=var_sizes[2], paddings=paddings[2], input_strides=[1, 1])
    loops.append(nvdla_injection_3)

    # layer 4
    # 4: [(256, 384, 3, 3),    (1, 256, 13, 13),   (1, 384, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_4 = [('m', 16), ('m', 16, True), ('c', 384), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_4 = [0, 1, 10]
    nvdla_injection_4 = Loop.loop(nvdla_vars_4, mem_dividers_4, d_type='i', sizes=var_sizes[3], paddings=paddings[3], input_strides=[1, 1])
    loops.append(nvdla_injection_4)

    # layer 5
    # 5: [(256, 256, 3, 3),    (1, 256, 13, 13),   (1, 256, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_5 = [('m', 16), ('m', 16, True), ('c', 256), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_5 = [0, 1, 10]
    nvdla_injection_5 = Loop.loop(nvdla_vars_5, mem_dividers_5, d_type='i', sizes=var_sizes[4], paddings=paddings[4], input_strides=[1, 1])
    loops.append(nvdla_injection_5)

    VAL_MAX = [65.44505310058594, 153.95208740234375, 238.4318389892578, 137.9015350341797, 79.4622573852539]
    mod_inj = model_injection(get_alexnet, dataset, 'alexnet', 'nvdla', loops, maxes=VAL_MAX, overwrite=True, debug=True)
    mod_inj.full_inject(mode="bit", bit=5, img_inds=[], debug=True, inj_sites=[], layers=[])

def run_nvdla_weights():
    # get the dataset and network
    dataset = get_dataset()
    
    num_layers, var_sizes, paddings, strides = get_conv_info(get_alexnet, dataset[0]['image'])

    # get the loop objects
    loops = []
    
    d_type = 'w'
    
    # layer 1
    # 1: [(64, 3, 11, 11),     (1, 64, 55, 55),    (1, 3, 224, 224),   (2, 2),     (4, 4)]
    nvdla_vars_1 = [('m', 4), ('m', 16, True), ('c', 3), ('q', 11), ('p', 11), ('c', 1), 
                    ('s', 11), ('r', 11), ('q', 5), ('p', 5), ('r', 1), ('s', 1)]
    mem_dividers_1w = [0, 1, 10]
    nvdla_injection_1 = Loop.loop(nvdla_vars_1, mem_dividers_1w, d_type=d_type, sizes=var_sizes[0], paddings=paddings[0], input_strides=[4, 4])
    loops.append(nvdla_injection_1)

    # layer 2
    # 2: [(192, 64, 5, 5),     (1, 192, 27, 27),   (1, 64, 27, 27),    (2, 2),     (1, 1)]
    nvdla_vars_2 = [('m', 12), ('m', 16, True), ('c', 64), ('q', 8), ('p', 8), ('c', 1), 
                    ('s', 5), ('r', 5), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_2w = [0, 1, 10]
    nvdla_injection_2 = Loop.loop(nvdla_vars_2, mem_dividers_2w, d_type=d_type, sizes=var_sizes[1], paddings=paddings[1], input_strides=[1, 1])
    loops.append(nvdla_injection_2)

    # layer 3
    # 3: [(384, 192, 3, 3),    (1, 384, 13, 13),   (1, 192, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_3 = [('m', 24), ('m', 16, True), ('c', 192), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_3w = [0, 1, 10]
    nvdla_injection_3 = Loop.loop(nvdla_vars_3, mem_dividers_3w, d_type=d_type, sizes=var_sizes[2], paddings=paddings[2], input_strides=[1, 1])
    loops.append(nvdla_injection_3)

    # layer 4
    # 4: [(256, 384, 3, 3),    (1, 256, 13, 13),   (1, 384, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_4 = [('m', 16), ('m', 16, True), ('c', 384), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_4w = [0, 1, 10]
    nvdla_injection_4 = Loop.loop(nvdla_vars_4, mem_dividers_4w, d_type=d_type, sizes=var_sizes[3], paddings=paddings[3], input_strides=[1, 1])
    loops.append(nvdla_injection_4)

    # layer 5
    # 5: [(256, 256, 3, 3),    (1, 256, 13, 13),   (1, 256, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_5 = [('m', 16), ('m', 16, True), ('c', 256), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_5w = [0, 1, 10]
    nvdla_injection_5 = Loop.loop(nvdla_vars_5, mem_dividers_5w, d_type=d_type, sizes=var_sizes[4], paddings=paddings[4], input_strides=[1, 1])
    loops.append(nvdla_injection_5)

    VAL_MAX = [65.44505310058594, 153.95208740234375, 238.4318389892578, 137.9015350341797, 79.4622573852539]
    img_inds = [46682, 7174, 25266, 20546, 21432, 11278, 34091, 43676, 36650, 35665, 1230, 5002, 43288, 40393, 7420, 32524, 34305, 18479, 7068, 10317, 40909, 408, 36743, 13065, 29275, 19160, 39110, 48822, 32220, 43096, 1219, 3221, 27341, 25203, 8125, 42698, 14530, 3210, 44292, 48187, 15912, 17289, 46640, 12198, 11693, 9411, 27420, 41892, 36781, 40879, 16616, 33695, 19354, 14178, 2492, 2783, 45270, 25813, 17976, 17925, 1394, 25675, 42908, 22083, 15782, 46222, 45715, 4099, 6151, 46450, 33650, 18904, 11190, 36895, 49823, 16408, 40557, 9920, 30480, 9598, 17606, 14870, 38048, 26602, 8300, 2585, 33886, 20564, 1501, 7154, 36094, 25587, 5677, 46361, 2653, 12684, 837, 31840, 24381, 38293]
    mod_inj = model_injection(get_alexnet, dataset, 'alexnet', 'nvdla', loops, maxes=VAL_MAX, overwrite=False, debug=True, d_type=d_type)
    mod_inj.full_inject(mode="bit", bit=5, img_inds=img_inds, debug=True, inj_sites=[], layers=[])
    
def run_eyeriss_resnet18_inputs():
    # get the dataset and network
    dataset = get_dataset()
    
    num_layers, var_sizes, paddings, strides = get_conv_info(get_resnet18, dataset[0]['image'])

    # get the loop objects
    loops = []
    
    inj_inds = []

    # layer 0
    eyeriss_vars_0 = [('q', 5), ('c', 3), ('m', 4), ('p', 55), ('q', 11, True), ('s', 11, True),
                      ('r', 11), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_0i = [0, 2, 6]
    mem_dividers_0w = [0, 6]
    eyeriss_injection_1 = Loop.loop(eyeriss_vars_0, mem_dividers_0i, d_type='i', sizes=var_sizes[0], input_strides=[4, 4])
    loops.append(eyeriss_injection_1)
    
    # layer 1
    eyeriss_vars_1 = [('q', 5), ('c', 3), ('m', 4), ('p', 55), ('q', 11, True), ('s', 11, True),
                      ('r', 11), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_1i = [0, 2, 6]
    mem_dividers_1w = [0, 6]
    eyeriss_injection_1 = Loop.loop(eyeriss_vars_1, mem_dividers_1i, d_type='i', sizes=var_sizes[0], input_strides=[4, 4])
    loops.append(eyeriss_injection_1)

    # layer 2
    eyeriss_vars_2 = [('q', 9), ('c', 16), ('m', 3), ('p', 27), ('m', 4, True), ('q', 3, True), ('c', 2, True),
                     ('s', 5, True), ('r', 5), ('c', 2), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_2i = [0, 2, 8]
    mem_dividers_2w = [0, 8]
    eyeriss_injection_2 = Loop.loop(eyeriss_vars_2, mem_dividers_2i, d_type='i', sizes=var_sizes[1], input_strides=[1, 1])
    loops.append(eyeriss_injection_2)

    # layer 3
    eyeriss_vars_3 = [('m', 3), ('c', 12), ('m', 8), ('p', 13), ('q', 13, True), ('c', 4, True),
                     ('s', 3, True), ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_3i = [0, 2, 7]
    mem_dividers_3w = [0, 6]
    eyeriss_injection_3 = Loop.loop(eyeriss_vars_3, mem_dividers_3i, d_type='i', sizes=var_sizes[2], input_strides=[1, 1])
    loops.append(eyeriss_injection_3)

    # layer 4
    eyeriss_vars_4 = [('c', 24), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_4i = [0, 2, 6]
    mem_dividers_4w = [0, 6]
    eyeriss_injection_4 = Loop.loop(eyeriss_vars_4, mem_dividers_4i, d_type='i', sizes=var_sizes[3], input_strides=[1, 1])
    loops.append(eyeriss_injection_4)

    # layer 5
    eyeriss_vars_5 = [('c', 16), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_5i = [0, 2, 6]
    mem_dividers_5w = [0, 6]
    eyeriss_injection_5 = Loop.loop(eyeriss_vars_5, mem_dividers_5i, d_type='i', sizes=var_sizes[4], input_strides=[1, 1])
    loops.append(eyeriss_injection_5)
    
    # layer 6
    eyeriss_vars_6 = [('c', 16), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_5i = [0, 2, 6]
    mem_dividers_5w = [0, 6]
    eyeriss_injection_5 = Loop.loop(eyeriss_vars_5, mem_dividers_5i, d_type='i', sizes=var_sizes[4], input_strides=[1, 1])
    loops.append(eyeriss_injection_5)
    
    # layer 5
    eyeriss_vars_5 = [('c', 16), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_5i = [0, 2, 6]
    mem_dividers_5w = [0, 6]
    eyeriss_injection_5 = Loop.loop(eyeriss_vars_5, mem_dividers_5i, d_type='i', sizes=var_sizes[4], input_strides=[1, 1])
    loops.append(eyeriss_injection_5)

    VAL_MAX = [65.44505310058594, 153.95208740234375, 238.4318389892578, 137.9015350341797, 79.4622573852539]
    
    debug = True
    mod_inj = model_injection(get_resnet18, dataset, 'alexnet', 'eyeriss', loops, maxes=VAL_MAX, overwrite=False, debug=debug)
    mod_inj.full_inject(mode="bit", bit=5, img_inds=[], debug=debug, inj_sites=inj_inds)

def run_plot_input():
    arch_name = 'nvdla'
    mem_levels = ['DRAM', 'GlobalBuffer', 'InputRegs']
    
    # arch_name = 'eyeriss'
    # mem_levels = ['DRAM', 'GlobalBuffer', 'InputRegs']
    
    plot_eyeriss = plotter(arch_name, 5, 'alexnet', d_type='i', add_on="")
    plot_eyeriss.plot(0.55, mem_levels, img_name="input", agg_layers=True)
    
def run_plot_weight():
    arch_name = 'nvdla'
    mem_levels = ['DRAM', 'GlobalBuffer', 'WeightRegs']
    
    # arch_name = 'eyeriss'
    # mem_levels = ['DRAM', 'WeightRegs']
    
    plot_eyeriss = plotter(arch_name, 5, 'alexnet', d_type='w', add_on="")
    plot_eyeriss.plot(0.55, mem_levels, img_name="weight")


if __name__=="__main__":
    # net = get_resnet18()
    # print_layer_sizes(net)
    
    run_nvdla_inputs()
    # run_nvdla_weights()
    
    # run_eyeriss_inputs()
    # run_eyeriss_weights()
    
    run_plot_input()
    # run_plot_weight()
    
    pass