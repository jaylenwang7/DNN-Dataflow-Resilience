from dataset import get_dataset
from helpers import *
from info_model import *
from loop import Loop
from model_injection import ModelInjection
from plotter import Plotter
from max_model import *
# from scalene import scalene_profiler

# if you want standard set of img indices to use across multiple experiments
# set of 100 images
sample_img_inds = [46682, 7174, 25266, 20546, 21432, 11278, 34091, 43676, 36650, 35665, 
                   1230, 5002, 43288, 40393, 7420, 32524, 34305, 18479, 7068, 10317, 
                   40909, 408, 36743, 13065, 29275, 19160, 39110, 48822, 32220, 43096, 
                   1219, 3221, 27341, 25203, 8125, 42698, 14530, 3210, 44292, 48187, 
                   15912, 17289, 46640, 12198, 11693, 9411, 27420, 41892, 36781, 40879, 
                   16616, 33695, 19354, 14178, 2492, 2783, 45270, 25813, 17976, 17925, 
                   1394, 25675, 42908, 22083, 15782, 46222, 45715, 4099, 6151, 46450, 
                   33650, 18904, 11190, 36895, 49823, 16408, 40557, 9920, 30480, 9598, 
                   17606, 14870, 38048, 26602, 8300, 2585, 33886, 20564, 1501, 7154, 
                   36094, 25587, 5677, 46361, 2653, 12684, 837, 31840, 24381, 38293]

sample_resnet18_correct_img_inds = [25152, 25724, 24901, 17371, 38038, 33971, 20401, 2349, 1305, 29513, 
                                    41135, 3268, 22335, 6307, 5992, 28434, 4856, 38515, 634, 4345, 
                                    33547, 23846, 20115, 30897, 21166, 45335, 7651, 11808, 4008, 
                                    1289, 32803, 25221, 27092, 11189, 27035, 25078, 10862, 24554, 
                                    25242, 43925, 7877, 43380, 28138, 39585, 22966, 12599, 20741, 
                                    11821, 18569, 11472, 47851, 22595, 39458, 4906, 40229, 35531, 
                                    27236, 8819, 3664, 33885, 24701, 43354, 29861, 26381, 16112, 
                                    30014, 44986, 10766, 17598, 22301, 11559, 17620, 47615, 9839, 
                                    6363, 40093, 26555, 14834, 34648, 17463, 27690, 28057, 22569, 
                                    44976, 24350, 42394, 32133, 21753, 36895, 1215, 30869, 44203, 
                                    30583, 9135, 11787, 27549, 35371, 15303, 1263, 12495]

sample_resnet18_correct_img_inds2 = [47954, 15766, 27187, 11948, 39609, 19792, 4785, 7539, 39025, 27848, 
                                     21369, 30689, 19290, 17066, 37888, 29550, 35999, 38688, 14589, 17802, 
                                     20755, 24738, 1515, 45277, 24869, 38481, 20884, 19879, 30945, 26961, 
                                     49454, 10504, 2082, 19615, 5791, 23980, 22481, 26090, 42121, 22937, 
                                     39033, 14041, 33218, 40055, 6033, 44883, 36531, 25603, 32308, 40989, 
                                     38478, 22163, 40921, 15540, 12577, 13161, 26162, 21088, 30717, 41688, 
                                     12317, 17581, 26933, 43763, 38480, 37230, 36331, 36472, 18034, 22320, 
                                     31481, 2936, 16301, 30080, 45712, 47908, 29129, 16199, 46379, 21498, 
                                     37792, 14990, 25322, 49822, 48683, 12616, 36544, 8044, 30432, 16955, 
                                     23151, 16317, 49390, 48113, 49691, 15180, 773, 18106, 35598, 28573]


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

# all taken over 1000 trials of randomly chosen images
ALEXNET_MAX = [60.63791275024414, 153.95208740234375, 157.78179931640625, 114.32186126708984, 59.255977630615234]
ALEXNET_MIN = [-64.5768814086914, -287.6980285644531, -304.3017272949219, -232.31802368164062, -115.81592559814453]

RESNET18_MAX = [36.13761901855469, 10.887314796447754, 3.855881929397583, 9.794642448425293, 3.2933764457702637, 
                8.875794410705566, 4.089364528656006, 4.49278450012207, 5.494577884674072, 3.7413206100463867, 
                5.697751522064209, 3.7465898990631104, 1.6894534826278687, 4.060337543487549, 2.0673913955688477,
                3.928053855895996, 2.009352684020996, 2.0958352088928223, 3.4922521114349365, 3.5720021724700928]
RESNET18_MIN = [-35.18002700805664, -16.452085494995117, -7.513538837432861, -11.784904479980469, -5.422745704650879, 
                -9.174695014953613, -3.548478841781616, -5.52264404296875, -5.273807048797607, -3.147308349609375, 
                -6.394559383392334, -4.2612624168396, -1.897513508796692, -5.157448768615723, -2.5855915546417236, 
                -5.034069538116455, -1.9327064752578735, -1.9150633811950684, -5.472661018371582, -0.9709599614143372]

MOBILENETV3_MAX = [50.529754638671875, 159.1470184326172, 0.0, 1.3555821180343628, 16.691547393798828, 84.53617095947266, 
                   63.21651077270508, 32.450191497802734, 61.84816360473633, 8.025238990783691, 16.3727970123291, 
                   35.572021484375, 27.39885139465332, 2.7719931602478027, 3.0307681560516357, 12.965798377990723, 
                   11.035303115844727, 6.568864345550537, 5.350219249725342, 4.612285137176514, 2.2055413722991943, 
                   14.39225959777832, 9.653714179992676, 3.4543395042419434, 3.8758585453033447, 3.6229655742645264, 
                   43.400718688964844, 29.129955291748047, 6.451019287109375, 5.666597366333008, 16.60890007019043, 
                   17.242544174194336, 8.99718952178955, 3.5118703842163086, 3.9826126098632812, 2.1765975952148438, 
                   20.474945068359375, 9.144041061401367, 5.380688667297363, 5.058890342712402, 6.589954376220703, 
                   11.745735168457031, 6.583943843841553, 5.67405891418457, 10.3109769821167, 3.338714599609375, 
                   14.867364883422852, 4.966531276702881, 9.095283508300781, 12.926734924316406, 6.952933311462402, 46.999759674072266]
MOBILENETV3_MIN = [-47.88440704345703, -39.84259033203125, -0.021632174029946327, -1.137220025062561, -17.77903938293457, 
                   -95.75336456298828, -42.84724807739258, -78.0679702758789, -110.25776672363281, -11.161054611206055, 
                   -13.076906204223633, -56.625022888183594, -57.72230529785156, -24.24806785583496, -3.1747801303863525, 
                   -17.344646453857422, -11.472933769226074, -8.96963882446289, -11.953343391418457, -9.164261817932129, 
                   -2.3819632530212402, -14.280878067016602, -7.006469249725342, -4.329557418823242, -7.006653785705566, 
                   -3.565438747406006, -41.6951904296875, -24.19619369506836, -14.262306213378906, -6.403060436248779,
                   -17.50071907043457, -12.253621101379395, -6.836885929107666, -2.2842328548431396, -4.8658528327941895, 
                   -2.354025363922119, -19.027925491333008, -16.893497467041016, -5.314076900482178, -6.654882907867432, 
                   -6.4946699142456055, -12.471386909484863, -9.682262420654297, -3.6828505992889404, -7.681639194488525, 
                   -3.773265838623047, -18.606332778930664, -7.794033050537109, -5.584883213043213, -8.374652862548828, 
                   -8.250393867492676, -35.55913162231445]

IMAGENET_IMGS_PATH = '../loop-injection/ILSVRC2012_img_val/'
IMAGENET_LABELS_PATH = '../loop-injection/LOC_val_solution.csv'

def run_eyeriss_inputs():
    # get the dataset
    dataset = get_dataset(IMAGENET_LABELS_PATH, IMAGENET_IMGS_PATH)
    
    num_layers, var_sizes, paddings, strides = get_conv_info(get_alexnet, dataset[0]['image'])

    # get the loop objects
    loops = []

    # layer 1
    eyeriss_vars_1 = [('q', 5), ('c', 3), ('m', 4), ('p', 55), ('q', 11, True), ('s', 11, True),
                      ('r', 11), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_1i = [0, 2, 6]
    mem_dividers_1w = [0, 6]
    eyeriss_injection_1 = Loop(eyeriss_vars_1, mem_dividers_1i, d_type='i', sizes=var_sizes[0], input_strides=[4, 4])
    loops.append(eyeriss_injection_1)

    # layer 2
    eyeriss_vars_2 = [('q', 9), ('c', 16), ('m', 3), ('p', 27), ('m', 4, True), ('q', 3, True), ('c', 2, True),
                     ('s', 5, True), ('r', 5), ('c', 2), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_2i = [0, 2, 8]
    mem_dividers_2w = [0, 8]
    eyeriss_injection_2 = Loop(eyeriss_vars_2, mem_dividers_2i, d_type='i', sizes=var_sizes[1], input_strides=[1, 1])
    loops.append(eyeriss_injection_2)

    # layer 3
    eyeriss_vars_3 = [('m', 3), ('c', 12), ('m', 8), ('p', 13), ('q', 13, True), ('c', 4, True),
                     ('s', 3, True), ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_3i = [0, 2, 7]
    mem_dividers_3w = [0, 6]
    eyeriss_injection_3 = Loop(eyeriss_vars_3, mem_dividers_3i, d_type='i', sizes=var_sizes[2], input_strides=[1, 1])
    loops.append(eyeriss_injection_3)

    # layer 4
    eyeriss_vars_4 = [('c', 24), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_4i = [0, 2, 6]
    mem_dividers_4w = [0, 6]
    eyeriss_injection_4 = Loop(eyeriss_vars_4, mem_dividers_4i, d_type='i', sizes=var_sizes[3], input_strides=[1, 1])
    loops.append(eyeriss_injection_4)

    # layer 5
    eyeriss_vars_5 = [('c', 16), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_5i = [0, 2, 6]
    mem_dividers_5w = [0, 6]
    eyeriss_injection_5 = Loop(eyeriss_vars_5, mem_dividers_5i, d_type='i', sizes=var_sizes[4], input_strides=[1, 1])
    loops.append(eyeriss_injection_5)
    
    debug = True
    mod_inj = ModelInjection(get_alexnet, dataset, 'alexnet', 'eyeriss', loops, maxes=ALEXNET_MAX, mins=ALEXNET_MIN, overwrite=True, debug=debug)
    mod_inj.full_inject(mode="bit", bit=5, img_inds=[], debug=debug, inj_sites=[], layers=[], sample_correct=False)

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
    eyeriss_injection_1 = Loop(eyeriss_vars_1, mem_dividers_1, d_type=d_type, sizes=var_sizes[0], input_strides=[4, 4])
    loops.append(eyeriss_injection_1)

    # layer 2
    eyeriss_vars_2 = [('q', 9), ('c', 16), ('m', 3), ('p', 27), ('m', 4, True), ('q', 3, True), ('c', 2, True),
                     ('s', 5, True), ('r', 5), ('c', 2), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_2 = [0, 8]
    eyeriss_injection_2 = Loop(eyeriss_vars_2, mem_dividers_2, d_type=d_type, sizes=var_sizes[1], input_strides=[1, 1])
    loops.append(eyeriss_injection_2)

    # layer 3
    eyeriss_vars_3 = [('m', 3), ('c', 12), ('m', 8), ('p', 13), ('q', 13, True), ('c', 4, True),
                     ('s', 3, True), ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_3 = [0, 7]
    eyeriss_injection_3 = Loop(eyeriss_vars_3, mem_dividers_3, d_type=d_type, sizes=var_sizes[2], input_strides=[1, 1])
    loops.append(eyeriss_injection_3)

    # layer 4
    eyeriss_vars_4 = [('c', 24), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_4 = [0, 6]
    eyeriss_injection_4 = Loop(eyeriss_vars_4, mem_dividers_4, d_type=d_type, sizes=var_sizes[3], input_strides=[1, 1])
    loops.append(eyeriss_injection_4)

    # layer 5
    eyeriss_vars_5 = [('c', 16), ('m', 16), ('p', 13), ('q', 13, True), ('c', 4, True), ('s', 3, True),
                     ('r', 3), ('c', 4), ('m', 16), ('s', 1), ('r', 1)]
    mem_dividers_5 = [0, 6]
    eyeriss_injection_5 = Loop(eyeriss_vars_5, mem_dividers_5, d_type=d_type, sizes=var_sizes[4], input_strides=[1, 1])
    loops.append(eyeriss_injection_5)
    
    debug = True
    mod_inj = ModelInjection(get_alexnet, dataset, 'alexnet', 'eyeriss', loops, d_type=d_type, maxes=ALEXNET_MAX, mins=ALEXNET_MIN, file_addon="", debug=debug)
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
    nvdla_injection_1 = Loop(nvdla_vars_1, mem_dividers_1, d_type='i', sizes=var_sizes[0], paddings=paddings[0], input_strides=[4, 4])
    loops.append(nvdla_injection_1)

    # layer 2
    # 2: [(192, 64, 5, 5),     (1, 192, 27, 27),   (1, 64, 27, 27),    (2, 2),     (1, 1)]
    nvdla_vars_2 = [('m', 12), ('m', 16, True), ('c', 64), ('q', 8), ('p', 8), ('c', 1), 
                    ('s', 5), ('r', 5), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_2 = [0, 1, 10]
    nvdla_injection_2 = Loop(nvdla_vars_2, mem_dividers_2, d_type='i', sizes=var_sizes[1], paddings=paddings[1], input_strides=[1, 1])
    loops.append(nvdla_injection_2)

    # layer 3
    # 3: [(384, 192, 3, 3),    (1, 384, 13, 13),   (1, 192, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_3 = [('m', 24), ('m', 16, True), ('c', 192), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_3 = [0, 1, 10]
    nvdla_injection_3 = Loop(nvdla_vars_3, mem_dividers_3, d_type='i', sizes=var_sizes[2], paddings=paddings[2], input_strides=[1, 1])
    loops.append(nvdla_injection_3)

    # layer 4
    # 4: [(256, 384, 3, 3),    (1, 256, 13, 13),   (1, 384, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_4 = [('m', 16), ('m', 16, True), ('c', 384), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_4 = [0, 1, 10]
    nvdla_injection_4 = Loop(nvdla_vars_4, mem_dividers_4, d_type='i', sizes=var_sizes[3], paddings=paddings[3], input_strides=[1, 1])
    loops.append(nvdla_injection_4)

    # layer 5
    # 5: [(256, 256, 3, 3),    (1, 256, 13, 13),   (1, 256, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_5 = [('m', 16), ('m', 16, True), ('c', 256), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_5 = [0, 1, 10]
    nvdla_injection_5 = Loop(nvdla_vars_5, mem_dividers_5, d_type='i', sizes=var_sizes[4], paddings=paddings[4], input_strides=[1, 1])
    loops.append(nvdla_injection_5)

    mod_inj = ModelInjection(get_alexnet, dataset, 'alexnet', 'nvdla', loops, maxes=ALEXNET_MAX, mins=ALEXNET_MIN, overwrite=False, debug=True)
    correct_rate = mod_inj.full_inject(mode="bit", bit=5, img_inds=[], debug=True, inj_sites=[], layers=[])
    return correct_rate

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
    nvdla_injection_1 = Loop(nvdla_vars_1, mem_dividers_1w, d_type=d_type, sizes=var_sizes[0], paddings=paddings[0], input_strides=[4, 4])
    loops.append(nvdla_injection_1)

    # layer 2
    # 2: [(192, 64, 5, 5),     (1, 192, 27, 27),   (1, 64, 27, 27),    (2, 2),     (1, 1)]
    nvdla_vars_2 = [('m', 12), ('m', 16, True), ('c', 64), ('q', 8), ('p', 8), ('c', 1), 
                    ('s', 5), ('r', 5), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_2w = [0, 1, 10]
    nvdla_injection_2 = Loop(nvdla_vars_2, mem_dividers_2w, d_type=d_type, sizes=var_sizes[1], paddings=paddings[1], input_strides=[1, 1])
    loops.append(nvdla_injection_2)

    # layer 3
    # 3: [(384, 192, 3, 3),    (1, 384, 13, 13),   (1, 192, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_3 = [('m', 24), ('m', 16, True), ('c', 192), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_3w = [0, 1, 10]
    nvdla_injection_3 = Loop(nvdla_vars_3, mem_dividers_3w, d_type=d_type, sizes=var_sizes[2], paddings=paddings[2], input_strides=[1, 1])
    loops.append(nvdla_injection_3)

    # layer 4
    # 4: [(256, 384, 3, 3),    (1, 256, 13, 13),   (1, 384, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_4 = [('m', 16), ('m', 16, True), ('c', 384), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_4w = [0, 1, 10]
    nvdla_injection_4 = Loop(nvdla_vars_4, mem_dividers_4w, d_type=d_type, sizes=var_sizes[3], paddings=paddings[3], input_strides=[1, 1])
    loops.append(nvdla_injection_4)

    # layer 5
    # 5: [(256, 256, 3, 3),    (1, 256, 13, 13),   (1, 256, 13, 13),   (1, 1),     (1, 1)]
    nvdla_vars_5 = [('m', 16), ('m', 16, True), ('c', 256), ('q', 4), ('p', 4), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers_5w = [0, 1, 10]
    nvdla_injection_5 = Loop(nvdla_vars_5, mem_dividers_5w, d_type=d_type, sizes=var_sizes[4], paddings=paddings[4], input_strides=[1, 1])
    loops.append(nvdla_injection_5)

    mod_inj = ModelInjection(get_alexnet, dataset, 'alexnet', 'nvdla', loops, maxes=ALEXNET_MAX, mins=ALEXNET_MIN, overwrite=False, debug=True, d_type=d_type)
    mod_inj.full_inject(mode="bit", bit=5, img_inds=[], debug=True, inj_sites=[], layers=[])
    
def run_eyeriss_resnet18_inputs():
    # get the dataset and network
    dataset = get_dataset(IMAGENET_LABELS_PATH, IMAGENET_IMGS_PATH)
    d_type='i'
    
    num_layers, var_sizes, paddings, strides = get_conv_info(get_resnet18, dataset[0]['image'])

    # get the loop objects
    inj_inds = []
    layers = []
    
    loops, names = get_loops(get_resnet18, 'timeloop_maps/eyeriss/resnet18/', var_sizes, paddings, strides)
    
    debug = True
    
    mod_inj = ModelInjection(get_resnet18, dataset, 'resnet18', 'eyeriss', loops, maxes=RESNET18_MAX, mins=RESNET18_MIN, overwrite=False, debug=debug, d_type=d_type, max_range=True)
    correct_rate = mod_inj.full_inject(mode="rand_bit", bit=range(1, 12), img_inds=sample_resnet18_correct_img_inds, debug=debug, inj_sites=inj_inds, layers=layers)
    
def run_eyeriss_resnet18_weights():
    # get the dataset and network
    dataset = get_dataset(IMAGENET_LABELS_PATH, IMAGENET_IMGS_PATH)
    d_type='w'
    
    num_layers, var_sizes, paddings, strides = get_conv_info(get_resnet18, dataset[0]['image'])

    # get the loop objects
    inj_inds = []
    layers = []
    
    loops, names = get_loops(get_resnet18, 'timeloop_maps/eyeriss/resnet18/', var_sizes, paddings, strides)
    
    debug = True
    
    mod_inj = ModelInjection(get_resnet18, dataset, 'resnet18', 'eyeriss', loops, maxes=RESNET18_MAX, mins=RESNET18_MIN, overwrite=False, debug=debug, d_type=d_type, max_range=True)
    correct_rate = mod_inj.full_inject(mode="rand_bit", bit=range(1, 12), img_inds=sample_resnet18_correct_img_inds, debug=debug, inj_sites=inj_inds, layers=layers)
    
def run_nvdla_resnet18_inputs():
    # get the dataset and network
    dataset = get_dataset(IMAGENET_LABELS_PATH, IMAGENET_IMGS_PATH)
    d_type='i'
    
    num_layers, var_sizes, paddings, strides = get_conv_info(get_resnet18, dataset[0]['image'])

    # get the loop objects
    loops = []
    
    d_type = 'i'
    
    def add_loop(loop_list, n):
        for i in range(n):
            loop_list.append(0)
        return loop_list
    
    # layer 0
    nvdla_vars = [('m', 4), ('m', 16, True), ('c', 3), ('q', 28), ('p', 28), ('c', 1), 
                    ('s', 7), ('r', 7), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[0], input_strides=strides[0])
    loops.append(nvdla_injection)
    
    # layer 1
    nvdla_vars = [('m', 4), ('m', 16, True), ('c', 64), ('q', 14), ('p', 14), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[1], input_strides=strides[1])
    loops.append(nvdla_injection)
    
    loops = add_loop(loops, 3)
    
    # layer 5
    nvdla_vars = [('m', 8), ('m', 16, True), ('c', 64), ('q', 7), ('p', 7), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[5], input_strides=strides[5])
    loops.append(nvdla_injection)
    
    # layer 6
    nvdla_vars = [('m', 8), ('m', 16, True), ('c', 128), ('q', 7), ('p', 7), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 4), ('p', 4), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[6], input_strides=strides[6])
    loops.append(nvdla_injection)
    
    loops = add_loop(loops, 3)
    
    # layer 10
    nvdla_vars = [('m', 16), ('m', 16, True), ('c', 128), ('q', 7), ('p', 7), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 2), ('p', 2), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[10], input_strides=strides[10])
    loops.append(nvdla_injection)
    
    loops = add_loop(loops, 2)
    
    # layer 13
    nvdla_vars = [('m', 16), ('m', 16, True), ('c', 256), ('q', 7), ('p', 7), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 2), ('p', 2), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[13], input_strides=strides[13])
    loops.append(nvdla_injection)
    
    loops = add_loop(loops, 4)
    
    # layer 18
    nvdla_vars = [('m', 32), ('m', 16, True), ('c', 512), ('q', 1), ('p', 1), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 7), ('p', 7), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[18], input_strides=strides[18])
    loops.append(nvdla_injection)
    
    # layer 19
    nvdla_vars = [('m', 32), ('m', 16, True), ('c', 512), ('q', 1), ('p', 1), ('c', 1), 
                    ('s', 3), ('r', 3), ('q', 7), ('p', 7), ('r', 1), ('s', 1)]
    mem_dividers = [0, 1, 10]
    nvdla_injection = Loop(nvdla_vars, mem_dividers, d_type=d_type, sizes=var_sizes[0], paddings=paddings[19], input_strides=strides[19])
    loops.append(nvdla_injection)
    
    debug = True
    inj_inds = []
    layers = [0, 1, 5, 6, 10, 13, 18, 19]
    
    mod_inj = ModelInjection(get_resnet18, dataset, 'resnet18', 'nvdla', loops, maxes=RESNET18_MAX, mins=RESNET18_MIN, overwrite=False, debug=debug, d_type=d_type, max_range=True)
    correct_rate = mod_inj.full_inject(mode="bit", bit=range(1, 9), img_inds=sample_resnet18_correct_img_inds2, debug=debug, inj_sites=inj_inds, layers=layers)
    
def get_network_max(get_net, get_dataset, n=1000):
    net = get_net()
    dataset = get_dataset(csv_file=IMAGENET_LABELS_PATH, root_dir=IMAGENET_IMGS_PATH)
    maxes, mins = get_range(net, dataset, n=n)
    print(maxes)
    print(mins)
    # pickle_object(maxes, 'maxes.pickle')
    # pickle_object(mins, 'maxes.pickle')
    
def get_resnet18_max_img(img):
    net = get_resnet18()
    maxes, mins = get_range_img(net, img)
    print(maxes)
    print(mins)

def run_plot_input(arch_name, net_name, add_on='', d_type='i', correct_rate=1.0):
    mem_levels = ['DRAM', 'shared_glb', 'ifmap_spad']
    # layers_to_exclude = {7, 12, 17}
    # num_layers = 20
    # layers = [l for l in range(num_layers) if l not in layers_to_exclude]
    layers = [0, 1, 5, 6, 10, 13, 18, 19]
    plot_eyeriss = Plotter(arch_name, net_name, d_type=d_type, add_on=add_on, layers=layers)
    # plot_eyeriss.plot(correct_rate=correct_rate, xlabels=mem_levels, img_name="input", agg_layers=False)
    # plot_eyeriss.plot_zeros()
    plot_eyeriss.plot_v2()
    
def run_plot_weight():
    arch_name = 'nvdla'
    mem_levels = ['DRAM', 'GlobalBuffer', 'WeightRegs']
    
    # arch_name = 'eyeriss'
    # mem_levels = ['DRAM', 'WeightRegs']
    
    plot_eyeriss = Plotter(arch_name, 5, 'alexnet', d_type='w', add_on="")
    plot_eyeriss.plot(0.55, mem_levels, img_name="weight")


if __name__=="__main__":
    # RUN BELOW TO GET THE VALUE RANGE
    # get_network_max(get_mobilenet_v3_small, get_dataset, n=10000)
    
    # RUN THE BELOW TO PRINT MODEL LAYER INFO
    print_layer_sizes(get_resnet18(), 'resnet18')
    # dataset = get_dataset()
    # print(get_conv_info(get_resnet18, dataset[0]['image'])[1][6])
    
    # RUN THE BELOW FOR NVDLA TESTS W/ ALEXNET
    # run_nvdla_inputs()
    # run_nvdla_weights()
    
    # RUN THE BELOW FOR EYERISS TESTS W/ EYERISS
    # run_eyeriss_inputs()
    # run_eyeriss_weights()
    
    # RUN THE BELOW FOR PLOTTING ONCE YOU HAVE RESULTS
    # run_plot_input('eyeriss', 'resnet18', add_on='0')
    # run_plot_weight()
    
    pass