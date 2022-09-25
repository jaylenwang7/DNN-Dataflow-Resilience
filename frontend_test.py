import torch
import time
from helpers import *
from dataset import *
import loop as Loop
from inject_model import InjectModel
from clean_model import CleanModel, run_clean
from info_model import get_layer_info
from typing import Callable

def run_thousand(get_net: Callable) -> None:
      
    dataset = get_dataset()
    net = get_net()
    img = dataset[0]['images']

    start_time = time.time()
    for i in range(1000):
        net(img)
    print("--- %s seconds ---" % (time.time() - start_time))
    pass

def run_thousand_injected_alexnet() -> None:
    M = 192
    C = 64
    R = 5
    S = 5
    P = 32
    Q = 32

    M0 = 16
    C0 = 1
    Q0 = 4
    P0 = 4
    R0 = 1
    S0 = 1

    M1 = int(M/M0)
    C1 = int(C/C0)
    Q1 = int(Q/Q0)
    P1 = int(P/P0)

    nvdla_vars = [('m', M1), ('m', M0, True), ('c', C1), ('q', Q1), ('p', P1), ('c', C0), 
                ('s', S), ('r', R), ('q', Q0), ('p', P0), ('r', 1), ('s', 1)]
    mem_dividers = [0, 2, 10]

    nvdla_injection = Loop.loop(nvdla_vars, mem_dividers)
    dram_inj = 0
    inj_coord = (0, 2, 2)
    nvdla_injection.inject(inj_coord, 0)
    sites = nvdla_injection.get_sites()

    dataset = get_dataset()
    alexnet = get_alexnet()
    img = dataset[0]['images']
    inject_net = InjectModel(alexnet, 1)

    start_time = time.time()
    for i in range(1000):
        inject_net.run_hook(img, inj_coord, sites)
    print("--- %s seconds ---" % (time.time() - start_time))
    
def test_inject():
    M = 192
    C = 64
    R = 5
    S = 5
    P = 32
    Q = 32

    M0 = 16
    C0 = 1
    Q0 = 4
    P0 = 4

    M1 = int(M/M0)
    C1 = int(C/C0)
    Q1 = int(Q/Q0)
    P1 = int(P/P0)

    nvdla_vars = [('m', M1), ('m', M0, True), ('c', C1), ('q', Q1), ('p', P1), ('c', C0), 
                ('s', S), ('r', R), ('q', Q0), ('p', P0), ('r', 1), ('s', 1)]
    mem_dividers = [0, 2, 10]

    nvdla_injection = Loop.loop(nvdla_vars, mem_dividers)
    inj_coord = (0, 2, 2)
    nvdla_injection.inject(inj_coord, 0)
    sites = nvdla_injection.get_sites()
    sites = sites[0]
    num_sites = len(sites)

    dataset = get_dataset()
    alexnet = get_alexnet()
    clean_alexnet = get_alexnet()
    img = dataset[0]['images']
    conv_id = 1
    inject_net = InjectModel(alexnet, conv_id)

    clean_net = CleanModel(clean_alexnet)
    _, clean_output, zeros = run_clean(clean_net, img, conv_id)

    # scalene_profiler.start()
    inject_net.run_hook(img, inj_coord, sites)
    
    inject_output = inject_net.get_output(conv_id)

    num_diff = compare_outputs(clean_output, inject_output)
    print("\nNumber of Sites: " + str(num_sites))
    print("Number of output neurons different: " + str(num_diff))
    if num_diff == num_sites:
        print("TEST PASSED - same number of sites and output diffs")
    else:
        print("TEST FAILED")

RESNET18_MAX = [36.13761901855469, 10.887314796447754, 3.855881929397583, 9.794642448425293, 3.2933764457702637, 
                8.875794410705566, 4.089364528656006, 4.49278450012207, 5.494577884674072, 3.7413206100463867, 
                5.697751522064209, 3.7465898990631104, 1.6894534826278687, 4.060337543487549, 2.0673913955688477,
                3.928053855895996, 2.009352684020996, 2.0958352088928223, 3.4922521114349365, 3.5720021724700928, 39.165164947509766]
RESNET18_MIN = [-35.18002700805664, -16.452085494995117, -7.513538837432861, -11.784904479980469, -5.422745704650879, 
                -9.174695014953613, -3.548478841781616, -5.52264404296875, -5.273807048797607, -3.147308349609375, 
                -6.394559383392334, -4.2612624168396, -1.897513508796692, -5.157448768615723, -2.5855915546417236, 
                -5.034069538116455, -1.9327064752578735, -1.9150633811950684, -5.472661018371582, -0.9709599614143372, -13.11787223815918]

VIT_224_MAX = [21.38619613647461, 13.137690544128418, 5.223912715911865, 22.591272354125977, 14.627334594726562, 8.52755355834961, 6.0006914138793945, 9.890716552734375, 10.100648880004883, 7.496172904968262, 2.838361978530884, 9.791748046875, 4.218319892883301, 7.627469062805176, 2.1772141456604004, 7.022871971130371, 9.97861385345459, 8.753116607666016, 2.603743314743042, 10.7113676071167, 9.364392280578613, 8.279913902282715, 1.913900375366211, 12.028547286987305, 16.298311233520508, 7.50547981262207, 3.8653643131256104, 11.191254615783691, 24.671464920043945, 8.172269821166992, 5.880016803741455, 8.650896072387695, 16.48027801513672, 9.161516189575195, 8.323050498962402, 21.478208541870117, 12.032465934753418, 9.890218734741211, 5.357192039489746, 39.167720794677734, 15.495792388916016, 10.776861190795898, 3.434782028198242, 82.16219329833984, 47.833740234375, 10.250325202941895, 7.804947376251221, 12.901052474975586, 27.16724395751953, 18.20071792602539]
VIT_224_MIN = [-19.463239669799805, -15.500661849975586, -5.93727970123291, -24.368146896362305, -10.279200553894043, -9.283302307128906, -6.6186628341674805, -15.092547416687012, -9.99910831451416, -8.01309871673584, -2.0055863857269287, -13.931312561035156, -6.891602039337158, -7.821361541748047, -1.683425784111023, -13.337827682495117, -20.036043167114258, -8.76114559173584, -2.5503268241882324, -16.220672607421875, -29.580533981323242, -8.281740188598633, -3.102416753768921, -14.465797424316406, -37.66347122192383, -8.265819549560547, -4.104371070861816, -9.637489318847656, -29.86809730529785, -8.226944923400879, -3.0855355262756348, -11.563165664672852, -7.577116966247559, -8.028410911560059, -5.703629493713379, -21.38522720336914, -5.2278337478637695, -9.873750686645508, -4.976212024688721, -45.72148513793945, -7.363213062286377, -9.736888885498047, -10.837485313415527, -176.64419555664062, -21.725860595703125, -11.022520065307617, -35.40918731689453, -22.09223175048828, -11.808673858642578, -6.43269157409668]
        
def test_inject():
    sites = []

    # inj_coord = (2, 48, 37)
    inj_coord = (100, 38)
    d_type = "i"
    get_net = get_deit_tiny

    dataset = get_dataset()
    net = get_net()
    clean_net = get_net()
    img = dataset[0]['images']
    layer_id = 1
    
    num_layers, var_sizes, paddings, strides, FC_sizes = get_layer_info(get_net, img)
    
    inject_net = InjectModel(net, layer_id, d_type=d_type)
    inject_net.set_range(VIT_224_MAX, VIT_224_MIN)
    inject_net.set_FC_size(FC_sizes[layer_id])

    clean_net = CleanModel(clean_net)
    _, clean_output, zeros = run_clean(clean_net, img, layer_id)

    inject_net.run_hook(img, inj_coord, sites, mode="bit", bit=1)
    
    inject_output = inject_net.get_output(layer_id)
    print(inject_net.pre_value)
    print(inject_net.post_value)
    print(inject_net.get_maxmin())

    num_diff, ranges = compare_outputs_range(clean_output, inject_output)
    print("Number of output neurons different: " + str(num_diff))
    print("Ranges: " + str(ranges))


if __name__=="__main__":
    # run_thousand(get_alexnet)
    # run_thousand_injected_alexnet()
    test_inject()