import torch
import time
from helpers import *
from dataset import *
import loop as Loop
from inject_model import InjectConvLayer
from clean_model import CleanModel, run_clean

def run_thousand():
    dataset = get_dataset()
    alexnet = get_alexnet()
    img = dataset[0]['image']
    img = torch.unsqueeze(img, 0)

    start_time = time.time()
    for i in range(1000):
        alexnet(img)
    print("--- %s seconds ---" % (time.time() - start_time))
    pass

def run_thousand_injected():
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
    img = dataset[0]['image']
    img = torch.unsqueeze(img, 0)
    inject_net = InjectConvLayer(alexnet, 1)

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
    img = dataset[0]['image']
    img = torch.unsqueeze(img, 0)
    conv_id = 1
    inject_net = InjectConvLayer(alexnet, conv_id)

    clean_net = CleanModel(clean_alexnet)
    clean_output, zeros = run_clean(clean_net, img, conv_id)

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

# run_thousand()
# run_thousand_injected()
test_inject()