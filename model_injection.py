from clean_model import CleanModel
from helpers import get_new_filename, check_inj_ind, check_inj_coord, compare_outputs
import torch
import torch.nn as nn
import random
import csv
from os.path import exists
from tqdm import trange
from pathlib import Path
from inject_model import InjectModel
from info_model import get_layer_info
from max_model import get_range
from typing import Any, List, Tuple
import itertools
import numpy as np

# class for an object that is used to inject into a model (for all the layers of the model)
class ModelInjection():
    # these are the data fields that are collected
    fields = ['Img Ind', 'Inj Ind', 'Level', 'BitInd', 'Site_ID', 'Top2Diff', 'CorrectClassConf', 'ClassifiedCorrect', 'Top5', 'Top Confs', 'Preval', 'Postval', 'NumSites', 'MaxMin', 'XEntropy','NumDiff', 'Zeros']
    d_types = ['i', 'w', 'o']
    
    # constructor
    def __init__(self, get_net: callable, dataset, net_name, arch_name, loops, d_type='i', 
                 verbose=False, overwrite=False, maxes=[], mins=[], file_addon='', debug=False, 
                 max_range=True, layers=[], top_dir="", batch_size=1, use_cpu=False, append=False):
        print("Constructing ModelInjection...")
        
        self.net = get_net()                    # given network
        # set device to run baseline model on
        self.set_device(use_cpu)
        
        self.debug = debug
        self.append = append
        self.max_range = max_range
        self.get_net = get_net                  # store function to get network
        self.dataset = dataset                  # dataset to get images from
        self.arch_name = arch_name              # name of the architecture (for file purposes)
        self.net_name = net_name                # name of the network used
        clean_net = get_net()                   # make a copy of the given net to use as inside wrapper for CleanModel
        self.clean_net = CleanModel(clean_net,  # clean network
                                    device=self.device)
        self.layer_sizes = []                   # list of sizes for each layer (m, p, s, r, etc.)
        self.paddings = []                      # list of padding sizes for each layer
        self.strides = []                       # list of stride lengths for each layer
        self.FC_sizes = []                      # 
        self.maxes = maxes                      # list of maxes for each layer (by default empty if already created)
        self.mins = mins                        # list of mins
        self.inject_layers = []                 # list of InjectModel objects (one per layer)
        self.num_layers = 0                     # number of layers
        self.filenames = {}                     # list of filenames for each layer
        self.log_file = ""                      # name of log file
        self.top_dir = ""
        self.file_addon = file_addon
        self.batch_size = batch_size            # batch size for inference

        self.loops = loops                      # list of loop objects for each layer
        self.num_mem_levels = 0                 # number of memory levels for the given arch
        self.set_mem_levels()                   # set the number of mem levels
        self.set_dtype(d_type)                  # data to inject into (weight, input, output?)
        
        self.overwrite = overwrite              # whether to overwrite the current output data file
        self.verbose = verbose                  # whether to be verbose and print stuff
        
        # find/set the top directory to use for output files
        self.set_top_dir(top_dir)
        # extract info from layers
        self.get_layer_info()
        # get layer objects
        self.get_inject_layers()
    
    def set_device(self, use_cpu=False):
        if not torch.cuda.is_available() or use_cpu:
            device = torch.device("cpu")
            print("Using device: CPU" )
        else:
            device = torch.device("cuda:0")
            print("Using device: " + torch.cuda.get_device_name(device))
        self.device = device
        self.net.to(self.device)

    def set_top_dir(self, top_dir) -> None:
        if not top_dir:
            top_dir = "data_results"
        top_dir += "/" + self.arch_name + "/" + self.net_name
        self.top_dir = top_dir
        
    def set_mem_levels(self) -> None:
        """Counts and sets the number of memory levels for the architecture.
        """
        
        # find an existing loop and count the number of mem_inds
        for loop in self.loops:
            if loop != None:
                self.num_mem_levels = len(loop.mem_inds)
                return
        
    def open_files(self, layers: List[int]=[]) -> None:
        """Opens all the filenames as set in the filenames param.

        Args:
            layers (List[int], optional): List of layers to open files for. Defaults to [].
        """
        
        # make sure filenames have been populated
        assert(self.filenames)
        for i in layers:
            filename = self.filenames[i]
            # if the data file doesn't exist - write the header of data names
            if not exists(filename) or self.overwrite:
                if not exists(filename):
                    print("Creating new file '" + filename + "'...")
                else:
                    print("Overwriting existing file'" + filename + "'...")
                    
                with open(filename, 'w', newline='') as csvfile: 
                    csvwriter = csv.writer(csvfile, delimiter=',') 
                    # write the headers into the csv file
                    fields = self.fields
                    csvwriter.writerow(fields)
        
    def get_filenames(self, file_addon: str, layers: List[int]=[]) -> None:
        """Sets the filenames param - giving each layer a directory to operate out of and a csv file to output data into.

        Args:
            file_addon (str): An addon to add onto the default naming convention of the csv files.
            layers (List[int], optional): List of layers to set the filenames for. Defaults to [].
        """        
        
        print("Getting filenames...")
        # if no layers given, use all layers
        if not layers:
            layers = range(self.num_layers)

        # loop through each layer and set filename for that layer (in appropriate dir)
        addon = -1
        temp_filenames = {}
        for i in layers:
            # create the nested directories
            dir = self.top_dir + "/layer" + str(i) + "/"
            p = Path(dir)
            p.mkdir(parents=True, exist_ok=True)

            # create filename and add to list
            filename = dir + "data_" + self.d_type_name
            # add the add_on to the default name if it exists
            if file_addon:
                filename += "_" + str(file_addon)
            temp_filenames[i] = filename
            
            # gets a new filename - as to not overwrite the other one
            if not self.overwrite and not self.append:
                filename, file_num = get_new_filename(filename)
                addon = max(file_num, addon)
            else: # if overwrite just set filename
                filename += ".csv"
                self.filenames[i] = filename
        
        # if no overwrite - change filenames to new ones
        if not self.overwrite and not self.append:
            # -1 means no addon
            if addon == -1:
                addon = ""
            
            # loop through layers and set filenames with addon to not overwrite
            for i in layers:
                # add name to list of filenames
                self.filenames[i] = temp_filenames[i] + str(addon) + ".csv"

    # sets the log file name
    def set_log_file(self) -> None:
        # open a log file
        p = Path(self.top_dir)
        p.mkdir(parents=True, exist_ok=True)
        log_filename = self.top_dir + "/log.txt"
        self.log_file = log_filename
        if not exists(self.log_file) or self.overwrite:
            open(self.log_file, 'w', newline='')

    # logs something to the log file
    def log(self, to_log: Any) -> None:
        if not self.log_file:
            self.set_log_file()
        with open(self.log_file, 'a', newline='') as f: 
            f.write(str(to_log) + "\n") 

    # gets InjectModel objects for each layer and their maxes (if not given)
    def get_inject_layers(self) -> None:     
        # if user doesn't pass in - get the maxes
        if not self.maxes and self.max_range:
            # get max values for each layer
            max_net = self.get_net()
            self.maxes, self.mins = get_range(max_net, self.dataset)
            self.log("Maxes: " + str(self.maxes))
            self.log("Mins: " + str(self.mins))

        print("Setting up InjectLayers...")
        for i in range(self.num_layers):
            # get the model and get the layer injection object
            net_inj = self.get_net()
            inject_layer = InjectModel(net_inj, i, d_type=self.d_type, device=self.device)
            # if this layer is FC, set its size
            if self.FC_sizes[i] != -1:
                inject_layer.set_FC_size(self.FC_sizes[i])
            # set the max for this layer
            if self.max_range:
                inject_layer.set_range(max_vals=self.maxes, min_vals=self.mins)
            else:
                inject_layer.set_range()
            # add to list of inject layers
            self.inject_layers.append(inject_layer)

    # sets the data type to inject into - checks one of i, w, o
    def set_dtype(self, d_type: str) -> None:
        assert(d_type in self.d_types)
        self.d_type = d_type
        if d_type == 'i':
            self.d_type_name = "inputs"
        elif d_type == 'w':
            self.d_type_name = "weights"
        elif d_type == 'o':
            self.d_type_name = "outputs"
        else:
            assert(False and "Invalid dtype")

    # returns the filename for the given layer (from layer_id).
    def get_filename(self, layer_id: int) -> str:    
        return self.filenames[layer_id]

    # get information for all layers (gets sizes, padding, stride, etc.)
    def get_layer_info(self) -> None:      
        print("Getting layer info...")
        num_layers, var_sizes, paddings, strides, FC_sizes = get_layer_info(self.get_net, self.dataset[0]['images'])
        self.num_layers = num_layers
        self.layer_sizes = var_sizes
        self.paddings = paddings
        self.strides = strides
        self.FC_sizes = FC_sizes
    
    def get_layer_nums(self) -> int:
        return self.num_layers

    # debugging function that checks whether 
    def check_sites(self, sites, layer_id: int, inj_ind) -> None:
        # order: m c s r q p h w
        layer_size = self.layer_sizes[layer_id]
        m, c, s, r, q, p, h, w = layer_size
        if self.d_type == 'i':
            # W = weight kernel size
            # S = stride
            # I = target index value
            # L = input size
            def get_input_window(W, S, I, L):
                o = 0
                i = 0
                ranges = []
                in_range = False
                while True:
                    # if out of bounds, stop
                    if i + W > L:
                        ranges.append(o)
                        break

                    # loop until you get in range
                    if I in range(i, i+W):
                        # if first time in range - append beginning of window
                        if not in_range:
                            in_range = True
                            ranges.append(o)
                    # if out of range
                    else:
                        # if you were in range - append end of window
                        if in_range:
                            ranges.append(o)
                            break
                    o += 1
                    i += S

                    assert(i < 1000)
                out_range = (ranges[0], ranges[1])
                return out_range

            strides = self.strides[layer_id]

            y_range = get_input_window(s, strides[0], inj_ind[1], h)
            x_range = get_input_window(r, strides[1], inj_ind[2], w)
            ranges = (range(m), y_range, x_range)

        elif self.d_type == 'w':
            m_i = inj_ind[0]
            ranges = (range(m_i, m_i+1), range(q), range(p))
        else: # for outputs, the sites are weights so
            m_i = inj_ind[0]
            ranges = (range(m_i, m_i+1), range(c), range(s), range(r))
        
        assert(len(sites[0]) == len(ranges))
        for site in sites:
            is_in = True
            for i in range(len(site)):
                if site[i] not in ranges[i]:
                    is_in = False
                    break
            if not is_in:
                print(site)
                print(ranges)
                assert(False)
            
    
    # this should be deprecated - this was before spatials
    # use to insert 'm' channels into output sites (since weren't processed before)
    # takes in a list of tuples, outputs a larger list of tuples
    def insert_channel(self, sites, channel=0, depth=16):
        out_sites = []
        for ind in sites:
            out_sites += [(i, ind[1], ind[2]) for i in range(channel, channel+depth)]
        return out_sites
    
    # set the name of the arch - used to change name
    def set_arch_name(self, new_name):
        self.arch_name = new_name
        if not exists(self.get_filename()) or self.overwrite:
            with open(self.get_filename(), 'w', newline='') as csvfile: 
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile, delimiter=',') 
                csvwriter.writerow(self.fields)
    
    # set the name of the model - used to change name
    def set_model_name(self, new_name):
        self.net_name = new_name
        if not exists(self.get_filename()) or self.overwrite:
            with open(self.get_filename(), 'w', newline='') as csvfile: 
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile, delimiter=',') 
                csvwriter.writerow(self.fields)
    
    # actually perform the injection - using the passed in inject_layer object for a single layer (given by layer_id)
    # for each of the images in img_inds,
    #   for each injection indices in inj_inds
    #       for each error site in error_sites (timed)
    #           inject into that error site using inject_layer
    
    # error_sites must be a 3-D list: [inj_ind][mem_level][sampled_timed_sites]
    def inject(self, img_inds, inj_inds, error_sites, inject_layer, layer_id, 
               mode="change_to", change_to=1000., bit=-1, debug_outputs=False):
        
        bit_val = bit
        count = 0
        # get if the layer is fully connected
        is_FC = inject_layer.get_is_FC()
        if is_FC:
            FC_size = inject_layer.get_FC_size()
            
        def get_nonzero_ind(inj_tup, start_from: int=1) -> int:
            # find first non-zero value after start_from
            for i in range(start_from, len(inj_tup)):
                if inj_tup[i] != 0:
                    return inj_tup[i]
            # if all zero, just return 0
            return 0
            
        # open the file
        self.open_files([layer_id])
        # loop through images
        for i in trange(0, len(img_inds), self.batch_size):
            end_batch = min(i + self.batch_size, len(img_inds))
            img_batch = img_inds[i:end_batch]
            # get image
            imgs = self.dataset[img_batch]['images']
            labels = self.dataset[img_batch]['labels']

            if debug_outputs:
                # clean_out - final output, clean_outputs - list of outputs of each layer, zeros - list of zeros for each layer
                clean_outs, clean_outputs, zeros = self.clean_net.run_clean(imgs, layer_id)
            
            # loop through inj locations
            assert(len(error_sites) == len(inj_inds))
            for ind in range(len(error_sites)):
                # get inj location and timed sites for all levels
                inj_ind = inj_inds[ind]
                # reform injection index for FC, based on input or weight
                if is_FC:
                    if self.d_type in ['i', 'o']:
                        if FC_size == 2:
                            inj_ind = (inj_ind[0],)
                        # TODO: not sure if this is generalizable beyond VIT
                        elif FC_size == 3:
                            inj_ind = (get_nonzero_ind(inj_ind, start_from=1), inj_ind[0])
                        else:
                            assert(False and "FC_size not supported")
                    elif self.d_type == 'w':
                        inj_ind = (inj_ind[0], inj_ind[1])
                    else:
                        assert(False and "d_type not supported")
                # get the sites for this injection index
                all_sites = error_sites[ind]
                
                # marks the site used for each image
                site_id = 0
            
                # loop through memory levels
                for inj_level in range(len(all_sites)):
                    # get timed sites for the level
                    level_sites = all_sites[inj_level]

                    # loop through timed sites for level
                    for t in range(len(level_sites)):
                        timed_site = level_sites[t]
                        if type(bit) is list:
                            bit_val = bit[count]
                        
                        num_sites = len(timed_site)
                        if self.d_type == 'o':
                            num_sites = 1
                        # if FC then change the output sites to have the right dims and order
                        if is_FC and self.d_type != 'o':
                            # TODO: also not sure about generalizability of this
                            if FC_size == 2:
                                timed_site = [(ts[0],) for ts in timed_site]
                            elif FC_size == 3:
                                timed_site = [(get_nonzero_ind(ts), ts[0]) for ts in timed_site]
                
                        # run the frontend PyTorch inference
                        inj_outs, pre_vals, post_vals = inject_layer.run_hook(imgs, inj_ind, timed_site, mode, change_to, bit_val)
                        maxmins = inject_layer.get_maxmin()
                        outputs = []
                        if debug_outputs:
                            injected_outputs = inject_layer.get_output(layer_id)
                            outputs = [clean_outputs, injected_outputs, clean_outs]
                        
                        # process outputs to output file
                        # NOTE: labels is returned as a list
                        process_outputs(inj_outs, img_batch, labels, inj_ind, inj_level, site_id, pre_vals, post_vals, 
                                        num_sites, layer_id, bit_val, maxmins, outs=outputs, zeros=zeros, filename=self.get_filename(layer_id))

                        # increment after changing injection location
                        count += 1
                        site_id += 1
                
    # given a clean network (net), some set of imgs (given by img_inds)
    # of the dataset - return the correct classification rates
    def get_baseline(self, img_inds):
        self.net.eval()
        
        correct = 0
        total = 0
        classifications = []
        # loop through each img ind provided
        for i in range(0, len(img_inds), self.batch_size):
            # get the slice of indices to batch together
            end_ind = min(i + self.batch_size, len(img_inds))
            inds = img_inds[i:end_ind]
            
            # run img(s) through network
            imgs = self.dataset[inds]['images']
            imgs = imgs.to(self.device)
            res = self.net(imgs)
            
            # get the top 5 classifications
            _, max_inds = torch.topk(res, 5, 1)
            max_inds = max_inds.to("cpu").tolist()
            classifications += max_inds
            
            # check if the top 1 confident matches the label (in csv file)
            labels = self.dataset[inds]['labels']
            for i in range(len(max_inds)):
                max_ind = max_inds[i]
                if max_ind[0] == labels[i]:
                    correct += 1
                total += 1

        return correct, total, classifications
                        
    def get_rand_imgs(self, num_imgs, sample_correct=True):
        # if classification of image doesn't matter - sample from any
        dataset_len = len(self.dataset)
        if not sample_correct:
            return random.sample(range(0, dataset_len), num_imgs)

        # samples to return
        samples = []
        # samples taken
        num_sampled = 0
        # set to sample from
        sample_from = set(range(dataset_len))
        
        # keep looping until you've gotten all samples or sampled everything
        while len(samples) < num_imgs:
            # if you've sampled all images, then print warning and stop
            if num_sampled == dataset_len:
                print("Not enough correct images in dataset to get sample, using " + str(len(samples)) + " instead")
                break
            # get a random image ind from dataset
            cand_img = random.sample(sample_from, 1)
            sample_from.remove(cand_img[0])
            # get the classification of the image
            correct, _, _ = self.get_baseline(cand_img)
            # if classified correct - add to samples
            if correct == 1:
                samples += cand_img
            num_sampled += 1
                
        return samples

    # performs a full injection over each layer of the provided model
    # num_imgs is number of images to sample over
    # mode is the bit injection mode
    # change_to is if change_to is activated
    # bit is for the bit to change - if is number, then will always inject into that one
    #   if is a range object, then is a range of bits to sample from
    def full_inject(self, num_imgs=100, mode="change_to", change_to=1000., bit=-1, img_inds=[], 
                    layers=[], debug=False, inj_sites=[], sample_correct=True, sites_method="", 
                    n_points=None, per_sample=None):
        print("Full injecting...")
        
        self.log("Starting new injection")
        self.get_filenames(self.file_addon, layers=layers)

        # get a sample of num_imgs images to use
        if not img_inds:
            print("Getting img inds...")
            img_inds = self.get_rand_imgs(num_imgs, sample_correct=sample_correct)
            self.log("Img inds (random):")
            self.log(img_inds)
        else:
            self.log("Img inds (given):")
            self.log(img_inds)
            print("Using given img inds...")

        if not sample_correct: 
            # get the baseline accuracy
            print("Getting baseline correct rate...")
            correct, total, classifications = self.get_baseline(img_inds)
            correct_rate = correct/total
            self.log(correct_rate)
            print("Correct rate: " + str(correct_rate))
        else:
            correct_rate = 1.0

        # if no user layers given - use all layers
        if not layers:
            print("No layers given, processing all layers...")
            layers = range(self.num_layers)
        
        # if user inj sites given - make sure given for all layers
        if inj_sites:
            assert(len(inj_sites) == len(layers))

        # loop through each layer
        count = 0
        for i in layers:
            # make sure given layers is valid
            assert(i >= 0 and i < self.num_layers)
            print("Performing injection into Layer " + str(i) + "...")
            # for each layer need to:
            # 1) get an InjectModel object for the layer and model
            # 2) set the max for the created object
            # 3) call get_rand for the layer's loop (passed in by user), giving injection sites
            # 4) call inject() - pass in (img_inds, inj_inds, sites, inject_layer)
            
            # if user provided sites - pass those to get_rand
            if inj_sites:
                injs = inj_sites[count]
            # else pass empty (will trigger random generation)
            else:
                injs = []

            # get the random index (pass in the loop object for this layer)
            if not sites_method or sites_method == "loop":
                sites, inj_inds, total_num = self.get_rand_loop(i, inj_inds=injs, per_sample=per_sample)
            elif sites_method == "region":
                sites, inj_inds, total_num = self.get_rand_region(i, inj_inds=injs, n_regions=8)
            elif sites_method == "random":
                sites, inj_inds, total_num = self.get_rand_sites(i, inj_inds=injs, per_sample=per_sample, n_points=n_points)
            else:
                assert(False)
            # if bit is passed as range, then sample random bits (one for each sample)
            if type(bit) is range:
                bit = self.get_rand_bits(bit, total_num*num_imgs)

            # get the inject_layer object for this layer
            inject_layer = self.inject_layers[i]
            # inject - will output into an out file
            self.inject(img_inds, inj_inds, sites, inject_layer, i, mode=mode, change_to=change_to, bit=bit, debug_outputs=debug)
            count += 1

        return correct_rate

    # return a list of num_samples in range given by bit_range
    def get_rand_bits(self, bit_range, num_samples):
        rand_bits = random.choices(bit_range, k=num_samples)
        return rand_bits

    # given a list of ranges, returns a list of num_injs randomly sampled indices
    # TODO: can change this to use numpy and be faster
    def get_rand_inds(self, lims, num_injs):
        # create sampled indices for each lim
        ind_sets = [random.choices(lim, k=num_injs) for lim in lims]

        # return a list of the indices (as list of tuples)
        inj_inds = []
        # loop through number of injections
        for i in range(num_injs):
            ind = []
            # for each variable sampled
            for j in range(len(lims)):
                # append the sampled index
                ind.append(ind_sets[j][i])
            # make tuple and append to list
            ind = tuple(ind)
            inj_inds.append(ind)

        return inj_inds
    
    def get_layer_limits(self, layer_ind: int):
        m, c, s, r, q, p, h, w = self.layer_sizes[layer_ind] # [m, c, s, r, q, p, h, w]
        stride = self.strides[layer_ind]
        # need to use transformed sizes instead of the layer_sizes
        # actually want to use original size and then transform into transformed internally
        if self.d_type == 'w':
            # use first four indices: m, c, s, r
            limits = [m, c, s, r]
            limits = [range(l) for l in limits]
        elif self.d_type == 'i':
            # this is to handle cutting off on the end - when the weight/stride leads to weird end behavior
            def reduce_by_10(limits):
                return [range(limit - limit//10) for limit in limits]
            
            # removes any indices that are not possible based on stride and weight size
            def check_stride_width(limit, stride, w):
                if stride <= w:
                    return limit
                
                new_limits = []
                for l in limit:
                    if check_inj_coord(l, stride, w):
                        new_limits.append(l)
                return new_limits
            
            limits = reduce_by_10((c, h, w))
            limits = [limits[0], check_stride_width(limits[1], stride[0], s), check_stride_width(limits[2], stride[1], r)]
        elif self.d_type == 'o':
            limits = [m, q, p]
            limits = [range(l) for l in limits]
        else:
            assert(False)
        
        return limits
            
    def get_rand_region(self, layer_ind, n_regions=2, inj_inds=[], num_injs=8):
        limits = self.get_layer_limits(layer_ind)
        m, c, s, r, q, p, h, w = self.layer_sizes[layer_ind]
        stride = self.strides[layer_ind]
        # get num_injs random indices
        if not inj_inds:
            self.log("Generated inj_inds for layer " + str(layer_ind))
            inj_inds = self.get_rand_inds(limits, num_injs)
        else:
            self.log("Using given inj_inds for layer " + str(layer_ind))
            # check that the injection indices passed in by the user are valid
            # only need to do this if injecting into inputs
            if self.d_type == 'i':
                for inj in inj_inds:
                    if not check_inj_ind(inj[1:], stride, (s, r)):
                        raise Exception("Invalid injection given for the stride and weight size of the layer.")
        
        # returns ranges of values dividing up the output into tiles
        def get_site_regions(d_type, window, n_regions):
            def get_iter_product(regions):
                return list(itertools.product(*regions))
            
            def range_split(r, n):
                return [n.tolist() for n in np.array_split(np.array(r), n)]
            
            # for input, only divide over output channels for now
            out_sites = []
            if d_type == 'i':
                channel_splits = range_split(window[0], n_regions)
                for cs in channel_splits:
                    out_sites.append(get_iter_product([cs, window[1], window[2]]))
            # for weights, divide into 2D tiles    
            elif d_type == 'w':
                q_splits = range_split(window[1], n_regions)
                p_splits = range_split(window[2], n_regions)
                for qs in q_splits:
                    for ps in p_splits:
                        out_sites.append(get_iter_product([window[0], qs, ps]))
            # for output, just return the window
            elif d_type == 'o':
                out_sites.append(get_iter_product(window))
            
            return out_sites
            
        # get the separate indices for each region
        all_sites = []
        total_num = 0
        for i in range(num_injs):
            inj_ind = inj_inds[i]
            window = self.get_inj_window(layer_ind, inj_ind)
            site_regions = get_site_regions(self.d_type, window, n_regions)
            all_sites.append([site_regions])
            total_num += len(site_regions)
            
        return all_sites, inj_inds, total_num
    
    def get_inj_window(self, layer_ind, inj_ind):
        inject_loop = self.loops[layer_ind]
        return inject_loop.get_original_window(inj_ind)
    
    def get_window_size(self, layer_ind, inj_ind=None):
        if inj_ind is None:
            limits = self.get_layer_limits(layer_ind)
            # limits is a list of ranges, pick middle value for each
            inj_ind = [l[len(l)//2] for l in limits]
        inject_loop = self.loops[layer_ind]
        window = inject_loop.get_original_window(inj_ind)
        size = 1
        for w in window:
            size *= len(w)
        return size
    
    # num_injs is the number of injection locations
    # per_sample is the number of samples to take from each injection location
    # # n_points is the number of points within the output window to choose per sample
    def get_rand_sites(self, layer_ind, num_injs=8, inj_inds=[], per_sample=100, n_points=2):
        if per_sample is None:
            per_sample = 100
        if n_points is None:
            n_points = 2
        if num_injs is None:
            num_injs = 8
        limits = self.get_layer_limits(layer_ind)
        m, c, s, r, q, p, h, w = self.layer_sizes[layer_ind]
        stride = self.strides[layer_ind]
        # get num_injs random indices
        if not inj_inds:
            self.log("Generated inj_inds for layer " + str(layer_ind))
            inj_inds = self.get_rand_inds(limits, num_injs)
        else:
            self.log("Using given inj_inds for layer " + str(layer_ind))
            # check that the injection indices passed in by the user are valid
            # only need to do this if injecting into inputs
            if self.d_type == 'i':
                for inj in inj_inds:
                    if not check_inj_ind(inj[1:], stride, (s, r)):
                        raise Exception("Invalid injection given for the stride and weight size of the layer.")
        
        all_sites = []
        total_num = 0
        # loop through number of injection locations
        for i in range(num_injs):
            inj_ind = inj_inds[i]
            # get the output window for this location
            window = self.get_inj_window(layer_ind, inj_ind)
            # get all the coordinates for that window 
            coords = list(itertools.product(*window))
            # pick per_sample random sets of coords
            inj_sites = []
            for j in range(per_sample):
                # randomly choose n_points from the window
                sites = random.sample(coords, n_points)
                inj_sites.append(sites)
                total_num += 1
            all_sites.append([inj_sites])
        
        return all_sites, inj_inds, total_num
        
                            
    # gets injection indices to use and samples a set of sites for each chosen index
    # must be called after get_layer_info
    # picks num_injs injection sites
    # for each injection site, picks 4 samples from each mem_level (if possible)
    # total injections per image = per_sample*num_injs*num_levels
    # so num_injs*per_sample*num_level samples
    def get_rand_loop(self, layer_ind, inj_inds=[], per_sample=4, num_injs=8):
        print("Get randing...")
        inject_loop = self.loops[layer_ind]
        limits = self.get_layer_limits(layer_ind)
        stride = self.strides[layer_ind]
        m, c, s, r, q, p, h, w = self.layer_sizes[layer_ind]

        # get num_injs random indices
        if not inj_inds:
            self.log("Generated inj_inds for layer " + str(layer_ind))
            inj_inds = self.get_rand_inds(limits, num_injs)
        else:
            self.log("Using given inj_inds for layer " + str(layer_ind))
            # check that the injection indices passed in by the user are valid
            # only need to do this if injecting into inputs
            if self.d_type == 'i':
                for inj in inj_inds:
                    if not check_inj_ind(inj[1:], stride, (s, r)):
                        raise Exception("Invalid injection given for the stride and weight size of the layer.")
        self.log(inj_inds)
        
        # collect sites for all three levels and all indices
        # i = inj locations, j = memory level, k = sites group
        all_sites = []
        for i in range(num_injs):
            mid_site = []
            for j in range(self.num_mem_levels):
                mid_site.append([])
            all_sites.append(mid_site)

        # total number of samples in all_sites
        total_num = 0
        
        # generate for num_injs indices
        for i in range(num_injs):
            # get the inj_ind
            inj_ind = inj_inds[i]
            # if injecting into input - add padding
            if self.d_type == 'i':
                inj_ind = list(inj_ind)
                inj_ind[1] += self.paddings[layer_ind][0]
                inj_ind[2] += self.paddings[layer_ind][1]
                inj_ind = tuple(inj_ind)
            
            if self.d_type != 'o':
                # get all timed sites
                inject_loop.inject_full(inj_ind)
                timed_sites = inject_loop.insert_spatial()
                # prune the sites to be within range
                timed_sites = inject_loop.prune_sites(timed_sites)
                # timed_sites[i][j][k]
                    # i = mem level
                    # j = discrete time groups (i.e. contains sets of sites created by considering time)
                    # k = possibilities within a time group
            
            # loop through the memory levels (given by a loop's mem_inds)
            for j in range(self.num_mem_levels):
                if self.d_type == 'o':
                    sites = [[inj_ind]]
                else:
                    # get this level's time groups
                    sites = timed_sites[j]

                # loop through list of time groups
                total_possible = 0
                timed_lens = []
                for k in range(len(sites)):
                    # increment total possible # sites
                    total_possible += len(sites[k])
                    # record the size of each timed group (# sites)
                    timed_lens.append(len(sites[k]))
                
                # get the number of samples to take
                # min in case the possibilities are less than the samples we want
                to_sample = min(total_possible, per_sample)
                # get timed samples - without replacement
                sample_inds = random.sample(range(0, total_possible), to_sample)

                # loop through sample indices
                samples = []
                for k in range(len(sample_inds)):
                    sample_ind = sample_inds[k]
                    total = 0
                    prev_total = 0
                    # find corresponding index from the offset
                    # loop through each time group
                    for l in range(len(timed_lens)):
                        # get size of timed group
                        timed_len = timed_lens[l]
                        # record previous total - increment total size of timed groups seen
                        prev_total = total
                        total += timed_len
                        # if the sample is less than the total - you passed your target
                        if sample_ind < total:
                            # index into this timed group - and get the site
                            samples.append(sites[l][sample_ind - prev_total])
                            break
                
                # loop through each time sample
                for sample in samples:
                    total_num += 1
                    all_sites[i][j].append(list(set(sample)))
                
        return all_sites, inj_inds, total_num
    
def get_topk(outs, labels, k=5):
    _, max_inds = torch.topk(outs, k)
    max_inds = max_inds.to('cpu').numpy()
    
    corrects = []
    for i in range(len(max_inds)):
        corrects.append(max_inds[i][0] == labels[i])
    
    percentage = torch.nn.functional.softmax(outs, dim=1) * 100
    percentage = percentage.to("cpu").numpy()
    top_confs = []
    correct_confs = []

    for i in range(len(max_inds)):
        top_confs.append(percentage[i][max_inds[i]].tolist())
        correct_confs.append(percentage[i][labels[i]])
    
    top2diffs = []
    for confs in top_confs:
        top2diffs.append(confs[0] - confs[1])

    return (max_inds, top_confs, correct_confs, top2diffs, corrects)


# outs = [clean layer outputs, injected layer outputs, final clean output]
def process_outputs(inj_outs, img_inds, labels, inj_ind, inj_level, site_id, pre_vals, post_vals, num_sites,
                    layer_id, bit, maxmins, outs=[], zeros=[], k=5, filename="", log_file="debug_log.txt"):
    
    num_outs = inj_outs.shape[0]
    assert(num_outs == len(img_inds))
    rows = []
    max_inds, top_confs, correct_confs, top2diffs, corrects = get_topk(inj_outs, labels, k)
    if outs:
        loss = nn.CrossEntropyLoss(reduction='none')
        xentropys = loss(inj_outs, outs[2]).tolist()
        
    for i in range(num_outs):
        row = [img_inds[i], 
               inj_ind, 
               inj_level, 
               bit, 
               site_id,
               top2diffs[i], 
               correct_confs[i], 
               corrects[i], 
               max_inds[i], 
               top_confs[i], 
               pre_vals[i], 
               post_vals[i], 
               num_sites, 
               (maxmins[0][i], maxmins[1][i])]

        if outs:
            row += [xentropys[i]]
            
            num_diff = compare_outputs(outs[0][i], outs[1][i])
            row += [num_diff]
        else:
            row += [None]
        
        if zeros:
            row += [((zeros[0][0][i], zeros[0][1]), 
                     (zeros[1][0][i], zeros[1][1]), 
                     (zeros[2][0], zeros[2][1]))]
        else:
            row += [None]
        
        rows.append(row)
    
    # write out data to csv file
    if filename:
        with open(filename, 'a', newline='') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile, delimiter=',')

            # writing the data rows 
            csvwriter.writerows(rows)
