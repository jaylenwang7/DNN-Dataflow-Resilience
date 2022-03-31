from clean_model import CleanModel, run_clean
from pathlib import Path
from helpers import *
import torch
import random
import csv
from os.path import exists
from tqdm import trange
from pathlib import Path
from inject_model import InjectConvLayer
from info_model import *
from max_model import *

# class for an object that is used to inject into a model (for all the conv layers of the model)
class model_injection():
    # these are the data fields that are collected
    fields = ['Img Ind', 'Inj Ind', 'Level', 'Top2Diff', 'CorrectClassConf', 'ClassifiedCorrect', 'Top 10', 'Top Confs', 'Preval', 'Postval', 'NumSites']
    d_types = ['i', 'w', 'o']
    
    # constructor
    def __init__(self, get_net: callable, dataset, net_name, arch_name, loops, d_type='i', 
                 verbose=False, overwrite=False, maxes=[], file_addon='', debug=False):
        self.debug = debug
        print("Constructing model_injection...")
        self.get_net = get_net                  # store function to get network
        self.dataset = dataset                  # dataset to get images from
        self.arch_name = arch_name              # name of the architecture (for file purposes)
        self.net_name = net_name                # name of the network used
        self.net = get_net()                    # given network
        clean_net = get_net()                   # make a copy of the given net to use as inside wrapper for CleanModel
        self.clean_net = CleanModel(clean_net)    # clean network
        self.conv_sizes = []                    # list of sizes for each conv layer (m, p, s, r, etc.)
        self.paddings = []                      # list of padding sizes for each conv layer
        self.strides = []                       # list of stride lengths for each conv layer
        self.maxes = maxes                      # list of maxes for each layer (by default empty if already created)
        self.inject_convs = []                  # list of InjectConvLayer objects (one per layer)
        self.num_layers = 0                     # number of layers
        self.filenames = []                     # list of filenames for each layer
        self.log_file = ''                      # name of log file
        self.top_dir = ''

        self.loops = loops                      # list of loop objects for each layer
        self.set_dtype(d_type)                  # data to inject into (weight, input, output?)
        
        self.overwrite = overwrite              # whether to overwrite the current output data file
        self.verbose = verbose                  # whether to be verbose and print stuff

        # extract info from conv layers
        self.get_conv_info()
        # get conv layer objects
        self.get_inject_convs()
        # set and get filenames
        self.get_filenames(file_addon)

    # open and collect the filenames for each layer
    def get_filenames(self, file_addon):
        print("Getting filenames...")
        top_dir = "data_results/" + self.arch_name + "/" + self.net_name
        self.top_dir = top_dir
        # loop through each layer and set filename for that layer (in appropriate dir)
        for i in range(self.num_layers):
            # create the nested directories
            dir = top_dir + "/conv" + str(i) + "/"
            p = Path(dir)
            p.mkdir(parents=True, exist_ok=True)

            # create filename and add to list
            filename = dir + "data_" + self.d_type_name
            # add the add_on to the default name if it exists
            if file_addon:
                filename += "_" + str(file_addon)
            
            # gets a new filename - as to not overwrite the other one
            if not self.overwrite:
                filename = get_new_filename(filename)
            else:
                filename += ".csv"
            
            # add name to list of filenames
            self.filenames.append(filename)

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
                    if self.debug:
                        fields += ["NumDiff"]
                    csvwriter.writerow(fields)

        # open a log file
        log_filename = self.top_dir + "/" + "log.txt"
        self.log_file = log_filename
        if not exists(self.log_file) or self.overwrite:
            open(self.log_file, 'w', newline='')

    def log(self, to_log):
        with open(self.log_file, 'a', newline='') as f: 
            f.write(str(to_log) + "\n") 

    # get InjectConvLayer objects and their maxes (if not given)
    def get_inject_convs(self):
        # if user doesn't pass in - get the maxes
        if not self.maxes:
            # get max values for each layer
            max_net = self.get_net()
            self.maxes = get_max(max_net, self.dataset)

        for i in range(self.num_layers):
            # get the model and get the conv layer injection object
            net_inj = self.get_net() # TODO: need to change this to handle any model
            inject_conv = InjectConvLayer(net_inj, i, inj_loc=self.d_type)
            # set the max for this layer
            inject_conv.set_max(self.maxes[i])
            # add to list of inject convs
            self.inject_convs.append(inject_conv)


    # sets the data type to inject into - checks one of i, w, o
    def set_dtype(self, d_type):
        assert(d_type in self.d_types)
        self.d_type = d_type
        if d_type == 'i':
            self.d_type_name = 'inputs'
        elif d_type == 'w':
            self.d_type_name = 'weights'

    # returns the current path to the filename - constructed from class parameters
    def get_filename(self, conv_id):
        return self.filenames[conv_id]

    # get information for all conv layers (gets sizes, padding, stride, etc.)
    def get_conv_info(self):
        print("Getting conv info...")
        num_layers, var_sizes, paddings, strides = get_conv_info(self.get_net, self.dataset[0]['image'])
        self.num_layers = num_layers
        self.conv_sizes = var_sizes
        self.paddings = paddings
        self.strides = strides

    def check_sites(self, sites, conv_id, inj_ind):
        conv_size = self.conv_sizes[conv_id]
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

            strides = self.strides[conv_id]
            m = conv_size[0]
            h = conv_size[6]
            w = conv_size[7]
            s = conv_size[2]
            r = conv_size[3]

            y_range = get_input_window(s, strides[0], inj_ind[1], h)
            x_range = get_input_window(r, strides[1], inj_ind[2], w)
            ranges = (range(m), y_range, x_range)
            for site in sites:
                is_in = site[0] in ranges[0] and site[1] in ranges[1] and site[2] in ranges[2] 
                if not is_in:
                    print(site)
                    print(ranges)
                    assert(False)

        elif self.d_type == 'w':
            m = inj_ind[0]
            q = conv_size[4]
            p = conv_size[5]
            ranges = (range(m, m+1), range(q), range(p))
            for site in sites:
                is_in = site[0] in ranges[0] and site[1] in ranges[1] and site[2] in ranges[2] 
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
    
    # actually perform the injection - using the passed in inject_conv object for a single conv_layer (given by conv_id)
    # for each of the images in img_inds,
    #   for each injection indices in inj_inds
    #       for each error site in error_sites (timed)
    #           inject into that error site using inject_conv
    def inject(self, img_inds, inj_inds, error_sites, inject_conv, conv_id, 
               mode="change_to", change_to=1000., bit=-1, debug_outputs=False):
        bit_val = bit
        count = 0
        # loop through images
        for i in trange(len(img_inds)):
            img_ind = img_inds[i]
            # self.log("Image: " + str(img_ind))
            # get image
            img = torch.unsqueeze(self.dataset[img_ind]['image'], 0)

            if debug_outputs:
                clean_outputs, zeros = run_clean(self.clean_net, img, conv_id)
            
            # loop through inj locations
            for ind in range(len(inj_inds)):
                # get inj location and timed sites for all levels
                inj_ind = inj_inds[ind]
                # self.log("inj_ind: " + str(inj_ind))
                timed_sites = error_sites[ind]
            
                # loop through memory levels
                for inj_level in range(len(timed_sites)):
                    # get timed sites for the level
                    level_sites = timed_sites[inj_level]

                    # loop through timed sites for level
                    for t in range(len(level_sites)):
                        timed_site = level_sites[t]
                        if type(bit) is list:
                            bit_val = bit[count]
                            count += 1
                        
                        # num_sites = len(timed_site)
                        set_sites = set(timed_site)
                        timed_site = list(set_sites)
                        num_sites = len(timed_site)
                        # self.log("timed_site: " + str(timed_site))
                        # run the frontend PyTorch inference
                        inj_out, pre_val, post_val = inject_conv.run_hook(img, inj_ind, timed_site, mode, change_to, bit_val)
                        outputs = []
                        if debug_outputs:
                            injected_outputs = inject_conv.get_output(conv_id)
                            outputs = [clean_outputs, injected_outputs]
                        # process outputs to output file
                        process_outputs(inj_out, img_ind, self.dataset[img_ind]['label'], inj_ind, inj_level, pre_val, post_val, 
                                        num_sites, conv_id, outs=outputs, filename=self.get_filename(conv_id))
                        
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
            if num_sampled == dataset_len:
                print("Not enough correct images in dataset to get sample, using " + str(len(samples)) + " instead")
                break
            # get a random image ind from dataset
            cand_img = random.sample(sample_from, 1)
            sample_from.remove(cand_img[0])
            # get the classification of the image
            correct, _, _ = get_baseline(self.net, cand_img, self.dataset)
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
                    layers=[], debug=False, inj_sites=[], sample_correct=True):
        print("Full injecting...")
        
        self.log("Starting new injection")

        # get a sample of 100 images to use
        if not img_inds:
            print("Getting img inds...")
            img_inds = self.get_rand_imgs(num_imgs, sample_correct=sample_correct)
            self.log("Img inds (random):")
            self.log(img_inds)
        else:
            self.log("Img inds (given):")
            self.log(img_inds)
            print("Using given img inds...")

        # get the baseline accuracy
        print("Getting baseline...")
        correct, total, classifications = get_baseline(self.net, img_inds, self.dataset)
        correct_rate = correct/total

        # if no user layers given - use all layers
        if not layers:
            print("No layers given, processing all layers...")
            layers = range(self.num_layers)
        
        # if user inj sites given - make sure given for all layers
        # TODO: do this on a per layer basis
        if inj_sites:
            assert(len(inj_sites) == self.num_layers)

        # loop through each layer
        for i in layers:
            # make sure given layers is valid
            assert(i >= 0 and i < self.num_layers)
            # for each layer need to:
            # 1) get an InjectConvLayer object for the layer and model
            # 2) set the max for the created object
            # 3) call get_rand for the layer's loop (passed in by user), giving injection sites
            # 4) call inject() - pass in (img_inds, inj_inds, sites, inject_conv)
            
            # if user provided sites - pass those to get_rand
            if inj_sites:
                injs = inj_sites[i]
            # else pass empty (will trigger random generation)
            else:
                injs = []

            # get the random index (pass in the loop object for this layer)
            sites, inj_inds, total_num = self.get_rand(i, inj_inds=injs)
            # if bit is passed as range, then sample random bits (one for each sample)
            if type(bit) is range:
                bit = self.get_rand_bits(bit, total_num)

            # get the inject_conv object for this layer
            inject_conv = self.inject_convs[i]
            # inject - will output into an out file
            self.inject(img_inds, inj_inds, sites, inject_conv, i, mode=mode, change_to=change_to, bit=bit, debug_outputs=debug)

        # self.write_correct(correct_rate)
        self.log(correct_rate)
        print("correct rate: " + str(correct_rate))
        return correct_rate

    # return a list of num_samples in range given by bit_range
    def get_rand_bits(self, bit_range, num_samples):
        rand_bits = random.choices(bit_range, k=num_samples)
        return rand_bits

    # given a list of ranges, returns a list of num_injs randomly sampled indices
    # TODO: can change this to use numpy and be faster
    def get_rand_inds(self, lims, num_injs):
        # create sampled indices for each lim
        ind_sets = [random.choices(range(lim), k=num_injs) for lim in lims]

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
                            
    # gets injection indices to use and samples a set of sites for each chosen index
    # must be called after get_conv_info
    # picks num_injs injection sites
    # for each injection site, picks 4 samples from each mem_level (if possible)
    # total injections per image = per_sample*num_injs*num_levels
    # so num_injs*per_sample*num_level samples
    def get_rand(self, layer_ind, inj_inds=[], per_sample=4, num_injs=8):
        print("Get randing...")
        inject_loop = self.loops[layer_ind]
        c_info = self.conv_sizes[layer_ind] # [m, c, s, r, q, p, h, w]
        # need to use transformed sizes instead of the conv_sizes
        # actually want to use original size and then transform into transformed internally
        if self.d_type == 'w':
            #         m          c          s          r
            limits = (c_info[0], c_info[1], c_info[2], c_info[3])
        elif self.d_type == 'i':
            #         c          h          w
            def reduce_by_10(limits):
                new_limits = list(limits)
                for i in range(len(limits)):
                    new_limits[i] -= new_limits[i]//10
                return tuple(new_limits)

            limits = reduce_by_10((c_info[1], c_info[6], c_info[7]))
        else:
            assert(False)

        # get num_injs random indices
        if not inj_inds:
            self.log("Generated inj_inds for layer " + str(layer_ind))
            inj_inds = self.get_rand_inds(limits, num_injs)
        else:
            self.log("Using given inj_inds for layer " + str(layer_ind))
        self.log(inj_inds)
        
        # collect sites for all three levels and all 10 indices
        ALL_sites = []
        for i in range(num_injs):
            mid_site = []
            for j in range(3):
                mid_site.append([])
            ALL_sites.append(mid_site)

        # total number of samples in ALL_sites
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
            for j in range(len(self.loops[0].mem_inds)):
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
                    ALL_sites[i][j].append(sample)
                
        return ALL_sites, inj_inds, total_num
    
def print_topk(out, correct_class, k=5):
    _, max_inds = torch.topk(out, k, 1)
    max_inds = torch.squeeze(max_inds)
    max_inds = max_inds.numpy()
    
    classified_correct = True
    if max_inds[0] != correct_class:
        classified_correct = False
    
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    confs = []
    for i in range(max_inds.shape[0]):
        index = max_inds[i]
        conf = percentage[index].item()
        confs.append(conf)

    correct_conf = percentage[correct_class].item()
    top2diff = confs[0] - confs[1]
    return (max_inds, confs, correct_conf, top2diff, classified_correct)

    
def process_outputs(inj_out, img_ind, correct_class, inj_ind, inj_level, pre_val, post_val, num_sites,
                    conv_id, outs=[], k=5, filename="", log_file="debug_log.txt"):
    
    max_inds, confs, correct_conf, top2diff, classified_correct = print_topk(inj_out, correct_class, k)

    row = [img_ind, inj_ind, inj_level, top2diff, correct_conf, classified_correct, max_inds, confs, pre_val, post_val, num_sites]

    if outs:
        num_diff = compare_outputs(outs[0], outs[1])
        row += [num_diff]
        # with open(log_file, 'a', newline='') as f: 
        #     f.write("num_diff = " + str(num_diff) + ", num_sites = " + str(num_sites) + "\n")
    
    # row = [img_ind, inj_ind, inj_level, top2diff, correct_conf, classified_correct, max_inds, confs, pre_val, post_val, num_sites]
    
    # write out data to csv file
    if filename:
        with open(filename, 'a', newline='') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile, delimiter=',')

            # writing the data rows 
            csvwriter.writerow(row)