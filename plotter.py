from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from IPython.display import display
from helpers import *
import pandas as pd
import numpy as np
import os
import csv
import seaborn as sn

PLOT_COLOR = 'black'
SITES_COLOR = 'tab:blue'
TITLE_SIZE = 18
TICK_SIZE = 12
AXES_SIZE = 14
MARKER_SIZE = 8

# object used to plot data
class Plotter():
    
    def __init__(self, arch_name, net_name, max_mins, layers=[], d_type='i', add_on='', overwrite=True):
        self.arch_name = arch_name
        self.layers = layers
        self.net_name = net_name
        self.filenames = []
        self.d_type = d_type
        self.add_on = add_on
        self.overwrite = overwrite
        
        assert(d_type in ['i', 'w', 'o'])
        if d_type == 'i':
            self.d_type_name = 'inputs'
        elif d_type == 'w':
            self.d_type_name = 'weights'
        else:
            self.d_type_name = 'outputs'
            
        if not self.layers:
            self.get_layers()
        else:
            self.layers.sort()
        
        self.extracted = False
            
        self.maxes = []
        self.mins = []
        self.error = []
        self.xentropy = []
        self.numsites = []
        self.siteratio = []
        self.sparsity = []
            
        self.set_filenames(add_on)
        self.extract_data(max_mins)
        
    def set_overwrite(self, overwrite):
        self.overwrite = overwrite
        
    def get_layers(self):
        layers = [os.path.basename(os.path.normpath(f.path)) for f in os.scandir("data_results/" + self.arch_name + "/" + self.net_name + "/") if f.is_dir()]
        layers = [get_str_num(layer) for layer in layers]
        layers = [layer for layer in layers if layer != None]
        layers.sort()
        self.layers = layers

    def set_filenames(self, add_on=''):
        top_dir = "data_results/" + self.arch_name + "/" + self.net_name
        self.top_dir = top_dir
        # get the filenames for each layer
        for i in self.layers:
            # create the nested directories
            dir = top_dir + "/layer" + str(i) + "/"
            filename = dir + "data_" + self.d_type_name
            if add_on:
                filename += add_on
            filename += ".csv"
            self.filenames.append(filename)

        # create directory to store the plot images
        data_dir = "data_results/stats/"
        p = Path(data_dir)
        p.mkdir(parents=True, exist_ok=True)
        stats_filename = data_dir + "stats_" + self.d_type_name + ".csv"
        self.stats_file = stats_filename
        if not exists(self.stats_file):
            print("Opening new stats file at: " + self.stats_file)
            with open(stats_filename, 'w', newline='') as csvfile: 
                csvwriter = csv.writer(csvfile, delimiter=',') 
                # write the headers into the csv file
                csvwriter.writerow(self.fields)
        
        img_dir = top_dir + "/plots/"
        self.img_dir = img_dir
        p = Path(img_dir)
        p.mkdir(parents=True, exist_ok=True)
        
        corr_dir = img_dir + "corrs/"
        self.corr_dir = corr_dir
        p = Path(corr_dir)
        p.mkdir(parents=True, exist_ok=True)
        

    # manipulate data along axis_title on the passed in df
    def agg_data(self, df, axis_title):
        # group by user input
        group = df.groupby([axis_title])
        
        # get the aggregate statistics wanted (mean, stddev)
        data2collect = ["CorrectClassConf", "XEntropy", "NumSites"]
        data_mean = group[data2collect].mean()
        data_mean = data_mean.add_suffix("_mean")
        data_std = group[data2collect].std()
        data_std = data_std.add_suffix("_std")
        data = data_mean.join(data_std)
        
        # get the number of samples and error rate
        num_samples = group.size()
        error_rate = group["ClassifiedCorrect"].sum() / num_samples
        error_rate.name = "Error Rate"
        num_samples.name = "Num Samples"
        
        data = data.join(error_rate)
        data = data.join(num_samples)
        
        return data
    
    def get_zero_data(self, df):
        # get the zero rate
        data_zeros = df["Zeros"]
        ndata = 3
        zeros = [0]*ndata
        tots = [0]*ndata
        # loop through each data point in the group data
        for z in data_zeros:
            # get the entire tuple
            zero_tuple = eval(z)
            # loop through each zero tuple
            for i in range(ndata):
                # sum up number of zeros and total number of els
                zeros[i] += zero_tuple[i][0]
                tots[i] += zero_tuple[i][1]

        # aggregate the data
        return zeros, tots, len(data_zeros)
    
    def collect_zero_data(self, input_only=False, nonzeros=True):
        if input_only:
            out_data = []
            out_tots = []
        else:
            out_data = [[] for i in range(3)]
            out_tots = [[] for i in range(3)]
        out_nums = []
        
        # for each layer
        for i in range(len(self.layers)):
            # read out the data - comes in form (output, input, weight) for each as totals
            df = pd.read_csv(self.filenames[i])
            zeros, tots, num = self.get_zero_data(df)
            
            # append the total number of samples taken (num is just a single int)
            out_nums.append(num)
            
            # this is the ratio of values that are zero, so zero_ratio is len=3
            zero_ratio = []
            for i in range(len(zeros)):
                zero_ratio.append(zeros[i]/tots[i])
            
            # if you want portion that are nonzero then get that
            if not nonzeros:
                zero_ratio = [1.0 - z for z in zero_ratio]
            
            # place data in out lists
            if input_only:
                out_data.append(zero_ratio[1])
                out_tots.append(tots[1])
            else: 
                for i in range(3):
                    out_data[i].append(zero_ratio[i])
                    out_tots[i].append(tots[i])
            
        return out_data, out_tots, out_nums

    # collect necessary data along an axis of the collected data
    # given by axis_title - for the layer layer_id
    def collect_layer_data(self, layer_id, axis_title):
        df = pd.read_csv(self.filenames[layer_id])
        return self.agg_data(df, axis_title), len(df)
    
    # collect data for all layers - grouped by the axis_title
    def agg_layer_data(self, axis_title):
        # list of all dfs, one for each layer file
        all_df = []
        # loop through all layer files and collect dfs
        for i in range(len(self.filenames)):
            df = pd.read_csv(self.filenames[i])
            all_df.append(df)
        # concat all the layer dfs and aggregate it
        final_df = pd.concat(all_df, ignore_index=True)
        return self.agg_data(final_df, axis_title)
    
    def plot_zeros(self):
        # plot on x axis the layers and on y axis the sparsity
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        zero_data = self.collect_zero_data()
        # plot for output, input, weight
        for i in range(len(zero_data)):
            ax = plt.plot(self.layers, zero_data[i])
            
        fig.legend(labels=["Outputs", "Inputs", "Weights"],
                   loc="right")
        
        img_name, _ = get_new_filename(self.img_dir + "zeros", "png")
        plt.savefig(img_name)
        

    # plot the aggregated results of all the layers on the same plot
    # compared to the baseline
    def plot(self, level_names=[], agg_layers=True, just_data=False):
        
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        axes1 = axes.twinx()
        labels = []
        avg_sites = []
        
        if not agg_layers:
            # loop through layers
            for i in range(len(self.layers)):
                # plot the error rate
                axes.plot(self.avg_errors, style='o', ms=10, logx=False)

                # plot the given labels for the x values (i.e. DRAM, etc.)
                tick_list = [i for i in range(len(level_names))]
                axes.set_xticks(tick_list)
                axes.set_xticklabels(level_names, fontsize=TICK_SIZE)
                
                labels.append("Layer " + str(self.layers[i]))
        else:
            # plot the error rate
            axes.plot(self.avg_errors, marker='o', markersize=MARKER_SIZE, color=PLOT_COLOR)
            axes.set_ylim(bottom=0.0)

            axes1.plot(self.avg_sites, linestyle="--", marker='o', markersize=MARKER_SIZE, color=SITES_COLOR)
            axes1.set_ylabel("Avg. number of reuse sites", color=SITES_COLOR, fontsize=AXES_SIZE)
            axes1.tick_params(axis='y', labelcolor=SITES_COLOR)
            
            # plot the given labels for the x values (i.e. DRAM, etc.)
            tick_list = [i for i in range(len(level_names))]
            axes.set_xticks(tick_list)
            axes.set_xticklabels(level_names, fontsize=TICK_SIZE)
        
        fig.legend(labels=labels,
                   loc="right")

        plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.4, 
                            hspace=0.4)

        
        axes.set_xlabel("Memory Level (larger \u2192 smaller)", fontsize=AXES_SIZE)
        title = fancy_names[self.net_name] + " on " + fancy_names[self.arch_name] + self.d_type_name
        axes.set_ylabel("Avg. Top-1 Error", fontsize=AXES_SIZE)
        title += " Top-1 Error"
        
        axes.set_title(title, fontsize=TITLE_SIZE)

        # don't overwrite older image
        target_name = self.img_dir + "levels_" + self.d_type + self.add_on
        plt.savefig(get_name(target_name, overwrite=self.overwrite))
        plt.close('all')
    
    # fields used for stats files
    fields = ["Model", "Arch", "Err0", "Err1", "Err2", "Sites0", "Sites1", "Sites2", "Rat0", "Rat1", "Rat2", "Within", "Outside", "1to0", "0to1", "AvgPreval", "AvgPostval", "AvgDiff", "NonZeroRate", "ZeroRate", "NumNonZero", "NumZero", "Bit1", "Bit2", "Bit3", "Bit4", "Bit5", "Bit6", "Bit7", "Bit8", "NumSamples"]
    def collect_stats(self, thresh=2.0):
        all_dfs = []
        nsamples = 0
        for i in range(len(self.layers)):
            df = pd.read_csv(self.filenames[i])
            nsamples += len(df)
            if "BitInd" not in df:
                print(self.filenames[i])
                assert(False)
            all_dfs.append(df)
        df = pd.concat(all_dfs)
        
        df["Thresh"] = df["Preval"].apply(lambda x: x >= thresh or x <= -thresh)
        df["0to1"] = df.apply(lambda x: x["Preval"] <= x["Postval"], axis=1)
        df["Diff"] = df.apply(lambda x: abs(max(min(x["Postval"], eval(x["MaxMin"])[0]), eval(x["MaxMin"])[1]) - x["Preval"]), axis=1)
        df["AbsPreval"] = df["Preval"].apply(lambda x: abs(x))
        df["AbsPostval"] = df.apply(lambda x: abs(max(min(x["Postval"], eval(x["MaxMin"])[0]), eval(x["MaxMin"])[1])), axis=1)
        df["ZeroRate"] = df["Preval"].apply(lambda x: x==0)
        
        avg_vals = [df["AbsPreval"].mean(), df["AbsPostval"].mean(), df["Diff"].mean()]
        
        groups = df.groupby("Thresh")
        num_samples = groups.size()
        thresh_error = (groups["ClassifiedCorrect"].sum() / num_samples).tolist()
        
        groups = df.groupby("0to1")
        num_samples = groups.size()
        change_error = (groups["ClassifiedCorrect"].sum() / num_samples).tolist()
        
        groups = df.groupby("BitInd")
        num_samples = groups.size()
        bit_error = (groups["ClassifiedCorrect"].sum() / num_samples).tolist()
        
        groups = df.groupby("ZeroRate")
        zero_samples = groups.size()
        zero_error = (groups["ClassifiedCorrect"].sum() / zero_samples).tolist()
        
        def pad_list(lst, to_len=3):
            lst += [None] * (to_len-len(lst))
            return lst
        
        row = [fancy_names[self.net_name], fancy_names[self.arch_name]] + pad_list(self.avg_errors) + pad_list(self.avg_sites) + pad_list(self.avg_rats) + pad_list(thresh_error, to_len=2) + change_error + avg_vals + pad_list(zero_error, to_len=2) + pad_list(zero_samples.to_list(), to_len=2) + bit_error + [nsamples]
        
        with open(self.stats_file) as inf:
            reader = csv.reader(inf.readlines())
        
        with open(self.stats_file, 'w', newline='') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile, delimiter=',')
            print("Writing " + self.net_name + " on " + self.arch_name + " stats to: " + str(self.stats_file))
            contained = False
            changed = False
            first = True
            for line in reader:
                if first:
                    for i in range(len(line)):
                        if line[i] != self.fields[i]:
                            print("Updating fields")
                            csvwriter.writerow(self.fields)
                            changed = True
                            break
                    if not changed:
                        csvwriter.writerow(line)
                    first = False
                else:  
                    if line[0] == self.net_name and line[1] == self.arch_name:
                        print("Overwriting old stats")
                        csvwriter.writerow(row)
                        contained = True
                        break
                    else:
                        csvwriter.writerow(line)
            csvwriter.writerows(reader)
            
            if not contained:
                csvwriter.writerow(row)
                
    def extract_data(self, maxes_mins):
        # get sparsity data - comes out as list of list of three values (output, input, weight)
        # only really concerned with input - also all levels are the same
        # NOTE: these numbers are aggregate not averaged for each data point
        zeros, tots, nums = self.collect_zero_data(input_only=False, nonzeros=False)
        zeros = zeros[1]
        
        new_tots = []
        for i in range(len(self.layers)):
            new_tots.append(tots[0][i]/
                            nums[i])
        tots = new_tots
        
        # get the number of levels
        level_data, num = self.collect_layer_data(0, "Level")
        nlevels = len(level_data['Error Rate'].tolist())
        self.nlevels = nlevels
        
        # loop through each level and get the data for each level
        all_error = [[] for i in range(nlevels)]
        all_xentropy = [[] for i in range(nlevels)]
        all_numsites = [[] for i in range(nlevels)]
        num_samples = []
        for i in range(len(self.layers)):
            # get a table with the data for each level
            level_data, num = self.collect_layer_data(i, "Level")
            num_samples.append(num)
            
            # extract a column of the data and turn into a list, with data for each level
            error_data = level_data['Error Rate'].tolist()
            xentropy_data = level_data['XEntropy_mean'].tolist()
            numsites_data = level_data['NumSites_mean'].tolist()
            
            # transfer the data in the list to an aggregate list
            for j in range(nlevels):
                all_error[j].append(error_data[j])
                all_xentropy[j].append(xentropy_data[j])
                all_numsites[j].append(numsites_data[j])
        
        site_rats = []
        for i in range(nlevels):
            site_rat = []
            for j in range(len(self.layers)):
                site_rat.append(all_numsites[i][j]/tots[j])
            site_rats.append(site_rat)
        
        self.maxes = []
        self.mins = []
        maxes = maxes_mins[0]
        mins = maxes_mins[1]
        for layer in self.layers:
            self.maxes.append(maxes[layer])
            self.mins.append(mins[layer])
            
        self.error = all_error
        self.xentropy = all_xentropy
        self.numsites = all_numsites
        self.siteratio = site_rats
        self.sparsity = zeros
        self.num_samples = num_samples
        
        self.avg_errors = []
        self.avg_sites = []
        self.avg_rats = []
        for i in range(self.nlevels):
            self.avg_errors.append(1 - np.mean(np.array(self.error[i])))
            self.avg_sites.append(np.mean(np.array(self.numsites[i])))
            self.avg_rats.append(np.mean(np.array(self.siteratio[i])))
        
        self.extracted = True
        
    def diff_data(self):
        error_arr = np.zeros(len(self.layers))
        avg_diffs = []
        avg_sites = []
        for i in range(self.nlevels):
            avg_diffs.append(np.mean(np.array(self.error[i]) - error_arr))
            error_arr = np.array(self.error[i])
            
            avg_sites.append(np.mean(np.array(self.numsites[i])))
        self.log("Avg diffs: " + str(avg_diffs))
        self.log("Avg sites: " + str(avg_sites))
        
            
    def plot_v2(self, level_names=[], xentropy=False, sparsity=False, num_sites=False, overlay=True, sites_ratio=False, maxes_mins=False):
        assert(self.extracted)
        
        # plot on x axis the layers and on y axis the correct rate
        fig, axes = plt.subplots(figsize=(10, 8))
        assert(sparsity + num_sites + maxes_mins + sites_ratio <= 1)
        if sparsity or num_sites or maxes_mins or sites_ratio:
            axes1 = axes.twinx()
            axes1.tick_params(labelsize=TICK_SIZE)
        axes.tick_params(labelsize=TICK_SIZE)
        
        # plot the error data
        for i in range(self.nlevels):
            if not xentropy:
                axes.plot(self.layers, self.error[i], linestyle='-', marker='o')
                # axes.plot(self.layers, [1 - e for e in self.error[i]])
                # axes.set_ylim(bottom=0)
            else:
                axes.plot(self.layers, self.xentropy[i])
        
        # below plots other data alongside the error data       
        cmap = plt.get_cmap('tab10')
        if sparsity:
            spars_color = "tab:brown"
            spars_linestyle = "--"
            axes1.set_ylabel("Perc. of inputs zero", color=spars_color, fontsize=AXES_SIZE)
            axes1.plot(self.layers, self.sparsity, color=spars_color, linestyle=spars_linestyle, marker='o')
            axes1.tick_params(axis='y', labelcolor=spars_color)
        elif sites_ratio:
            linestyle = "--"
            for i in range(self.nlevels):
                axes1.plot(self.layers, self.siteratio[i], linestyle=linestyle, marker='o')
            axes1.set_ylabel("Ratio of sites/total elements", fontsize=AXES_SIZE)
        elif num_sites:
            linestyle = "--"
            axes1.set_ylabel("Avg. number of reuse sites", fontsize=AXES_SIZE)
            for i in range(self.nlevels):
                color = cmap(i)
                axes1.plot(self.layers, self.numsites[i], color=color, linestyle=linestyle, marker='o')
        elif maxes_mins:
            max_linestyle = "--"
            min_linestyle = ":"
            color = "k"
            axes1.set_ylabel("Range of output values of layer", fontsize=AXES_SIZE)
            axes1.plot(self.layers, self.maxes, color=color, linestyle=max_linestyle, marker='o')
            axes1.plot(self.layers, self.mins, color=color, linestyle=min_linestyle, marker='o')
            
        fig.legend(labels=level_names)
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # set axis titles
        axes.set_xlabel("Layer number", fontsize=AXES_SIZE)
        if not xentropy:
            axes.set_ylabel("Avg. Top-1 Accuracy", fontsize=AXES_SIZE)
        else:
            axes.set_ylabel("Avg. XEntropy", fontsize=AXES_SIZE)
        
        title = fancy_names[self.net_name] + " on " + fancy_names[self.arch_name] + " " + self.d_type_name
        axes.set_title(title, fontsize=TITLE_SIZE)
        
        # save the figure to a png file in the `plots` directory
        target_name = self.img_dir + "layers_" + self.d_type + self.add_on
        if xentropy:
            target_name += "_x"
            
        if sparsity:
            target_name += "_sp"
        elif sites_ratio:
            target_name += "_srat"
        elif num_sites:
            target_name += "_ns"
        elif maxes_mins:
            target_name += "_max"
        plt.savefig(get_name(target_name, overwrite=self.overwrite))
        plt.close('all')
        
    def correlate(self, level_names=[]):
        assert(self.extracted)
        
        if not level_names:
            for i in range(self.nlevels):
                level_names.append(str(i))
        
        # DATA WITH LEVELS: --> make separate correlation matrices (or avg out for all the levels)
        # error, xentropy, num sites, site ratio, 
        corrMs = []
        for i in range(len(level_names)):
            # get the correlation matrix
            level_data = {"Error": self.error[i], 
                          "Entropy": self.xentropy[i], 
                          "Sites": self.numsites[i],
                          "SiteRatio": self.siteratio[i]}
            df = pd.DataFrame(level_data)
            corr = df.corr()
            corrMs.append(corr)
            
            # create heatmap
            fig = plt.figure()
            fig.suptitle(" " + self.net_name + " on " + self.arch_name + " " + self.d_type_name + " " + level_names[i])
            sn.heatmap(corr, annot=True, center=0, vmin=-1, vmax=1)
            
            # get name for saving the img and save it
            target_name = self.corr_dir + "corr_" + self.d_type + "_level" + str(i) + self.add_on
            plt.savefig(get_name(target_name, overwrite=self.overwrite))
        plt.close('all')
            
        # WITHOUT: --> correlate them with all the levels (or just the avg)
        # sparsity, maxes
        data = {}
        for i in range(self.nlevels):
            data["Level"+str(i)] = self.error[i]
        data["Sparsity"] = self.sparsity
        data["Maxes"] = self.maxes
        data["Mins"] = self.mins
        df = pd.DataFrame(data)
        
        corr = df.corr()
        fig = plt.figure()
        fig.suptitle("Overall " + self.net_name + " on " + self.arch_name + " " + self.d_type_name)
        sn.heatmap(corr, annot=True, center=0, vmin=-1, vmax=1)
        
        # get name for saving the img and save it
        target_name = self.corr_dir + "corr_" + self.d_type + self.add_on
        plt.savefig(get_name(target_name, overwrite=self.overwrite))
        plt.close('all')
        
def get_name(target_name:str, overwrite=True):
    if not overwrite:
        img_name, _ = get_new_filename(target_name, "png")
    else:
        img_name = target_name + ".png"
    print("Creating plot at: " + str(img_name))
    return img_name


fancy_names = {"alexnet": "AlexNet", 
               "deit_tiny": "DeiT-tiny",
               "efficientnet_b0": "EfficientNet-B0",
               "resnet18": "ResNet-18",
               "nvdla": "NVDLA",
               "eyeriss": "Eyeriss",
               "simba": "Simba"}

def get_max_mat(data_dict):
    y_subplots = max(len(d) for d in data_dict.values())
    error_mat = [[] for i in range(y_subplots)]
    sites_mat = [[] for i in range(y_subplots)]
    for arch_name in data_dict:
        y_coord = 0
        arch_dict = data_dict[arch_name]
        for net_name in arch_dict:
            error, avg_sites = arch_dict[net_name]
            error_mat[y_coord].append(max(error))
            sites_mat[y_coord].append(max(avg_sites))
            y_coord += 1
    return error_mat, sites_mat

def combine_plots(data_dict, names_dict, d_type, bar_plot=False, all_d_types=False):
    print("Creating BIG plot...")
    x_subplots = len(data_dict)
    y_subplots = max(len(d) for d in data_dict.values())
    fig, axes = plt.subplots(y_subplots, x_subplots, figsize=(8, 10), sharex=True)
    
    x_coord = 0
    ax_mat = [[] for i in range(y_subplots)]
    max_errors, max_sites = get_max_mat(data_dict)

    row_maxes = []
    for e, s in zip(max_errors, max_sites):
        row_maxes.append((max(e), max(s)))
        
    for arch_name in data_dict:
        y_coord = 0
        arch_dict = data_dict[arch_name]
        for net_name in arch_dict:
            error, avg_sites = arch_dict[net_name]
            level_names = names_dict[arch_name]
            
            ax = axes[y_coord, x_coord]
            twin_ax = ax.twinx()
            ax_mat[y_coord].append((ax, twin_ax))
            
            if not error:
                ax.axis('off')
                twin_ax.axis('off')
                continue
            
            if not bar_plot:
                ax.plot(error, marker='o', markersize=MARKER_SIZE, color=PLOT_COLOR)
                row_max = row_maxes[y_coord]
                # ax.set_ylim(bottom=0)
                ax.set_ylim(bottom=0, top=row_max[0]*1.1)
                
                twin_ax.plot(avg_sites, linestyle="--", marker='o', markersize=MARKER_SIZE, color=SITES_COLOR)
                # twin_ax.set_ylim(bottom=0)
                twin_ax.set_ylim(bottom=0, top=row_max[1]*1.1)
            else:
                ax.bar(error, marker='o', markersize=MARKER_SIZE, color=PLOT_COLOR)
                twin_ax.plot(avg_sites, linestyle="--", marker='o', markersize=MARKER_SIZE, color=SITES_COLOR)
            twin_ax.tick_params(axis='y', labelcolor=SITES_COLOR)
            
            # plot the given labels for the x values (i.e. DRAM, etc.)
            tick_list = [i for i in range(len(level_names))]
            ax.set_xticks(tick_list)
            ax.set_xticklabels(level_names, fontsize=TICK_SIZE, rotation=45)
            
            if x_coord > 0:
                ax.set_yticklabels([])
            if x_coord < x_subplots - 1:
                twin_ax.set_yticklabels([])
        
            # axes.set_xlabel("Memory Level (larger \u2192 smaller)", fontsize=AXES_SIZE)
            # axes.set_ylabel("Avg. Top-1 Error", fontsize=AXES_SIZE)
            # axes.set_title(title, fontsize=TITLE_SIZE)
            
            # plt.ylabel("Avg. number of reuse sites", color='tab:blue', fontsize=AXES_SIZE)
                
            y_coord += 1
        x_coord += 1
    
    # get name for saving the img and save it
    data_dir = "data_results/plots/"
    p = Path(data_dir)
    p.mkdir(parents=True, exist_ok=True)
    target_name = data_dir + "big_plot_" + d_type
    plt.savefig(get_name(target_name, overwrite=False))
    plt.close('all')
    