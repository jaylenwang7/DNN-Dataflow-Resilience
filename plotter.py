from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from IPython.display import display
from torch import zeros_like
from helpers import *
import pandas as pd
import os
import functools as ft

# object used to plot data
class Plotter():
    def __init__(self, arch_name, net_name, layers=[], d_type='i', add_on=''):
        self.arch_name = arch_name
        self.layers = layers
        self.net_name = net_name
        self.filenames = []
        self.d_type = d_type
        self.add_on = add_on
        if d_type == 'i':
            self.d_type_name = 'inputs'
        elif d_type == 'w':
            self.d_type_name = 'weights'
        if not self.layers:
            self.get_layers()
        else:
            self.layers.sort()
        self.set_filenames(add_on)
        
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
        img_dir = top_dir + "/plots/"
        self.img_dir = img_dir
        p = Path(img_dir)
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
    # given by axis_title - for the layer conv_id
    def collect_layer_data(self, conv_id, axis_title):
        df = pd.read_csv(self.filenames[conv_id])
        return self.agg_data(df, axis_title)
    
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
    def plot(self, correct_rate=-1, xlabels=[], show_chart=False, show=True, img_name='', agg_layers=False):
        
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        labels = []
        # plot dotted line for original correct rate
        axes.axhline(y=correct_rate, color='black',ls='--')
        labels.append("No error")
        
        if not agg_layers:
            # loop through layers
            for i in range(len(self.layers)):
                # collect data aggregated by level
                level_data = self.collect_layer_data(i, "Level")

                # display chart if want
                if show_chart:
                    display(level_data)

                # plot the error rate
                ax = level_data['Error Rate'].plot(style='o', ms=10, logx=False)

                # plot the given labels for the x values (i.e. DRAM, etc.)
                tick_list = [i for i in range(len(xlabels))]
                ax.set_xticks(tick_list)
                ax.set_xticklabels(xlabels, fontsize=12)
                
                labels.append("Conv" + str(self.layers[i]))
        else:
            level_data = self.agg_layer_data("Level")
            # plot the error rate
            ax = level_data['Error Rate'].plot(style='o', ms=10, logx=False)
            
            # plot the given labels for the x values (i.e. DRAM, etc.)
            tick_list = [i for i in range(len(xlabels))]
            ax.set_xticks(tick_list)
            ax.set_xticklabels(xlabels, fontsize=12)
        
        fig.legend(labels=labels,
                   loc="right")
                #    prop={'size': 20},
                #    bbox_to_anchor=(0.5, 0.5))

        plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.4, 
                            hspace=0.4)

        # TODO: not general
        title = self.net_name + " on " + self.arch_name + " Top-1 Acc."
        axes.set_title(title)
        axes.set_xlabel("Memory Level (larger \u2192 smaller)", fontsize=14)
        axes.set_ylabel("Mean Top-1 Accuracy", fontsize=14)
        # axes.set_ylim(bottom=0.45)
        # if requested save img of plot in dir
        if img_name:
            # don't overwrite older image
            img_name, _ = get_new_filename(self.img_dir + img_name, "png")
            plt.savefig(img_name)
        if show:
            plt.show()
            
    def plot_v2(self, level_names=[], xentropy=False, sparsity=False, num_sites=False, overlay=True, sites_ratio=False, maxes_mins=[]):
        # plot on x axis the layers and on y axis the correct rate
        fig, axes = plt.subplots(figsize=(10, 8))
        plot_maxes = bool(maxes_mins)
        assert(sparsity + num_sites + plot_maxes + sites_ratio <= 1)
        if sparsity or num_sites or plot_maxes or sites_ratio:
            axes1 = axes.twinx()
            
        if plot_maxes:
            new_maxes = []
            new_mins = []
            maxes = maxes_mins[0]
            mins = maxes_mins[1]
            for layer in self.layers:
                new_maxes.append(maxes[layer])
                new_mins.append(mins[layer])
            maxes = new_maxes
            mins = new_mins
        
        # get sparsity data - comes out as list of list of three values (output, input, weight)
        # only really concerned with input - also all levels are the same
        # NOTE: these numbers are aggregate not averaged for each data point
        zeros, tots, nums = self.collect_zero_data(input_only=False, nonzeros=False)
        input_zeros = zeros[1]
        new_tots = []
        for i in range(len(self.layers)):
            new_tots.append(tots[0][i]/
                            nums[i])
        tots = new_tots
        
        # get the number of levels
        level_data = self.collect_layer_data(0, "Level")
        nlevels = len(level_data['Error Rate'].tolist())
        
        # loop through each level and get the data for each level
        all_error = [[] for i in range(nlevels)]
        all_xentropy = [[] for i in range(nlevels)]
        all_numsites = [[] for i in range(nlevels)]
        for i in range(len(self.layers)):
            # get a table with the data for each level
            level_data = self.collect_layer_data(i, "Level")
            # display(level_data)
            
            # extract a column of the data and turn into a list, with data for each level
            error_data = level_data['Error Rate'].tolist()
            xentropy_data = level_data['XEntropy_mean'].tolist()
            numsites_data = level_data['NumSites_mean'].tolist()
            
            # transfer the data in the list to an aggregate list
            for j in range(nlevels):
                all_error[j].append(error_data[j])
                all_xentropy[j].append(xentropy_data[j])
                all_numsites[j].append(numsites_data[j])  
        
        for i in range(nlevels):
            if not xentropy:
                axes.plot(self.layers, all_error[i])
            else:
                axes.plot(self.layers, all_xentropy[i])
                
        cmap = plt.get_cmap('tab10')
                
        if sparsity:
            spars_color = "tab:brown"
            spars_linestyle = "--"
            axes1.set_ylabel("Perc. of inputs zero", color=spars_color)
            axes1.plot(self.layers, input_zeros, color=spars_color, linestyle=spars_linestyle)
            axes1.tick_params(axis='y', labelcolor=spars_color)
        elif sites_ratio:
            linestyle = "--"
            for i in range(nlevels):
                site_rat = []
                for j in range(len(self.layers)):
                    site_rat.append(all_numsites[i][j]/tots[j])
                axes1.plot(self.layers, site_rat, linestyle=linestyle)
            axes1.set_ylabel("Ratio of sites/total elements")
        elif num_sites:
            linestyle = "--"
            axes1.set_ylabel("Mean number of reuse sites")
            for i in range(nlevels):
                color = cmap(i)
                axes1.plot(self.layers, all_numsites[i], color=color, linestyle=linestyle)
        elif plot_maxes:
            max_linestyle = "--"
            min_linestyle = ":"
            color = "k"
            axes1.set_ylabel("Range of output values of layer")
            axes1.plot(self.layers, maxes, color=color, linestyle=max_linestyle)
            axes1.plot(self.layers, mins, color=color, linestyle=min_linestyle)
            
        fig.legend(labels=level_names)
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # set axis titles
        fontsize = 14
        axes.set_xlabel("Layer number", fontsize=fontsize)
        if not xentropy:
            axes.set_ylabel("Mean Top-1 Accuracy", fontsize=fontsize)
        else:
            axes.set_ylabel("Mean XEntropy", fontsize=fontsize)
        
        title = self.net_name + " on " + self.arch_name + " " + self.d_type_name
        axes.set_title(title)
        
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
        elif plot_maxes:
            target_name += "_max"
        img_name, _ = get_new_filename(target_name, "png")
        print("Creating plot at: " + str(img_name))
        plt.savefig(img_name)