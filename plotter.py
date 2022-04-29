from matplotlib import pyplot as plt
from pathlib import Path
from IPython.display import display
from helpers import *
import pandas as pd
import os

# object used to plot data
class Plotter():
    def __init__(self, arch_name, net_name, layers=[], d_type='i', add_on=''):
        self.arch_name = arch_name
        self.layers = layers
        self.net_name = net_name
        self.filenames = []
        self.d_type = ''
        if d_type == 'i':
            self.d_type_name = 'inputs'
        elif d_type == 'w':
            self.d_type_name = 'weights'
        if not self.layers:
            self.get_layers()
        self.set_filenames(add_on)
        
    def get_layers(self):
        layers = [os.path.basename(os.path.normpath(f.path)) for f in os.scandir("data_results/" + self.arch_name + "/" + self.net_name + "/") if f.is_dir()]
        layers = [get_str_num(layer) for layer in layers]
        self.layers = [layer for layer in layers if layer != None]

    def set_filenames(self, add_on=''):
        top_dir = "data_results/" + self.arch_name + "/" + self.net_name
        self.top_dir = top_dir
        # get the filenames for each layer
        for i in self.layers:
            # create the nested directories
            dir = top_dir + "/conv" + str(i) + "/"
            filename = dir + "data_" + self.d_type_name
            if add_on:
                filename += "_" + add_on
            filename += ".csv"
            self.filenames.append(filename)

        # create directory to store the plot images
        img_dir = top_dir + "/plots/"
        self.img_dir = img_dir
        p = Path(img_dir)
        p.mkdir(parents=True, exist_ok=True)
    
    # manipulate data along axis_title on the passed in df
    def agg_data(self, df, axis_title):
        group = df.groupby([axis_title])
        
        data2collect = ["CorrectClassConf"]
        
        data_mean = group[data2collect].mean()
        data_mean = data_mean.add_suffix("_mean")
        
        data_std = group[data2collect].std()
        data_std = data_std.add_suffix("_std")
        
        data = data_mean.join(data_std)
        
        num_samples = group.size()
        error_rate = group["ClassifiedCorrect"].sum() / num_samples
        error_rate.name = "Error Rate"
        num_samples.name = "Num Samples"
        
        data = data.join(error_rate)
        data = data.join(num_samples)
        
        return data

    # collect necessary data along an axis of the collected data
    # given by axis_title - for the layer conv_id
    def collect_layer_data(self, conv_id, axis_title):
        df = pd.read_csv(self.filenames[conv_id])
        return self.agg_data(df, axis_title)
    
    # collect data for all layers - grouped by the axis_title
    def agg_layer_data(self, axis_title):
        all_df = []
        for i in range(len(self.filenames)):
            df = pd.read_csv(self.filenames[i])
            all_df.append(df)
        final_df = pd.concat(all_df, ignore_index=True)
        return self.agg_data(final_df, axis_title)

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
            img_name = get_new_filename(self.img_dir + img_name, "png")
            plt.savefig(img_name)
        if show:
            plt.show()