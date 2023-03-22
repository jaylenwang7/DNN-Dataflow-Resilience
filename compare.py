import argparse
import os
import pandas as pd
import pickle
import scipy
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Compare hardware way and naive way')
    parser.add_argument('network', type=str, help='Network name')
    parser.add_argument('arch', type=str, help='Architecture name')
    parser.add_argument('dtype', type=str, help='Data type')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.dtype == 'i':
        dtype = 'input'
    elif args.dtype == 'o':
        dtype = 'output'
    elif args.dtype == 'w':
        dtype = 'weight'
    else:
        raise ValueError('Invalid data type')

    hw_avgs = {}
    predicted_avgs = {}
    hw_site_counts = {}

    hw_dir = "loop_results_pickle"
    # loop through each layer directory within {hw_dir}/{arch}/{network}
    for layer in os.listdir(f'{hw_dir}/{args.arch}/{args.network}'):
        # if not a dir or doesn't have layer in the name, skip
        if not os.path.isdir(f'{hw_dir}/{args.arch}/{args.network}/{layer}') or 'layer' not in layer:
            continue
        layer_num = int(layer.split('layer')[1].strip('_'))
        # ========= For working with data_results =========
        # data_file = f'{hw_dir}/{args.arch}/{args.network}/{layer}/data_{dtype}s.csv'
        # # read in the data as a pandas dataframe
        # data_csv = pd.read_csv(data_file)
        # # find the average of 'ClassifiedCorrect' column
        # avg = data_csv['ClassifiedCorrect'].mean()
        # hw_avgs[layer_num] = avg
        # # groupby 'NumSites' and then only use the count of each group
        # site_counts = data_csv['NumSites'].value_counts()
        # hw_site_counts[layer_num] = site_counts
        # =================================================

        loop_pickle_file = f'{hw_dir}/{args.arch}/{args.network}/{layer}/{dtype}_rates.pkl'
        if not os.path.exists(loop_pickle_file):
            continue
        with open(loop_pickle_file, 'rb') as f:
            data = pickle.load(f)
        avg = data['avg']
        site_counts = data['site_counts']
        # sum up the total number of sites
        total_sites = 0
        for site_count in site_counts.index:
            total_sites += site_counts[site_count]
        hw_avgs[layer_num] = avg
        hw_site_counts[layer_num] = site_counts

        # load the pickle file
        pickle_file = f'data_results_pickle/{args.network}/layer_{layer_num}/{dtype}_rates.pkl'
        data = None
        if not os.path.exists(pickle_file):
            continue
        with open(pickle_file, 'rb') as f:
            # the data is a pandas series
            data = pickle.load(f)
            data = data["out_rates"][0]

        # interpolate the data as a function of the number of sites using a cubic spline
        if len(data) >= 4:
            interp = scipy.interpolate.interp1d(data.index, data.values, kind='cubic', fill_value='extrapolate')
        else:
            interp = scipy.interpolate.interp1d(data.index, data.values, kind='linear', fill_value='extrapolate')
        
        predicted_avg = 0
        for site_count in site_counts.index:
            # find the predicted value for the value of site count
            # weight by the fraction of the data that has that value of site count
            predicted_avg += interp(site_count) * (site_counts[site_count] / total_sites)
        predicted_avgs[layer_num] = predicted_avg

    def order_dict(d):
        return {k: d[k] for k in sorted(d.keys())}
    hw_avgs = order_dict(hw_avgs)
    predicted_avgs = order_dict(predicted_avgs)
    
    # get correlation between the hw_avgs and predicted_avgs
    hw_avgs = pd.Series(hw_avgs)
    predicted_avgs = pd.Series(predicted_avgs)

    # concatenate the hw_avg and predicted_avg series
    all_avgs = pd.concat([hw_avgs, predicted_avgs], axis=1)
    # add a column for the difference between the two
    all_avgs['diff'] = all_avgs[0] - all_avgs[1]
    print(all_avgs)
    # print average of absolute difference
    print(f'Average absolute difference: {all_avgs["diff"].abs().mean()}')

    print(f'Correlation between HW and predicted: {hw_avgs.corr(predicted_avgs)}')

    # plot as bar chart
    all_avgs['diff'].plot.bar()
    plt.show()

if __name__ == '__main__':
    main()