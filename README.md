# DNN-Dataflow-Resilience
Source code for my senior thesis (ES100) project aiming to build an analysis framework for the resilience of DNN accelerator dataflow architectures. 

## Getting Started
To start:
1. Clone this repository locally by using `git clone`.
2. Install all necessary dependencies using `pip install -r requirements.txt`.
3. To run an Eyeriss or NVDLA example test, see `run.py`, which will output data to a folder called `data_results`. To run tests on either the frontend or backend, see `backend_test.py` and `frontend_test.py`.

The `ImageNet` folder provides a set of validation labels (given in `validation_labels.csv`) and a set of 100 images in the ImageNet 2012 validation dataset. To use a different dataset, see `dataset.py` for more details. 

To generate your own mappings for different networks, you will have to install and use [Timeloop](https://github.com/NVlabs/timeloop). Example mappings for NVDLA are provided in `run.py` for a few networks and mapping files (generated from Timeloop) for ResNet18 on Eyeriss are provided within the `timeloop_mappings` folder.

## Tool High-Level Overview
![whoops](https://github.com/jaylenwang7/DNN-Dataflow-Resilience/blob/main/figures/framework.jpg)

The tool consists of a frontend and backend. The backend first uses the loop nests provided by the user (which can be produced by Timeloop) as well as the information of the layer sizes (for things like padding and stride) to generate the locations at the output of the layer where an error could propagate. It does this by using the loop nest to see, if an error occurred at some level of memory (flip flop, buffer, DRAM, etc.) how that error would be shared in memory, and thus, propagate to the output. The frontend then 

* The backend implementation is mainly in `loop.py` and `loop_var.py`, which do the loop nest simulation. 
* The frontend implementation is mainly in `inject_model.py`, which performs the injection through a wrapper of the 
* The frontend and backend are tied together through `model_injection.py` which makes calls to both (runs the frontend and backend) and also handles data collection.

## Scripts
The following are Python scripts that can be run from the command line to spin up the tool (either to run experiments or to validate the tool itself):

* `run.py` contains all the code used for the experiments to generate the data I used in the paper. This file also contains very helpful scripts that boot up the entire tool and run experiments from scratch.
    - `run_injection(get_alexnet, "alexnet", "eyeriss", d_type="i")` is an example of running an injection experiment for Alexnet running on Eyeriss - the network name can be changed to one of the default ones (see `run.py` for which networks are automatically supported)
    - If running a new network/architecture pairing, you do not have to supply the max and min as it will be generated automatically. However, if you want to run an injection experiment multiple times then you should run something like `get_network_max(get_alexnet, get_dataset, n=10000)`.

* `frontend_test.py` contains tests used to test the frontend infrastructure. Such tests includes testing how much overhead the injection framework adds onto a normal inference. It also includes things like making sure that the proper number of output injection sites are actually different from a clean injection - if you know you're picking a certain number of output sites, then the number of elements that are different between the injected output and the clean output should be that exact amount. 

* `backend_test.py` contains tests used to specifically test the backend infrastructure. These tests include injections into specific locations, and the output sites then get outputted to a `data_results/` directory where they can be checked against the expected output. For NVDLA, this was done by hand, where the output sites as sited by FIdelity were checked by hand for each memory level. Likewise, this was done for Eyeriss as well, where the test can be run to verify that injections into the input result in a row of the output window being affected - and the results of injections not verified in the paper also fall in line with expectation with an understanding of the Eyeriss dataflow.

## Data Output
* Injection data (data resulting from a full experiment running through the front and backend) is automatically outputted into a `data_results/` directory. This directory will then have a directory for each architecture, a `plot/` directory for plots that aggregate over all architectures, and a `stats/` directory for statistics that are aggregated over all layers and injections for each network/arch pairing.  
* Within the arch-specific `data_results/` directory, you will find directories for each network. Within those you can find directories for each layer injected into, which then hold csv files with data for the result of each injection performed. There is also a `log.txt` file which contains outputs for each experiment with what indices/images/etc. were used for future reference.

## Plotting
* Plots can be generated through using the `Plotter` object in `Plotter.py`. 
* The aggregate stats files are useful for generating plots about statistics such as fault rates for different bits etc. To generate plots, I suggest (as I did to generate figures) opening them with Excel or another spreadsheet tool and generating plots and seeing any trends that way.


**NOTE this is a work in progress, so documentation is sparse and there are bugs/improvements that are being made.**
