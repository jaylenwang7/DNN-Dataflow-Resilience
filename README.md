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


## Scripts
The following are Python scripts that can be run from the command line to spin up the tool (either to run experiments or to validate the tool itself):

* `run.py` contains all the experiments used to generate the data I used in the paper (see the comments for which figures/data corresponds to which lines to run). This file also contains very helpful scripts that boot up the entire tool and run experiments from scratch.

* `frontend_test.py` contains tests used to test the frontend infrastructure. Such tests includes testing how much overhead the injection framework adds onto a normal inference. It also includes things like making sure that the proper number of output injection sites are actually different from a clean injection - if you know you're picking a certain number of output sites, then the number of elements that are different between the injected output and the clean output should be that exact amount. 

* `backend_test.py` contains tests used to specifically test the backend infrastructure. These tests include injections into specific locations, and the output sites then get outputted to a `data_results/` directory where they can be checked against the expected output. For NVDLA, this was done by hand, where the output sites as sited by FIdelity were checked by hand for each memory level. Likewise, this was done for Eyeriss as well, where the test can be run to verify that injections into the input result in a row of the output window being affected - and the results of injections not verified in the paper also fall in line with expectation with an understanding of the Eyeriss dataflow.

## Data Output
* Injection data (data resulting from a full experiment running through the front and backend) is 


**NOTE this is a work in progress, so documentation is sparse and there are bugs/improvements that are being made.**
