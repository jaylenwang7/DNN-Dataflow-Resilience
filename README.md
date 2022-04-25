# DNN-Dataflow-Resilience
Source code for my senior thesis (ES100) project aiming to build an analysis framework for the resilience of DNN accelerator dataflow architectures. 

To start:
1. Clone this repository locally by using `git clone`.
2. Install all necessary dependencies using `pip install -r requirements.txt`.
3. To run an Eyeriss or NVDLA example test, see `run.py`, which will output data to a folder called `data_results`. To run tests on either the frontend or backend, see `backend_test.py` and `frontend_test.py`.

The `ImageNet` folder provides a set of validation labels (given in `validation_labels.csv`) and a set of 100 images in the ImageNet 2012 validation dataset. To use a different dataset, see `dataset.py` for more details. 

To generate your own mappings for different networks, you will have to install and use [Timeloop](https://github.com/NVlabs/timeloop). Example mappings are provided in `run.py` and mapping files (generated from Timeloop) for ResNet18 on Eyeriss are provided within the `timeloop_mappings` folder. 

**NOTE this is a work in progress, so documentation is sparse and there are bugs/improvements that are being made.**
