# TIMELOOP-INJECTION DOCKER SETUP

### To Start

1. Make sure Docker is installed and running locally, then cd into this directory (`timeloop-injection/`).
2. Run

        docker-compose pull
        docker-compose up

3. You should see two links to paste into a browser - use whichever one works (often the second one works the best).
4. This should take you to a Jupyter Notebook environment where you can start working by opening `network-profile.ipynb`.

### Included Files/Directories
* `network-profile.ipynb`: Python notebook that provides the boiler plate code to run Timeloop and perform mappings. Requires very little code to write and just need to change some variables. See the notebook for more details.
* `profiler.py`: This contains code to first convert a PyTorch model into something readable by Timeloop, then it runs Timeloop commands for each layer and outputs mappings into certain locations.
* `archs/`: This is where you can put your own custom Timeloop architectures. By default you are given two examples: Eyeriss and Simba. 

### Using Timeloop
* This Docker setup provides an easy way to run Timeloop to provide a very low barrier of entry. 
* To include your own Timeloop architecture, you will need to be familiar with Timeloop. For all the details please see the Timeloop documentation: https://accelergy.mit.edu/tutorial.html and https://github.com/NVlabs/timeloop.
    - In short - make a new directory for your architecture within the `timeloop-injection/worskpace/archs/` directory.
    - This directory will then need to have three subdirectories: `arch`, `constraints`, and `mapper`.
    - `arch` contains yaml files that define the architecture setup, such as types of memory, memory sizes, etc.
    - `constraints` provides information to the mapper about the architecture dataflow where `*_arch_constraints` provides constraints that are strict for the architecture while `*_map_constraints` provide helpful constraints for the mapper that aren't necessary but will reduce the mapping space to make the search faster.
    - `mapper` provides a single yaml file that sets settings for the mapper such as the search algorithm, threads, timeout, and victory condition. If you find the mapper is taking too long, try changing settings here (specifically try reducing the victory condition and timeout).