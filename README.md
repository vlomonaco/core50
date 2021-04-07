<img src="http://i.imgur.com/2UyfKHs.png?1" width="80" align="left">

# CORe50 

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
[![built with Python2.7](https://img.shields.io/badge/build%20with-python2.7-red.svg)](https://www.python.org/)
[![built with Caffe](https://img.shields.io/badge/build%20with-caffe-brightgreen.svg)](http://caffe.berkeleyvision.org/)
[![built with Sacred](https://img.shields.io/badge/build%20with-sacred-yellow.svg)](https://github.com/IDSIA/sacred)

## A new Dataset and Benchmark for Continual Learning and Object Recognition, Detection and Segmentation

----------------------------------------------

- [x] CORe50 core code-base
- [x] CORe50 benchmark configuration files
- [x] Easy-to-access results data and batches configurations
- [x] Easy-setup, getting started and Python data loader
- [x] Experiments ported to Python 3.x
- [x] **New realease and additional baselines within [Avalanche](https://avalanche.continualai.org)**

In this page we provide the code and all the materials related to the **CORe50** 
benchmark. If you plan to use this dataset or other resources you'll find in this page, please cite our latest papers ["CORe50: a New Dataset and Benchmark for Continuous Object Recognition"](http://proceedings.mlr.press/v78/lomonaco17a.html) and ["Fine-Grained Continual Learning"](https://arxiv.org/abs/1907.03799): 

	@InProceedings{lomonaco2017core50,
	   title = {CORe50: a New Dataset and Benchmark for Continuous Object Recognition},
	   author = {Vincenzo Lomonaco and Davide Maltoni},
	   booktitle = {Proceedings of the 1st Annual Conference on Robot Learning},
	   pages = {17--26},
	   year = {2017},
	   volume = {78}
	}

	@article{lomonaco2019nicv2,
	   title = {Fine-Grained Continual Learning},
	   author = {Vincenzo Lomonaco and Davide Maltoni and Lorenzo Pellegrini},
	   journal = {Arxiv preprint arXiv:1907.03799},
	   year = {2019}
	}
	
You can find more information about the dataset/benchmark as well as additional data to download at: 
[vlomonaco.github.io/core50](http://vlomonaco.github.io/core50).

----------------------------------------------

## Dependencies

In order to extecute the code in the repository you'll need to install the following dependencies in a [Python 3.x](https://www.python.org/) environment:

* [Numpy](https://pypi.python.org/pypi/numpy/1.6.1): _Matrices operations and stuff_

```bash
pip install numpy
```

* [Sacred](https://github.com/IDSIA/sacred): _Experiments Manager_

```bash
pip install sacred
```

* [Caffe](http://caffe.berkeleyvision.org/): _Current DL back-end (easily interchangeable)_

Follow the step-by-step guide for installing caffe [here](http://caffe.berkeleyvision.org/installation.html). 

----------------------------------------------

## Project Structure
Up to now the projects is structured as follows:

- [`confs/`](confs): In this folder you can find all the experiments configurations and the caffe definition files. sI, sII and sIII stand for the NI, NC and NIC scenarios, respectively.
- [`core/`](core): The actual code of the benchmark.
- [`data/`](data): After the setup it will be created and filled with data needed for the experiments. It will also be used for storing partial computations.
- [`extras/`](extras): Results and configuration files you can download without delving into the code.
- [`scripts/`](scripts): It contains useful scripts to help you downloading the necessary materials, setup the environment or load the data for your experiments in Python.
- [`LICENSE`](LICENSE): Standard Creative Commons Attribution 4.0 International License.
- [`README.md`](README.md): This instructions file.
- [`run_sI_exps.sh`](run_sI_exps.sh): Simple bash script for running the _"New Instances (NI)"_ experiments with the different architectures and strategies
- [`run_sII_exps.sh`](run_sII_exps.sh): Simple bash script for running the _"New Classes (NC)"_ experiments with the different architectures and strategies
- [`run_sIII_exps.sh`](run_sIII_exps.sh): Simple bash script for running the _"New Instances and Classes (NIC)"_ experiments with the different architectures and strategies

----------------------------------------------

## Getting Started

First of all, let's clone the repository:

```bash
git clone https://github.com/vlomonaco/core50.git
```

Then, in order to run the experiments and reproduce the benchmark we need to download the pre-trained models and the CORe50 dataset. This can be automatically done using the script provided:

```bash
cd core50
./scripts/bash/fetch_data_and_setup.sh
```

All the data will be downloaded in the [`data/`](data) directory. After this initial step you can directly run the experiments with the bash scripts [`run_sI_exps.sh`](run_sI_exps.sh), [`run_sII_exps.sh`](run_sII_expts.sh) and [`run_sIII_exps.sh`](run_sI_expts.sh) for the NI, NC and NIC scenarios respectively. 

For example, reproducing the first scenario experiments can be as easy as running:

```bash
./run_sI_exps.sh
```

Since this experiments can take a while (also more than 24h depending on the scenario) you can also disable some experiments just by commenting them in the bash script.

----------------------------------------------

## Troubleshooting

- If you find different results from out benchmark (for a few percentage points) _that is to be expected_! First of all because we use the `cudnn` engine which is not fully deterministic for convolutions. Second because the error may be accumulated during the incremental learning process. If you want full reproducibility (which means a ~2x in terms of time needed) just set the `engine` param of convolutions [to 1](http://caffe.berkeleyvision.org/tutorial/layers/convolution.html).

- If you find some trouble with the `freezeweights` strategy this is probably because you need to reset the learning rate multipliers in the prototxt (sorry, my bad.. I'm currently working on a new version of the code for creating the prototxt files instead of modifying them on the fly.).

- Hey! If you find any trouble don't get frustrated, just ask, we'll answer in a few hours! :-)

----------------------------------------------

## License

This work is licensed under a <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. 

----------------------------------------------

## Author

* [Vincenzo Lomonaco](http://vincenzolomonaco.com) - email: *vincenzo.lomonaco@unibo.it*
