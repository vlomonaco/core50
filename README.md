<img src="http://i.imgur.com/2UyfKHs.png?1" width="80" align="left">

# CORe50 

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
[![built with Python2.7](https://img.shields.io/badge/build%20with-python2.7-red.svg)](https://www.python.org/)
[![built with Caffe](https://img.shields.io/badge/build%20with-caffe-brightgreen.svg)](http://caffe.berkeleyvision.org/)
[![built with Sacred](https://img.shields.io/badge/build%20with-sacred-yellow.svg)](https://github.com/IDSIA/sacred)

## A new Dataset and Benchmark for Continuous Object Recognition

#### *WARNING: This repository is still under construction!*

- [x] CORe50 core code-base
- [x] CORe50 benchmark configuration files
- [x] Easy-to-access results data and baches configurations
- [ ] Getting started scripts
- [ ] Easy-setup and demos

In this page we provide the code and all the materials related to the **CORe50** 
benchmark. If you plan to use this dataset or other resources you'll find in this page, please **cite our latest paper**: 

	@article{lomonaco2017core50,
       title={CORe50: a New Dataset and Benchmark for Continuous Object Recognition},
       author={Lomonaco, Vincenzo and Maltoni, Davide},
       journal={arXiv preprint arXiv:1705.03550},
       year={2017}
	}

You can find more information about the dataset and benchmark at: 
[vlomonaco.github.io/core50](http://vlomonaco.github.io/core50).

## Dependencies

In order to extecute the code in the repository you'll need to install the following dependencies:

* [Python 2.7](https://www.python.org/)
* [Numpy](https://pypi.python.org/pypi/numpy/1.6.1)
* [Caffe](http://caffe.berkeleyvision.org/)
* [Sacred](https://github.com/IDSIA/sacred)

You can find a step-by-step guide for installing caffe [here](http://caffe.berkeleyvision.org/installation.html). 
Sacred is not really necessary (and you can easily remove from the source code if you want to). Still, it is very nice for managing a lot of experiments configurations and [it's very simple to install](https://github.com/IDSIA/sacred#installing). Numpy can be installed via your distribution package manager or [pip](https://pypi.python.org/pypi/pip).

## Getting Started

Up to now, the code used to run the experiments is all here. Still, we plan to
add in the near future the sacred configuration files and a single script which
can install missing dependencies, download all the (external) necessary 
materials, set the local paths for you in order to easily reproduce the baselines
you'll find in the paper. 

### Project Structure
Up to now the projects is structured as follows:
```
CORe50/
├── confs
│   ├── sI
│   │   └── ...
│   │   
│   ├── sII
│   │   └── ...
│   │   
│   └── sIII 
│       └── ...
│
├── core
│   ├── convert_lmdb.py
│   ├── core50_inc_finetuning.py
│   ├── create_filelist_utils.py
│   ├── create_sI_filelist.py
│   ├── create_sII_filelist.py
│   ├── create_sIII_filelist.py
│   └── inc_finetuning.py
│
├── data
│   ├── batches_filelists.zip
│   ├── dataset_dims.zip
│   ├── NC.tsv
│   ├── NIC.tsv
│   ├── NI.tsv
│   ├── results.pkl
│   ├── results_tsv.zip
│   ├── seq_results.pkl
│   └── seq.tsv
│
├── LICENSE
├── README.md
├── run_sI_exps.sh
├── run_sII_exps.sh
├── run_sIII_exps.sh
└── scripts
```

In the **confs** folder you can find all the experiments configurations and the caffe
definition files. In **core** the actual code of the benchmark; in **data** results and configuration files you can download without delving into the code. 
Finally, **scripts** (currently void) will contain useful scripts to help you downloading the necessary extra materials. Bash scripts will be updated soon for running the benchmark without much pain. 

## License

This work is licensed under a <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. 

## Author

* [Vincenzo Lomonaco](http://vincenzolomonaco.com) - email: *vincenzo.lomonaco@unibo.it*
