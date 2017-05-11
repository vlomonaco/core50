<img src="http://i.imgur.com/2UyfKHs.png?1" width="80" align="left">

# Core50 

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
[![built with Python2.7](https://img.shields.io/badge/build%20with-python2.7-red.svg)](https://www.python.org/)
[![built with Caffe](https://img.shields.io/badge/build%20with-caffe-brightgreen.svg)](http://caffe.berkeleyvision.org/)
[![built with Caffe](https://img.shields.io/badge/build%20with-sacred-yellow.svg)](https://github.com/IDSIA/sacred)

## A new Dataset and Benchmark for Continuous Object Recognition

#### *WARNING: This repository is still under construction!*

In this page we provide the code and all the materials related to the CORe50 
benchmark. If you plan to use this dataset or other resources you'll find in this page, please cite our latest paper: 

>  Vincenzo Lomonaco and Davide Maltoni. **"CORe50: a new Dataset and Benchmark for Continuous Object Recognition"**. arXiv preprint arXiv:1511.03163 (2017). 

You can find more information about the dataset and benchmark at: 
[vlomonaco.github.io/core50](http://vlomonaco.github.io/core50).


## About the code

Up to now, the code used to run the experiments is all here. Still, we plan to
add in the near future the sacred configuration files and a single script which
can install missing dependencies, download all the (external) necessary 
materials, set the local paths for you in order to easily reproduce the baselines
you'll find in the paper. 

## Dependencies

In order to extecute the code in the repository you'll need to install the following dependencies:

* [Python 2.7](https://www.python.org/)
* [Caffe](http://caffe.berkeleyvision.org/)
* [Sacred](https://github.com/IDSIA/sacred)

You can find a step-by-step guide for installing caffe [here](http://caffe.berkeleyvision.org/installation.html). 
Sacred is not really necessary (and you can easily remove from the source code if you want to). Still, it is very nice for managing a lot of experiments configurations and [it's very simple to install](https://github.com/IDSIA/sacred#installing).

## License

This work is licensed under a <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. 

## Author

* [Vincenzo Lomonaco](http://vincenzolomonaco.com) - email: *vincenzo.lomonaco@unibo.it*
