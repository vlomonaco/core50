#!/usr/bin/env bash

# Setup
mkdir data
mkdir data/logs
mkdir data/snapshots


echo "Downloading caffenet pre-trained model..."
wget --directory-prefix='../data/' http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel

echo "Downloading VGG_CNN_M pre-trained model..."
wget --directory-prefix='../data/' http://www.robots.ox.ac.uk/%7Evgg/software/deep_eval/releases/bvlc/VGG_CNN_M.caffemodel

echo "Downloading Core50 (128x128 version)..."
wget --directory-prefix='../data/' http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip

echo "Unzipping Core50..."
unzip ../data/core50_128x128.zipfile.zip -d ../data/
