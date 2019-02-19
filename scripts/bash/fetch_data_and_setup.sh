#!/usr/bin/env bash

# Setup
DIR="$( cd "$( dirname "$0" )" && pwd )"
mkdir $DIR/../../data
mkdir $DIR/../../data/logs
mkdir $DIR/../../data/snapshots

echo "Downloading caffenet pre-trained model..."
wget --directory-prefix=$DIR'/../../data/' http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

echo "Downloading VGG_CNN_M pre-trained model..."
wget --directory-prefix=$DIR'/../../data/' http://www.robots.ox.ac.uk/%7Evgg/software/deep_eval/releases/bvlc/VGG_CNN_M.caffemodel

echo "Downloading Core50 (128x128 version)..."
wget --directory-prefix=$DIR'/../../data/' http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip

echo "Unzipping Core50..."
unzip $DIR/../../data/core50_128x128.zip -d $DIR/../../data/
