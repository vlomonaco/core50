#!/usr/bin/env bash

echo "running mid-caffenet, naive.."
python core50_inc_finetuning.py with sI/mid-caffeNet/naive.json -c "mid-caffenet, naive strategy, 10-run experiments, 2000 first batch it" 2> caffe.out > caffenet_naive.out
sleep 1
echo "running mid-caffenet, cumulative.."
python core50_inc_finetuning.py with sI/mid-caffeNet/cumulative.json -c "mid-caffenet, cumulative strategy, 10-run experiments, 2000 first batch it" 2> caffe.out > caffenet_cum.out
sleep 1

echo "running mid-vgg, naive.."
python core50_inc_finetuning.py with sI/mid-vgg-cnn-m/naive.json -c "mid-vgg, naive strategy, 10-run experiments, 2000 first batch it" 2> caffe.out > vgg_naive.out
echo "running mid-vgg, cumulative.."
python core50_inc_finetuning.py with sI/mid-vgg-cnn-m/cumulative.json -c "mid-vgg, cumulative strategy, 10-run experiments, 2000 first batch it" 2> caffe.out > vgg_cumulative.out

