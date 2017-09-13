#!/usr/bin/env bash

# Running NC experiments
echo "running mid-caffenet, naive.."
python core/core50_inc_finetuning.py with confs/sII/mid-caffeNet/naive.json -c "sII (final): mid-caffenet (full), naive strategy, 10-run experiments" 2> data/logs/caffe.out > data/logs/sII_caffenet_naive.out
sleep 1
echo "running mid-caffenet, copyweights_with_reinit.."
python core/core50_inc_finetuning.py with confs/sII/mid-caffeNet/copyweights_with_reinit.json -c "mid-caffenet, copyweights_with_reinit strategy, 10-run experiments" 2> data/logs/affe.out > data/logs/sII_caffenet_cp_wre.out
sleep 1
echo "running mid-caffenet, cumulative.."
python core/core50_inc_finetuning.py with confs/sII/mid-caffeNet/cumulative.json -c "sII 1.0: mid-caffenet (full), cumulative strategy, 10-run experiments" 2> data/logs/caffe.out > data/logs/sII_caffenet_cum.out
sleep 1

echo "running mid-vgg, naive.."
python core/core50_inc_finetuning.py with confs/sII/mid-vgg/naive.json -c "sII (final): mid-vgg, naive strategy, 10-run experiments" 2> data/logs/caffe.out > data/logs/sII_vgg_naive.out
sleep 1
echo "running mid-vgg, copyweights_with_reinit.."
python core/core50_inc_finetuning.py with confs/sII/mid-vgg/copyweights_with_reinit.json -c "sII (final): mid-vgg, copyweights_with_reinit strategy, 10-run experiments" 2> data/logs/caffe.out > data/logs/sII_vgg_cp_wre.out
sleep 1
echo "running mid-vgg, cumulative.."
python core/core50_inc_finetuning.py with confs/sII/mid-vgg/cumulative.json -c "sII 1.0: mid-vgg (full), cumulative strategy, 10-run experiments" 2> data/logs/caffe.out > data/logs/sII_vgg_cum.out
sleep 1

echo "running mid-caffenet, copyweights.."
python core/core50_inc_finetuning.py with confs/sII/mid-caffeNet/copyweights.json -c "mid-caffenet (conv5fc8), copyweights strategy, 10-run experiments" 2> data/logs/caffe.out > data/logs/sII_caffenet_cp.out
sleep 1
echo "running mid-caffenet, freezeweights.."
python core/core50_inc_finetuning.py with confs/sII/mid-caffeNet/freezeweights.json -c "sII 0.0: mid-caffenet (multifc8), freezefc8 strategy, 10-run experiments" 2> data/logs/caffe.out > data/logs/sII_caffenet_freezefc8.out
sleep 1
