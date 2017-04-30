#!/usr/bin/env bash

echo "running mid-caffenet, naive.."
python core50_inc_finetuning.py with sIII/mid-caffeNet/naive.json -c "sIII 1.0: mid-caffenet, naive strategy, 10-run experiments" 2> caffe.out > sIII_caffenet_naive.out
sleep 1
echo "running mid-caffenet, copyweights_with_reinit.."
python core50_inc_finetuning.py with sIII/mid-caffeNet/copyweights_with_reinit.json -c "sIII: mid-caffenet, copyweights_with_reinit strategy, 10-run experiments" 2> caffe.out > sIII_caffenet_cp_wre.out
sleep 1
echo "running mid-caffenet, cumulative.."
python core50_inc_finetuning.py with sIII/mid-caffeNet/cumulative.json -c "sIII 1.0: mid-caffenet (full), cumulative strategy, 10-run experiments" 2> caffe.out > sIII_caffenet_cum.out
sleep 1

echo "running mid-vgg, naive.."
python core50_inc_finetuning.py with sIII/mid-vgg/naive.json -c "sIII 1.0: mid-vgg (full), naive strategy, 10-run experiments" 2> caffe.out > sIII_vgg_naive.out
sleep 1
echo "running mid-vgg, copyweights_with_reinit.."
python core50_inc_finetuning.py with sIII/mid-vgg/copyweights_with_reinit.json -c "sIII 1.0: mid-vgg, copyweights_with_reinit strategy, 10-run experiments" 2> caffe.out > sIII_vgg_cp_wre.out
sleep 1
echo "running mid-vgg, cumulative.."
python core50_inc_finetuning.py with sIII/mid-vgg/cumulative.json -c "sIII 1.0: mid-vgg (full), cumulative strategy, 10-run experiments" 2> caffe.out > sIII_vgg_cum.out
sleep 1

