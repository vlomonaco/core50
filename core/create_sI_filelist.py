#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2017. Vincenzo Lomonaco. All rights reserved.                  #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-04-2017                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""

Simple script to create caffe filelists from the CORe50 dataset and for the
scenario I (NI - New instances) incremental/cumulative. See the docs for more
information about the parameterization. It can be run also as a standalone
script.

"""

# Dependencies
import shutil

# Local dependencies
from create_filelist_utils import create_filelist, load_filelist_per_sess


def create_sI_run_filelist(
   glob_file='/path/to/core50_root_dir/*/*/*',
   dest_bp='/insert/your/path/sI_inc/',
   dest_cum_bp='/insert/your/path/sI_cum/',
   all_sess=range(11),
   all_objs=range(50),
   cumulative=True,
   train_sess=[0, 1, 3, 4, 5, 7, 8, 10],
   test_sess=[2, 6, 9],
   batch_order=[x for x in range(8)]):
    """ Given some parameters, it creates the batches filelist and
        eventually the cumulative ones. """

    # Adjusting the batch order
    app = [-1] * 8
    for i, batch_idx in enumerate(batch_order):
        app[i] = train_sess[batch_idx]
    train_sess = app

    # Loading all the file lists divided by session
    filelist_all_sess = load_filelist_per_sess(glob_file)

    # Create training batches filelists
    for i, sess in enumerate(train_sess):
        # create first batch
        create_filelist(dest_bp + "train_batch_" + str(i).zfill(2),
                        filelist_all_sess, [sess], all_objs)

    # Creating test filelist
    create_filelist(dest_bp + "test", filelist_all_sess, test_sess,
                    all_objs)

    # Creating the cumulative version if requested
    if cumulative:
        all_lines = []
        for batch_id in range(len(train_sess)):
            with open(dest_bp + 'train_batch_' +
                              str(batch_id).zfill(2) + '_filelist.txt',
                      'r') as f:
                all_lines += f.readlines()
            with open(dest_cum_bp + 'train_batch_' +
                              str(batch_id).zfill(2) + '_filelist.txt',
                      'w') as f:
                for line in all_lines:
                    f.write(line)
        shutil.copy(dest_bp + "test_filelist.txt", dest_cum_bp)


if __name__ == "__main__":
    """ Creating the filelists for a single run. """

    create_sI_run_filelist()
