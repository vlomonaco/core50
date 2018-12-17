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

Simple script to create caffe filelist from the CORe50 dataset and for the
scenario II (NC - New Classes) incremental/cumulative. Objs are chosen to
maximize their difference i.e. they are taken from different class when
possible. See the docs for more information about the parameterization.
It can be run also as a standalone script.

"""

# Dependences
import shutil

# Local dependencies
from create_filelist_utils import create_filelist, load_filelist_per_sess


def create_sII_run_filelist(
    glob_file='/path/to/core50_root_dir/*/*/*',
    dest_bp='/insert/your/path/sII_inc/',
    dest_cum_bp='/insert/your/path/sII_cum/',
    all_sess=range(11),
    all_objs=range(50),
    change_labels=True,
    cumulative=True,
    train_sess=[0, 1, 3, 4, 5, 7, 8, 10],
    test_sess=[2, 6, 9],
    batch_order=[x for x in range(9)]):
    """ Given some parameters, it creates the batches filelist and
        eventually the cumulative ones. """

    # Loading the filelists devided by sessions
    filelist_all_sess = load_filelist_per_sess(glob_file)

    objs = [[]] * 9
    objs_test = []

    # Here the creations of the batches (which class to choose for
    # each of them) is **independent** by the external seed. This means that
    # the units are static throughout the runs while only their order
    # can change. This is the same as for the NI scenario where the content of
    # the batches is fixed.

    # first batch
    objs[0] = [i * 5 for i in range(10)]
    objs_test += objs[0][:]

    # inc batches
    for batch_id in range(1, 9):
        if batch_id % 2 == 0:
            objs[batch_id] = [i * 5 + int(round(batch_id / 2.0)) for i in
                              range(5, 10)]
        else:
            objs[batch_id] = [i * 5 + int(round(batch_id / 2.0)) for i in
                              range(5)]

    # the first batch stay the same regardless of the order of inc batches
    app = [[]] * 9
    app[0] = objs[0][:]
    for i, batch_idx in enumerate(batch_order[1:]):
        app[i+1] = objs[batch_idx][:]
        objs_test += objs[batch_idx][:]
    objs = app

    # Since the freezeweights strategy needs the encountered classes to be in
    # adjacent neurons the labels are changed for this task (i.e. o1 has not
    # necessary the label 0).
    if change_labels:
        label_map = {label: i for i, label in enumerate(objs_test)}
        print("Label map:", label_map)

    print("obj train:", len(objs), sorted(objs))

    for batch_id in range(9):
        create_filelist(dest_bp + "train_batch_" + str(batch_id).zfill(2),
                        filelist_all_sess, train_sess, objs[batch_id],
                        label_map=label_map)

    print("obj test:", len(objs_test), sorted(objs_test))

    create_filelist(dest_bp + "test", filelist_all_sess,
                    test_sess, objs_test, label_map=label_map)

    # create the cumulative version
    if cumulative:
        all_lines = []
        for batch_id in range(len(batch_order)):
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

    create_sII_run_filelist()
