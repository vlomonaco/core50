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
scenario III (NIC - New Instances and Classes) incremental/cumulative. See the
docs for more information about the parameterization. It can be run also as a
standalone script.

"""

# Dependencies
import shutil
import numpy as np

# Local dependencies
from create_filelist_utils import create_filelist, load_filelist_per_sess

def create_sIII_run_filelist(
    glob_file='data/core50_128x128/*/*/*',
    dest_bp='/insert/your/path/sIII_inc/',
    dest_cum_bp='/insert/your/path/sIII_cum/',
    all_sess=range(11),
    all_objs=range(50),
    cumulative=True,
    batch_order=[x for x in range(79)]):
    """ Given some parameters, it creates the batches filelist and
        eventually the cumulative ones. """

    # Here the creations of the units (which obj in which sess)
    # is **independent** by the external seed. This means that the units are
    # static throughout the runs while only their order can change. This is
    # the same as for the NI and NC scenarios where the batches are fixed.
    rnd_state = np.random.get_state()
    np.random.seed(0)

    filelist_all_sess = load_filelist_per_sess(glob_file)
    train_sess = [0, 1, 3, 4, 5, 7, 8, 10]
    test_sess = [2, 6, 9]

    # Selecting the five objs for batch
    first_ten_objs = [i * 5 for i in range(10)]
    objs_after_first_b = []
    for id in all_objs:
        if id not in first_ten_objs:
            objs_after_first_b.append(id)

    np.random.shuffle(objs_after_first_b)
    objs_per_batch = np.reshape(objs_after_first_b, (8, 5))
    objs_per_batch = [row for row in objs_per_batch]

    # Creating units for classes after first batch
    units = []
    for sess in train_sess:
        for objs_id in objs_per_batch:
            units.append((sess, objs_id))

    # Creating for the first 10 classes split in two groups
    for sess in train_sess[1:]:
        units.append((sess, first_ten_objs[:5]))
        units.append((sess, first_ten_objs[5:]))

    # Suffling units
    np.random.shuffle(units)

    print("Number of incremental units: ", len(units))
    print("----- Unit details (sess, objs) ------")
    for unit in units:
        print(unit)

    # Creating first batch
    create_filelist(dest_bp + "train_batch_00", filelist_all_sess, [0],
                    first_ten_objs)

    # Creating test
    create_filelist(dest_bp + "test", filelist_all_sess, test_sess,
                    all_objs)

    # Reordering incremental units based on batch order
    new_units = [[]] * 78
    for i, id in enumerate(batch_order[1:]):
        new_units[i] = units[id-1]
    units = new_units

    # Creating incremental batches with units
    for batch_id, unit in enumerate(units):
        create_filelist(dest_bp + "train_batch_" +
                        str(batch_id + 1).zfill(2),
                        filelist_all_sess, [unit[0]], unit[1])

    # Creating the cumulative version
    if cumulative:
        all_lines = []
        for batch_id in range(len(units) + 1):
            with open(dest_bp + 'train_batch_' +
                      str(batch_id).zfill(2) + '_filelist.txt', 'r') as f:
                all_lines += f.readlines()
            with open(dest_cum_bp + 'train_batch_' +
                      str(batch_id).zfill(2) + '_filelist.txt', 'w') as f:
                for line in all_lines:
                    f.write(line)
        shutil.copy(dest_bp + "test_filelist.txt", dest_cum_bp)

    # Resetting previous rnd state
    np.random.set_state(rnd_state)


if __name__ == "__main__":
    """ Creating the filelists for a single run. """

    create_sIII_run_filelist()
