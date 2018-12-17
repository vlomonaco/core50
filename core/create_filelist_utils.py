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

This file contains some utility and common methods for creating the filelists
for the different scenarios and from the CORe50 dataset.

"""

# Depenencies
from glob import glob


def scale_labes(orig_list):
    """ This method given a list of number returns a dict with a scaled
        conversion. """

    convdict = {}
    ordered_list = sorted(orig_list)
    for i, num in enumerate(ordered_list):
        convdict[num] = i

    return convdict


def load_filelist_per_sess(glob_file):
    """ This method returns a dictionary {batch_name : (path, label)}. """

    batches = {}
    for filepath in sorted(glob(glob_file)):
        batch_name, label, filename = filepath.split('/')[-3:]
        batch_id = int(batch_name[1:])-1
        if batch_id not in batches.keys():
            batches[batch_id] = []
        batches[batch_id].append((filename, int(label[1:])-1))

    return batches


def create_filelist(filelist_name, batches, sess, objs, label_map=None):
    """ This method create a single filelist given a list of objs and sess. """

    with open(filelist_name + "_filelist.txt", 'w') as f:
        for batch_id, patterns in batches.items():
            for filename, label in patterns:
                if batch_id in sess and label in objs:
                    if label_map:
                        new_label = label_map[label]
                    else:
                        new_label = label
                    f.write('s' + str(batch_id+1) + '/o' + str(label+1) + '/' +
                            filename + ' ' + str(new_label) + '\n')

