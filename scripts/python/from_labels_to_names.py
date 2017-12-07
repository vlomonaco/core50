#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2017. Vincenzo Lomonaco. All rights reserved.                  #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 6-12-2017                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" Script for getting the label: object name mapping given a run and a
    scenario. """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pickle as pkl


def get_mapping(scen='ni', run=0):
    """ given a scenario and a run return a dict label2names. """

    print("Loading class names...")
    with open('core50_class_names.txt', 'r') as f:
        obj2name = {'o'+str(i+1): name.strip() for i, name in enumerate(f)}

    print("Loading paths...")
    with open('paths.pkl', 'rb') as f:
        paths = pkl.load(f)

    print("Loading paths...")
    with open('labels.pkl', 'rb') as f:
        labels = pkl.load(f)

    print("Loading LUP...")
    with open('LUP.pkl', 'rb') as f:
        LUP = pkl.load(f)

    names = []
    label2name = {}
    batch = -1  # the last one is the test in LUP.pkl and labels.pkl

    for idx in LUP[scen][run][batch]:
        names.append(obj2name[paths[idx].split('/')[-2]])

    for name, label in zip(names, labels[scen][run][batch]):
        if label not in label2name:
            label2name[label] = name

    return label2name


if __name__ == '__main__':

    print(get_mapping(scen='ni', run=0))







