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

Simple script to create the lmdb based on a directory in which
the patterns are divided in dirs (which names correspond to the int labels).
It can be run also as a standalone script.

"""

# Dependencies
import os


def from_filelist_to_lmdb(root, filelist_path, name, dst_lmdb,
                          compute_mean=False):
    """ Method to pass from the filelist (caffe format) to the the lmdb. """

    # Create lmdb
    print("Creating lmdb for: " + name)
    command = 'convert_imageset -encoded -shuffle -encode_type jpg ' + \
              root + ' ' + filelist_path + ' ' + \
              dst_lmdb + name + "_lmdb"

    # Print command
    os.system(command)

    if compute_mean:
        command = 'compute_image_mean ' + dst_lmdb + name + "_lmdb "\
                 + dst_lmdb + "mean_" + name + ".binaryproto"
        os.system(command)


if __name__ == "__main__":

    # Parametrization example
    from_filelist_to_lmdb(
        # src of the images (root)
        '/insert/your/path/',
        # where to save the filelist
        '/insert/your/path/',
        # base name of the lmdb
        'core50',
        # where to put it
        '/insert/your/path/'
    )
