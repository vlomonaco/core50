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

This sacred script can be used for recording all the incremental experiments
on CORe50 on an external DB and output file. Sacred is also very useful in order
to guarantee the reproducibility of  the experiments and to keep track of them
while they are still running.

This main script can be used independently of the underline Deep Learning
framework and model implementation. It orchestrates each experiment performing:

    1. The connection and data transfer to the sacred observer.
    2. On-the-fly creation of the batches train and test filelists for each run.
    3. The multi-run training and testing with the IncFtModel wrapper.

"""

# Sacred dependecies
from sacred import Experiment
from sacred.observers import MongoObserver

# Other dependencies
import numpy as np
import os

# Local dependencies
from inc_finetuning import IncFtModel
from create_sI_filelist import create_sI_run_filelist
from create_sII_filelist import create_sII_run_filelist
from create_sIII_filelist import create_sIII_run_filelist

# Add the Ingredient while creating the experiment
ex = Experiment('core50 incremental finetuning')

# We add the observer (if you don't have a configured DB
# then simply comment the line below line).
# ex.observers.append(MongoObserver.create(db_name='experiments_db'))


@ex.config
def cfg():
    """ Default configuration parameters. Overwritten by specific exps
        configurations. See the docs for more information about type and
        semantics of each parameter. """

    name = 'core50 incremental finetuning - sI'
    scenario = 1
    img_dim = 128
    data_path = '/insert/your/path/'
    lmdb_bp = '/insert/your/path/'
    filelist_bp = '/insert/your/path/sI'
    conf_bp = '/insert/your/path/'
    snapshots_bp = '/insert/your/path/'
    starting_weights = '/insert/your/path/bvlc_reference_caffenet.caffemodel'

    batches_num = 8
    num_runs = 10

    conf_files = {
        'solver_filename': conf_bp + 'inc_solver.prototxt',
        'net_filename': conf_bp + 'inc_train_val.prototxt'
    }

    # first batch learning rate
    first_batch_lr = 0.001

    # in order of importance
    lrs = [
        0.001,
        0.00005,
        0.00001
    ]

    num_inc_it = 100
    first_batch_it = 2000
    test_minibatch_size = 100
    stepsize = first_batch_it
    weights_mult = 3

    # naive if not specified
    strategy = 'fromscratch'
    fixed_test_set = True

    # random sees
    seed = 1


@ex.automain
def main(img_dim, conf_files, data_path, lmdb_bp, filelist_bp, snapshots_bp,
         first_batch_lr, lrs, num_inc_it, first_batch_it, test_minibatch_size,
         batches_num, starting_weights, stepsize, weights_mult, strategy,
         fixed_test_set, num_runs, seed, scenario):
    """ Main script which create the train/test filelists for each batch
        and run on-the-fly. Then it trains the model continuously via the
        IncFtModel class. """

    # setting the seed for reproducibility
    batches_idx = [str(x).zfill(2) for x in range(batches_num)]
    batch_order = [x for x in range(batches_num)]
    np.random.seed(seed)

    # For each run we operate sequentially
    for run in range(num_runs):
        run = str(run)
        ex.info[run] = {}

        # In scenario 2 (NC) and 3 (NIC) the first batch stay fixed
        # otherwise we shuffle everything
        if scenario != 1:
            inc_batch_order = batch_order[1:]
            np.random.shuffle(inc_batch_order)
            batch_order = [0] + inc_batch_order
        else:
            np.random.shuffle(batch_order)

        print("----------- Run " + run + " -----------")
        print("batches order: ", batch_order)
        print("-----------------------------")

        # Setting the meta filelists parameters
        path = filelist_bp + '_inc/run' + run + '/'
        cpath = filelist_bp + '_cum/run' + run + '/'

        for x in [path, cpath]:
            if not os.path.exists(x):
                os.makedirs(x)

        if strategy == 'fromscratch':
            cum = True
            curr_filelist_bp = cpath
        else:
            cum = False
            curr_filelist_bp = path

        # Actually creating the filelists depending on
        # the scenario sI (NI), sII (NC), sIII (NIC)
        if scenario == 1:
            create_sI_run_filelist(
                dest_bp=path,
                dest_cum_bp=cpath,
                cumulative=cum,
                batch_order=batch_order
            )
        elif scenario == 2:
            create_sII_run_filelist(
                dest_bp=path,
                dest_cum_bp=cpath,
                cumulative=cum,
                batch_order=batch_order
            )
        elif scenario == 3:
            create_sIII_run_filelist(
                dest_bp=path,
                dest_cum_bp=cpath,
                cumulative=cum,
                batch_order=batch_order
            )
        else:
            print("Error: scenario not known.")

        # Create object incFTModel
        inc_model = IncFtModel(img_dim, conf_files, data_path, lmdb_bp,
                               snapshots_bp, first_batch_lr, lrs, num_inc_it,
                               first_batch_it, test_minibatch_size,
                               starting_weights, stepsize, weights_mult,
                               use_lmdb=False, debug=False, strategy=strategy)

        # Now we can train it incrementally
        for idx in batches_idx:
            if fixed_test_set:
                test_filelist = curr_filelist_bp + 'test_filelist.txt'
            else:
                test_filelist = curr_filelist_bp + 'test_batch_' + \
                                idx + "_filelist.txt"

            s, acc, accs = inc_model.train_batch(
                curr_filelist_bp + 'train_batch_' + idx
                + "_filelist.txt", test_filelist)

            # Printing batch results
            print(s)
            print("Acc. per class:")
            for i, single_acc in enumerate(accs):
                print(str(i) + ': ' + str(round(single_acc, 3)).ljust(
                    10) + "\t", end="")
                if (i + 1) % 5 == 0:
                    print("")
            print("----------------------------")

            # Saving them on the DB
            if 'inc_accuracy' not in ex.info[run].keys():
                ex.info[run]['inc_accuracy'] = []
            if 'inc_accuracy per class' not in ex.info[run].keys():
                ex.info[run]['inc_accuracy per class'] = []
            ex.info[run]['inc_accuracy'].append(acc)
            ex.info[run]['inc_accuracy per class'].append(accs.tolist())
