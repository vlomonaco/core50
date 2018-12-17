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

This file contains the IncFtModel class which can be used for continuously
finetuning a DNN with caffe. Similar wrappers can be easily re-implemented
in Keras, Tensorflow, PyTorch, etc.. without much effort.

"""

# standard dependencies
import os
import caffe
import numpy as np
import pickle as pkl

# local dependencies
from convert_lmdb import from_filelist_to_lmdb


class IncFtModel:
    """
        This class can be use for incremental finetuning of a single net on
        many subsequent batches.

        Strategies can be:
            - naive: standard backpropagation update
            - copyfc8: for sII and sIII, we save and copy back the fc8 weights,
                       if clas already encountered -> avg of weights
            - copyfc8_with_reinit: for sII and sIII, like copyfc8 but with
                                   train reinit after each batch
            - freezefc8: for sII. set the lr of previous encountered class to 0
            - fromscratch: at each batch we train the model from scratch (useful
                           for cumulative experiments
    """

    def __init__(self, img_dim, conf_files, data_path, lmdb_bp, snapshots_bp,
                 first_batch_lr, lrs, num_inc_it, first_batch_it,
                 test_minibatch_size, starting_weights, stepsize,
                 weights_mult, use_lmdb=False, debug=False,
                 strategy='naive'):

        self.img_dim = img_dim
        self.conf_files = conf_files
        self.data_path = data_path
        self.lmdb_bp = lmdb_bp
        self.snapshots_bp = snapshots_bp
        self.first_batch_lr = first_batch_lr
        self.lrs = lrs
        self.num_inc_it = num_inc_it
        self.first_batch_it = first_batch_it
        self.test_minibatch_size = test_minibatch_size
        self.prev_weights = starting_weights
        self.stepsize = stepsize
        self.weights_mult = weights_mult

        self.use_lmdb = use_lmdb
        self.debug = debug
        self.strategy = strategy

        self.tot_class_num = 50
        self.num_batch_proc = 0
        self.current_weights = None
        self.encountered_class = [0] * self.tot_class_num

        caffe.set_mode_gpu()
        caffe.set_device(0)

    @staticmethod
    def change_solver_params(solver_filename, net, num_it, test_iter, stepsize,
                             snapshot_name, base_lr):
        """ This method can be used for changing the parameters in the solver
            file. """

        f = open(solver_filename, 'r')
        lines = f.readlines()
        f = open(solver_filename, 'w')
        new_lines = []
        for line in lines:
            if line.startswith('net:'):
                new_lines.append('net: "' + net + '"\n')
            elif line.startswith('test_iter:'):
                new_lines.append('test_iter: ' + str(test_iter) + '\n')
            elif line.startswith('stepsize:'):
                new_lines.append('stepsize: ' + str(stepsize) + '\n')
            elif line.startswith('max_iter:'):
                new_lines.append('max_iter: ' + str(num_it) + '\n')
            elif line.startswith('base_lr:'):
                new_lines.append('base_lr: ' + str(base_lr) + '\n')
            elif line.startswith('snapshot:'):
                new_lines.append('snapshot: ' + str(num_it) + '\n')
            elif line.startswith('snapshot_prefix:'):
                new_lines.append('snapshot_prefix: "' + snapshot_name + '"\n')
            else:
                new_lines.append(line)
        f.writelines(new_lines)

    @staticmethod
    def change_net_params(net_filename, train_data_path, test_data_path='',
                          root_folder='', shuffle='true', use_lmdb=False,
                          lr2freeze=''):
        """ This method can be used for changing the parameters in the net
            file. """

        f = open(net_filename, 'r')
        lines = f.readlines()
        f = open(net_filename, 'w')
        new_lines = []
        found = 0
        lr_mult_found = 0
        stage_test_on_train = False
        current_layer_name = ''
        for line in lines:
            app_line = line.replace(' ', '').replace("\t", '')
            if app_line.startswith('stage:"test-on-train"'):
                stage_test_on_train = True
            if app_line.startswith('source:'):
                if found == 0:
                    found += 1
                    new_lines.append('\tsource: "' + train_data_path + '"\n')
                else:
                    if stage_test_on_train:
                        new_lines.append('\tsource: "' + '/.'
                                         + train_data_path + '"\n')
                        stage_test_on_train = False
                    else:
                        if test_data_path == '':
                            new_lines.append(line)
                        else:
                            new_lines.append('\tsource: "' + test_data_path
                                             + '"\n')
            elif app_line.startswith('image_data_param') or \
                    app_line.startswith('data_param'):
                if use_lmdb:
                    new_lines.append('\tdata_param {\n')
                else:
                    new_lines.append('\timage_data_param {\n')
            elif app_line.startswith('root_folder:'):
                if not use_lmdb:
                    new_lines.append('\troot_folder: "' + root_folder + '"\n')
            elif app_line.startswith('shuffle:'):
                if not use_lmdb:
                    new_lines.append('\tshuffle: ' + shuffle + '\n')
            elif app_line.startswith('backend:'):
                if not use_lmdb:
                    continue
            elif app_line.startswith('type:"Data"') or \
                    app_line.startswith('type:"ImageData"'):
                if use_lmdb:
                    new_lines.append('type:"Data"\n')
                else:
                    new_lines.append('type:"ImageData"\n')
            elif app_line.startswith('name:'):
                s = app_line.split(':')[-1]
                current_layer_name = s.replace('"', '').strip('\n')
                lr_mult_found = 0
                new_lines.append(line)
            elif app_line.startswith('lr_mult:'):
                if lr2freeze and current_layer_name == lr2freeze:
                    new_lines.append('\tlr_mult: 0\n')
                elif lr2freeze == 'reset' and \
                        current_layer_name.startswith('mid_fc8_'):
                    lr_mult_found += 1
                    new_lines.append('\tlr_mult: ' + str(lr_mult_found) + '\n')
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        f.writelines(new_lines)
        f.close()

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""

        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def line_count(fname):
        """ Count lines in file """

        i = 0
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    @staticmethod
    def extract_classes_id(fname):
        """ Given a caffe filelist return class ids in a unique list """

        ids = []
        with open(fname) as f:
            for line in f.readlines():
                ids.append(int(line.split()[-1]))
        return list(set(ids))

    def train_batch(self, train_filelist, test_filelist, name=None):
        """ Method to train a single batch """

        # counting the patterns in the batch
        train_size = self.line_count(train_filelist)
        test_size = self.line_count(test_filelist)

        # setting the name in case is not given
        if name is None:
            # I would aspect in 'batch_filelist' something like
            #  '/a/b/../c/name_filelist.txt'
            name = '_'.join(train_filelist.split('/')[-1].split('_')[:-1])

        # setting the learning rate depending on the batch size
        if self.num_batch_proc == 0:
            lr = self.first_batch_lr
        elif train_size > 100:
            # lr = 0.00001
            lr = self.lrs[0]
        elif train_size < 30:
            lr = self.lrs[2]
        else:
            lr = self.lrs[1]

        # setting num of iterations: more for the first batch
        if self.num_batch_proc == 0:
            num_it = self.first_batch_it
        else:
            num_it = self.num_inc_it

        if self.debug:
            test_iters = (test_size / self.test_minibatch_size) + 1
        else:
            test_iters = 0

        lr2freeze = ''
        if self.strategy == 'freezefc8':
            if self.num_batch_proc > 0:
                lr2freeze = 'mid_fc8_batch' + str(self.num_batch_proc - 1)
            else:
                lr2freeze = 'reset'

        # renaming the weights file for the current batch
        self.current_weights = self.snapshots_bp + name + '_iter_' + \
                               str(num_it) + '.caffemodel.h5'

        # adjusting the solver parameters
        self.change_solver_params(self.conf_files['solver_filename'],
                                  self.conf_files['net_filename'],
                                  num_it,
                                  test_iters,
                                  self.stepsize,
                                  self.snapshots_bp + name,
                                  lr
                                  )

        # adjusting the NET parameters
        if self.use_lmdb:
            # TODO: Warning! if you want to use the lmdb format the creation 
            # of the test set is missing here!
            # For a continous learning scenario creating a different lmdb for
            # each batch could be really expensive in terms of memory.
            # However if you plan to run the same experiments more times
            # the lmdb can significally reduce the computing time. Your choice,
            # we prefer (and tested only) the image_data format below.

            lmdb_name = name + '_x' + str(self.img_dim)
            self.change_net_params(self.conf_files['net_filename'],
                                   self.lmdb_bp + lmdb_name + '_lmdb',
                                   lr2freeze=lr2freeze
                                   )
            # creating the lmdb in case it not exists yet
            lmdb_path = self.lmdb_bp + lmdb_name + '_lmdb'
            if not os.path.exists(lmdb_path):
                from_filelist_to_lmdb(self.data_path, train_filelist, lmdb_name,
                                      self.lmdb_bp, compute_mean=True)

        else:
            # in case we use the image_data format
            self.change_net_params(self.conf_files['net_filename'],
                                   train_filelist,
                                   test_data_path=test_filelist,
                                   root_folder=self.data_path,
                                   lr2freeze=lr2freeze
                                   )

        # setting and running the solver
        solver = caffe.get_solver(str(self.conf_files['solver_filename']))

        # keep init weights
        init_weights = {}
        loaded_weights = {}
        if self.strategy == 'copyfc8_with_reinit' and self.num_batch_proc == 0:
            for clas in range(self.tot_class_num):
                init_weights[clas] = \
                    (solver.net.params['mid_fc8'][0].data[clas],
                     solver.net.params['mid_fc8'][1].data[clas])

        # prev_weights could be null in the first batch
        if self.prev_weights != '':
            solver.net.copy_from(str(self.prev_weights))

        # load previous weights
        if (self.strategy == 'copyfc8' or
            self.strategy == 'copyfc8_with_reinit') and \
           self.num_batch_proc > 0:
            # here load prev weigths fc8
            with open(self.snapshots_bp + 'weigths_fc8.pkl', 'rb') as f:
                init_weights, loaded_weights = pkl.load(f)
                # reinit weights if 'copyfc8_with_reinit'
                if self.strategy == 'copyfc8_with_reinit':
                    for clas, (weights, bias) in init_weights.items():
                        solver.net.params['mid_fc8'][0].data[clas] = weights
                        solver.net.params['mid_fc8'][1].data[clas] = bias

        # run the training
        solver.step(num_it)

        # weights stats for the training net
        tr_new_weights = []
        tr_new_biases = []
        tr_other_weights = []
        tr_other_biases = []

        # saving train weights stats
        if self.strategy == 'freezefc8':
            num_clas = solver.net.params[
                'mid_fc8_batch'+str(self.num_batch_proc)][0].data.shape[0]
            for clas in range(num_clas):
                tr_new_weights.append(
                    solver.net.params[
                        'mid_fc8_batch'+str(self.num_batch_proc)][0].data[clas]
                )
                tr_new_biases.append(
                    solver.net.params[
                        'mid_fc8_batch'+str(self.num_batch_proc)][1].data[clas]
                )
            # other batches
            for i in range(9):
                if i != self.num_batch_proc:
                    num_clas = solver.net.params[
                        'mid_fc8_batch' + str(i)][0].data.shape[0]
                    for clas in range(num_clas):
                        tr_other_weights.append(
                            solver.net.params[
                                'mid_fc8_batch' + str(i)][0].data[
                                clas]
                        )
                        tr_other_biases.append(
                            solver.net.params[
                                'mid_fc8_batch' + str(i)][1].data[
                                clas]
                        )
        else:
            for clas in range(self.tot_class_num):
                if clas in self.extract_classes_id(train_filelist):
                    tr_new_weights.append(
                        solver.net.params['mid_fc8'][0].data[clas])
                    tr_new_biases.append(
                        solver.net.params['mid_fc8'][1].data[clas])
                else:
                    tr_other_weights.append(
                        solver.net.params['mid_fc8'][0].data[clas])
                    tr_other_biases.append(
                        solver.net.params['mid_fc8'][1].data[clas])

        # delete training net from memory
        del solver

        # running the test net
        test_net = caffe.Net(str(self.conf_files['net_filename']),
                             str(self.current_weights), caffe.TEST)

        # in case we are not in the first batch and
        # the strategy is 'copyfc8...'
        if (self.strategy == 'copyfc8' or
            self.strategy == 'copyfc8_with_reinit') and \
            self.num_batch_proc > 0:

            # here load prev weigths fc8
            for clas, (weights, bias) in loaded_weights.items():
                if self.num_batch_proc == 1:
                    weights *= self.weights_mult

                # we average the weights if we encountered the same class
                # in the past and in the current batch
                count = self.encountered_class[clas]
                if clas in self.extract_classes_id(train_filelist) \
                        and count > 0:
                    weights = (weights * count +
                               test_net.params['mid_fc8'][0].data[clas]) / \
                              (count + 1)
                    bias = (bias * count +
                            test_net.params['mid_fc8'][1].data[clas]) / \
                           (count + 1)

                test_net.params['mid_fc8'][0].data[clas] = weights
                test_net.params['mid_fc8'][1].data[clas] = bias

        test_iters = (test_size // self.test_minibatch_size + 1)

        hits_per_class = [0 for i in range(self.tot_class_num)]
        pattern_per_class = [0 for i in range(self.tot_class_num)]

        # computing the accuracy
        for it in range(test_iters):
            blobs = test_net.forward(blobs=['label', 'mid_fc8'])

            labels = blobs['label']
            labels = labels.astype(int)

            for label in labels:
                pattern_per_class[label] += 1

            probs_matrix = blobs['mid_fc8']
            for i, probs in enumerate(probs_matrix):
                pred_label = np.argmax(probs)
                if pred_label == labels[i]:
                    hits_per_class[pred_label] += 1

        accs = np.asarray(hits_per_class) / \
               np.asarray(pattern_per_class).astype(float)

        acc = np.sum(hits_per_class) / \
              np.sum(pattern_per_class).astype(float)

        # extracting from the net the weights to save in case of 'copyfc8'
        if self.strategy == 'copyfc8' or \
           self.strategy == 'copyfc8_with_reinit':
            weights_to_save = {}
            class_to_save = self.extract_classes_id(train_filelist)
            print(class_to_save)
            if loaded_weights:
                class_to_save += loaded_weights.keys()
            print(class_to_save)
            # we took only them belonging to encountered classes
            for clas in class_to_save:
                    weights_to_save[clas] = \
                        (test_net.params['mid_fc8'][0].data[clas],
                         test_net.params['mid_fc8'][1].data[clas])

            # saving weights
            with open(self.snapshots_bp + 'weigths_fc8.pkl', 'wb') as f:
                pkl.dump((init_weights, weights_to_save), f)

        # weights stats for the test net
        te_new_weights = []
        te_new_biases = []
        te_other_weights = []
        te_other_biases = []

        # saving test weights stats
        if self.strategy == 'freezefc8':
            num_clas = test_net.params[
                'mid_fc8_batch'+str(self.num_batch_proc)][0].data.shape[0]
            for clas in range(num_clas):
                te_new_weights.append(
                    test_net.params[
                        'mid_fc8_batch'+str(self.num_batch_proc)][0].data[clas]
                )
                te_new_biases.append(
                    test_net.params[
                        'mid_fc8_batch'+str(self.num_batch_proc)][1].data[clas]
                )
            # other batches
            for i in range(9):
                if i != self.num_batch_proc:
                    num_clas = test_net.params[
                        'mid_fc8_batch' + str(i)][0].data.shape[0]
                    for clas in range(num_clas):
                        te_other_weights.append(
                            test_net.params[
                                'mid_fc8_batch' + str(i)][0].data[
                                clas]
                        )
                        te_other_biases.append(
                            test_net.params[
                                'mid_fc8_batch' + str(i)][1].data[
                                clas]
                        )
        else:
            for clas in range(self.tot_class_num):
                if clas in self.extract_classes_id(train_filelist):
                    te_new_weights.append(
                        test_net.params['mid_fc8'][0].data[clas])
                    te_new_biases.append(
                        test_net.params['mid_fc8'][1].data[clas])
                else:
                    te_other_weights.append(
                        test_net.params['mid_fc8'][0].data[clas])
                    te_other_biases.append(
                        test_net.params['mid_fc8'][1].data[clas])

        tr_stats = "\n[Train-net] avg. new weights: " + \
                   str(np.mean(tr_new_weights)) + \
                   "\n[Train-net] avg. new biases: " + \
                   str(np.mean(tr_new_biases)) + \
                   "\n[Train-net] avg. other weights: " + \
                   str(np.mean(tr_other_weights)) + \
                   "\n[Train-net] avg. other biases: " + \
                   str(np.mean(tr_other_biases)) + \
                   "\n[Train-net] tot weights avg. : " + \
                   str(np.mean(tr_new_weights + tr_other_weights))

        te_stats = "\n[Test-net] avg. new weights: " + \
                   str(np.mean(te_new_weights)) + \
                   "\n[Test-net] avg. new biases: " + \
                   str(np.mean(te_new_biases)) + \
                   "\n[Test-net] avg. other weights: " + \
                   str(np.mean(te_other_weights)) + \
                   "\n[Test-net] avg. other biases: " + \
                   str(np.mean(te_other_biases))

        s = "Batch: " + str(self.num_batch_proc) + ", name: " + name + \
            " (size " + str(train_size) + '), Accuracy: ' + str(acc) + \
            tr_stats + te_stats

        del test_net

        # updating the sate for the next iter
        self.num_batch_proc += 1
        if self.strategy != 'fromscratch':
            self.prev_weights = self.current_weights

        for clas in self.extract_classes_id(train_filelist):
            self.encountered_class[clas] += 1

        return s, acc, accs
