# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import os

from loadstore.py import *

class Audio2Video:
    def __init__(self, path=None):
        self.args = {}
        self.args['vr']         = 0.2
        self.args['step_delay'] = 100
        self.args['dim_hidden'] = 60
        self.args['nlayers']    = 1
        self.args['keep_prob']  = 1
        self.args['seq_len']    = 100
        self.args['batch_size'] = 100
        self.args['grad_clip']  = 10
        self.args['lr']         = 1e-3
        self.args['dr']         = 1
        self.args['save_freq']  = 10
        
        if path is not None and os.path.exists(path):
            data = pd.read_csv(path, delimiter='\t')
            for key, value in zip(data['key'], data['value']):
                self.args[key] = value
                
    def load_data(self, name):
        pass_id, vali_rate=0.2, step_delay=20, seq_len=100, repreprocess=False
        self.inps, self.outps = loadData(pass_id        = name,
                                         vali_rate      = self.args['vr'],
                                         step_delay     = self.args['step_delay'],
                                         seq_len        = self.args['seq_len'],
                                         repreprocess   = False)
        # In this network, self.dimin = 28, self.dimout = 20
        self.dimin, self.dimout = self.inps.shape[1], self.outps.shape[1]

    def LSTM_model(self, mode='training'):
        '''Construct LSTM network for lip synthesizing
        ### Parameters
        mode          training or validation \\

        ### Return values
        (bool)        whether build LSTM network successfully
        '''
        # check mode
        if mode == 'validation':
            pass
        elif mode != 'training':
            print('Invalid mode. Please choose \'training\' or \'validation\'.')
            return False
        
        # prepare cell and placeholders
        cell = tf.contrib.cudnn_rnn.CudnnLSTM(self.args['nlayers'], self.args['dim_hidden'])
        input_data = tf.placeholder(tf.float32, [None, self.args['seq_len'], self.dimin])
        output_data= tf.placeholder(tf.float32, [None, self.args['seq_len'], self.dimout])
        
        # add dropout wrapper & define multilayer network
        kp = self.args['keep_prob']
        total_layers = self.args['nlayers']
        inner_layers = total_layers - 1
        if mode == 'training' and kp < 1:
            cell0 = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=kp)
            cell1 = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=kp, output_keep_prob=kp)
            network = tf.nn.rnn_cell.MultiRNNCell([cell0]*inner_layers + [cell1])
        else:
            # Note that dropout is not used in validation stage
            network = tf.nn.rnn_cell.MultiRNNCell([cell]*total_layers)       
        # add final weight matrix and final bias vector as said in the paper
        with tf.variable_scope('final') as final:
            final_w = tf.get_variable('final_w', [self.args['dim_hidden'], self.dimout])
            final_b = tf.get_variable('final_b', [self.dimout])
                    
        # define how to generate predicted output
        hiddens = []
        state = network.zero_state(self.args['batch_size'], tf.float32)
        with tf.variable_scope('LSTM'):
            for i in range(self.args['seq_len']):
                
                # avoid duplication of name
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                
                # one "clock cycle", output_i.shape=[batch_size, dim_hidden]
                hidden_i, state = network(inputs[:,i,:], state)
                hiddens.append(hidden_i)
        # hiddens.shape = [batch_size, dim_hidden] * seq_len
        tmp = tf.concat(hiddens, axis=1)
        # tmp.shape = [batch_size, dim_hidden*seq_len]
        hiddens = tf.reshape(tmp, [None, self.args['seq_len'], self.args['dim_hidden']])
        # hiddens.shape = [batch_size, seq_len, dim_hidden]
        output_hat = tf.nn.xw_plus_b(hiddens, final_w, final_b)
        # output_hat.shape = [batch_size, seq_len, dim_out]
        
        # define loss function
        
        # deal with gradient and optimization
        
        
        