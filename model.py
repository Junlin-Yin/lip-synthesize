# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import math
import os

from loadstore import loadData, restoreState, nextBatch

class Audio2Video:
    def __init__(self, path=None):
        self.args = {}
        self.args['vr']         = 0.2   # validation set ratio
        self.args['step_delay'] = 20    # step delay for LSTM (10ms per step)
        self.args['dim_hidden'] = 60    # dimension of hidden layer and cell state
        self.args['nlayers']    = 1     # number of LSTM layers
        self.args['keep_prob']  = 1     # dropout keep probability
        self.args['seq_len']    = 100   # sequence length (nframes per sequence)
        self.args['batch_size'] = 100   # batch size (nseq per batch)
        self.args['nepochs']    = 300   # number of epochs
        self.args['grad_clip']  = 10    # gradient clipping threshold
        self.args['lr']         = 1e-3  # initial learning rate
        self.args['dr']         = 1     # learning rate's decay rate
        self.args['save_freq']  = 10    # 
        
        if path is not None and os.path.exists(path):
            data = pd.read_csv(path, delimiter='\t')
            for key, value in zip(data['key'], data['value']):
                self.args[key] = value
                
    def initialize(self, name):
        '''Load data for training or validation and do some more initialization
        '''
        self.inps, self.outps = loadData(pass_id        = name,
                                         vali_rate      = self.args['vr'],
                                         step_delay     = self.args['step_delay'],
                                         seq_len        = self.args['seq_len'],
                                         repreprocess   = False)
        # In this network, self.dimin = 28, self.dimout = 20
        self.dimin, self.dimout = self.inps.shape[1], self.outps.shape[1]
        self.pass_id = name
        
        # initialize batch information
        self.nbatches, self.batch_pt = {}, {}
        for key in ['training', 'validation']:
            # count the total number of sequences we can get from the data
            nseq = sum([math.ceil(inp.shape[0] / self.args['seq_len']) for inp in self.inps[key]])
            # every batch_size sequences consist of one batch
            self.nbatches[key] = nseq // self.args['batch_size']
            # batch pointers
            self.batch_pt[key] = 0
        
    def LSTM_model(self, mode='training'):
        '''Construct LSTM network for lip synthesizing
        ### Parameters
        mode          training or validation \\
        '''
        # check mode
        if mode != 'training':
            mode = 'validation'
        
        # prepare cell and placeholders
        cell = tf.contrib.cudnn_rnn.CudnnLSTM(self.args['nlayers'], self.args['dim_hidden'])
        self.input_data = tf.placeholder(tf.float32, [None, self.args['seq_len'], self.dimin])
        self.output_data= tf.placeholder(tf.float32, [None, self.args['seq_len'], self.dimout])
        
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
        with tf.variable_scope('final'):
            final_w = tf.get_variable('final_w', [self.args['dim_hidden'], self.dimout])
            final_b = tf.get_variable('final_b', [self.dimout])
                    
        # define how to generate predicted output
        hiddens = []
        self.init_state = state = network.zero_state(self.args['batch_size'], tf.float32)
        with tf.variable_scope('LSTM'):
            for i in range(self.args['seq_len']):
                
                # avoid duplication of name
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                
                # one "clock cycle", output_i.shape=[batch_size, dim_hidden]
                hidden_i, state = network(self.input_data[:,i,:], state)
                hiddens.append(hidden_i)
        # hiddens.shape = [batch_size, dim_hidden] * seq_len
        tmp = tf.concat(hiddens, axis=1)
        # tmp.shape = [batch_size, dim_hidden*seq_len]
        hiddens = tf.reshape(tmp, [None, self.args['seq_len'], self.args['dim_hidden']])
        # hiddens.shape = [batch_size, seq_len, dim_hidden]
        output_hat = tf.nn.xw_plus_b(hiddens, final_w, final_b)
        # output_hat.shape = [batch_size, seq_len, dim_out]
        
        # define loss function (L2-norm error)
        self.loss = tf.reduce_mean(tf.squared_difference(output_hat, self.output_data))
        
        # deal with gradient and optimization
        self.lr = tf.Variable(0., trainable=False)
        tvars = tf.trainable_variables()            # all trainable variables in this model
        grads = tf.gradients(self.loss, tvars)      # partial{loss}/partial{tvars}
        # clip gradients to avoid gradient explosion
        grads, _ = tf.clip_by_global_norm(grads, self.args['grad_clip'])
        # optimizer
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        
    def train(self):
        feed_dict = {}
        init_lr, dr = self.args['lr'], self.args['dr']
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # restore model states
            startEpoch = restoreState(sess, self.pass_id)
            endEpoch = self.args['nepochs']
            for e in range(startEpoch, endEpoch):
                # in each epoch (totally <nbatches> batches per epoch)
                                
                # define learning rate of this epoch
                sess.run(tf.assign(self.lr, init_lr * (dr)**e))
                
                # feed initial cell state and hidden layer of every batch
                # when training each batch, always feed the initial state
                # because there's no relationships between two batches (just clear the memory)
                state_val = [(c.eval(), v.eval()) for (c, v) in self.init_state]
                for i, (c, v) in enumerate(self.init_state):
                    feed_dict[c], feed_dict[v] = state_val[i]
                    
                # reset batch pointers
                self.batch_pt['training'] = self.batch_pt['validation'] = 0
                    
                # define fetch list
                fetches = [self.loss, self.train_op]          
                for b in range(self.nbatches['training']):
                    # in each batch (totally <nseq> sequences per epoch)
                    x, y = nextBatch(inps       = self.inps, 
                                     outps      = self.outps, 
                                     mode       = 'training', 
                                     batch_pt   = self.batch_pt, 
                                     nbathces   = self.nbatches, 
                                     args       = self.args)
                    feed_dict[self.input_data], feed_dict[self.output_data] = x, y
                    
                    # do training in this step, get the loss
                    tloss, _ = sess.run(fetches, feed_dict)
                    reportBatch(tloss)
                    
                # at the end of each epoch, do validation
                validLoss = 0
             
                # define fetch list 
                # note that self.train_op is thrown away because we don't optimize
                # the model during validation
                fetches = [self.loss]
                for b in range(self.nbatches['validation']):
                    x, y = nextBatch(inps       = self.inps, 
                                     outps      = self.outps, 
                                     mode       = 'validation', 
                                     batch_pt   = self.batch_pt, 
                                     nbathces   = self.nbatches, 
                                     args       = self.args)
                    feed_dict[self.input_data], feed_dict[self.output_data] = x, y                  
                    vloss, = sess.run(fetches, feed_dict)
                    validLoss += vloss
                validLoss /= self.nbatches['validation']
                reportEpoch(validLoss)
        
    def test(self):
        pass
        
        
        