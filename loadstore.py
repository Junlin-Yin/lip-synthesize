# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import math
import random
import bisect

video_dir = 'video/'        # video raw features
audio_dir = 'audio/'        # audio raw features
save_dir  = 'save/'         # network saved data

video_fps = 30

def loadRawFeature(vali_rate):
    inps  = {'training':[], 'validation':[]}
    outps = {'training':[], 'validation':[]}
    
    vsubdirs = os.listdir(video_dir)
    vsubdirs = [x[:-4] for x in vsubdirs]
    unqlinks, unqcnts = np.unique(vsubdirs, return_counts=True)
    
    for link, cnt in zip(unqlinks, unqcnts):
        
        # extract audio features
        afeatures = np.load(os.path.join(audio_dir, link+'.npy'))   # (N, 15)
        audio           = afeatures[ :, :-1]                        # (N, 14)
        audio_diff      = afeatures[1:, :-1] - afeatures[:-1, :-1]  # (N-1, 14)
        audio_timestamp = afeatures[ :,  -1]
        
        vsubdir_list = [link + "}}" + str(100+i)[1:3] + '/' for i in range(cnt)]
        for vsubdir in vsubdir_list:
            
            # in each video fragments
            vfeatures = np.load(os.path.join(video_dir+vsubdir, 'fidsCoeff.npy'))
            with open(os.path.join(video_dir+vsubdir, 'startframe.txt')) as f:
                vstartfr = int(f.read())
            with open(os.path.join(video_dir+vsubdir, 'nframe.txt')) as f:
                vnfr = int(f.read())
                
            # slice corresponding audio fragment
            astartfr = bisect.bisect_left(audio_timestamp, vstartfr/video_fps)
            aendfr = bisect.bisect_right(audio_timestamp, (vstartfr+vnfr-1)/video_fps)
            inp = np.hstack((audio[astartfr:aendfr], audio_diff[astartfr:aendfr]))
            
            # interpolating video fragment
            outp= np.zeros((aendfr-astartfr, vfeatures.shape[1]))
            for i in range(outp.shape[0]):
                vfr = audio_timestamp[i+astartfr] * video_fps
                left_vfr = int(vfr)
                right_vfr = min(left_vfr+1, vfeatures.shape[0])
                alpha = vfr - left_vfr
                outp[i] = (1-alpha) * vfeatures[left_vfr] + alpha * vfeatures[right_vfr]
        
            # deciding training or validation
            key = 'training' if random.random() > vali_rate else 'validation'
            inps[key].append(inp)
            outps[key].append(outp)
            
    return inps, outps

def normalizeData(inps, outps):
    tinps, toutps = inps['training'], outps['training']
    vinps, voutps = inps['validation'], outps['validation']
    
    means, stds = [0]*2, [0]*2
    for idx, data in enumerate((tinps, toutps)):
        # idx = 0: inps
        # idx = 1: outps
        merged_data = np.vstack(data)        # merged_data.shape = (?, F)
        means[idx] = np.mean(merged_data, axis=0)
        stds[idx]  = np.std(merged_data, axis=0)
        
        for i in range(len(data)):
            data[i] = (data[i] - means[idx]) / stds[idx]
    
    for idx, data in enumerate((vinps, voutps)):
        # idx = 0: inps
        # idx = 1: outps
        for i in range(len(data)):
            data[i] = (data[i] - means[idx]) / stds[idx]

    norm_inps  = {'training':tinps, 'validation':vinps}
    norm_outps = {'training':toutps, 'validation':voutps}
    return norm_inps, norm_outps, means, stds

def loadData(pass_id, args, preprocess=False):
    '''load input and output from save_dir, or from video_dir and audio_dir
    if preprocessing is needed.
    ### Parameters
    pass_id        (str)  name of this pass, including training and testing \\
    args           (dict) argument dictionary containing vr, step_delay, seq_len, etc. \\
    preprocess     (bool) whether preprocessing is needed \\
    ### Return Values
    new_inps       (dict) processed input data \\
    new_outps      (dict) processed output data
    '''
    vali_rate  = args['vr']
    step_delay = args['step_delay']
    seq_len    = args['seq_len']
    
    if preprocess or os.path.exists(save_dir + pass_id + '/') == False:
        # extract raw features from video_dir and audio_dir
        inps, outps = loadRawFeature(vali_rate)
        
        # normalize them
        inps, outps, means, stds = normalizeData(inps, outps)
        
        # save them to model/pass_id/inout_data.npz
        if os.path.exists(save_dir + pass_id + '/') == False:
            os.mkdir(save_dir + pass_id + '/')
            
        np.savez(save_dir+pass_id+'/inout_data.npz', 
                 inps=inps,          outps=outps, 
                 inps_mean=means[0], outps_mean=means[1],
                 inps_std=stds[0],   outps_std=stds[1])
        
    # extract inputs and outputs from model/passId/inout_data.npz
    inout_data = np.load(save_dir+pass_id+'/inout_data.npz')
    inps, outps = inout_data['inps'], inout_data['outps']
    
    # deal with step delay
    new_inps, new_outps = {'training':[], 'validation':[]}
    for key in new_inps.keys():
        for inp, outp in zip(inps[key], outps[key]):
            # throw away those which have less than <step_delay+seq_len> frames
            if inp.shape[0] - step_delay >= seq_len:
                new_inps[key].append(np.copy(inp[step_delay:, :]))
                new_outps[key].append(np.copy(outp[:, :-step_delay if step_delay > 0 else None]))
    
    np.savez(save_dir+pass_id+'/inout_data_delay.npz',
             inps=new_inps, outps=new_outps, step_delay=step_delay)
    return new_inps, new_outps

def nextBatch(inps, outps, mode, batch_pt, nbatches, args):
    '''fetch next batch of inputs and outputs and update batch pointer
    ### Parameters
    inps     (dict) input dictionary which contains two list \\
    outps    (dict) output dictionary which contains two list \\
    mode     (str)  training or validation \\
    batch_pt (dict) batch pointer dictionary which contains two modes \\
    nbatches (dict) number-of-batch dictionary containing two modes \\
    args     (dict) argument dictionary containing batch_size, seq_len, etc. \\
    ### Return Values
    x        (list of ndarrays) input batch \\
    y        (list of ndarrays) output batch
    '''
    batch_size = args['batch_size']
    seq_len    = args['seq_len']
    
    x, y = [], []
    for i in range(batch_size):
        idx = batch_pt[mode]
        inp, outp = inps[mode][idx], outps[mode][idx]

        # formatting each sequence randomly
        startfr = random.randint([0, inp.shape[0]-seq_len-1])
        x.append(np.copy(inp[startfr : startfr+seq_len]))
        y.append(np.copy(outp[startfr: startfr+seq_len]))
        
        # determine whether to move to next fragment of data
        nseq = math.ceil(inp.shape[0] / seq_len)
        if random.random() < (1 / nseq):
            # let X be the number of sequences extracted from inp,
            # then E(X) = nseq
            batch_pt[mode] = (batch_pt[mode] + 1) % nbatches[mode]
        
    return x, y

def restoreState(sess, pass_id):
    '''restore network states
    ### Parameters
    sess       (Session) \\
    pass_id    (str)    name of this pass, including training and testing
    ### Return Values
    startEpoch (int)    we should continue training from this epoch 
    '''
    # define checkpoint saver
    saver = tf.train.Saver(tf.global_variables())

    # restore start epoch number and global variables
    save_pass_dir = save_dir + pass_id + '/'
    last_ckpt = tf.train.latest_checkpoint(save_pass_dir)
    if last_ckpt is None:
        # no ckpt file yet
        startEpoch = 0
    else:
        startEpoch = int(last_ckpt.split('-')[-1])
        saver.restore(sess, save_pass_dir)
    
    return startEpoch
        
    

if __name__ == '__main__':
    print('Hello, World')