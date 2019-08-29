# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
import math
import random
import bisect
import pickle

video_dir = 'video/'        # video raw features
audio_dir = 'audio/'        # audio raw features
save_dir  = 'save/'         # network saved data
log_dir   = 'log/'          # tensorboard data

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
            print(vsubdir)
            
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
                left_vfr = int(vfr - vstartfr)
                right_vfr = min(left_vfr+1, vfeatures.shape[0]-1)
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
    
    if preprocess or os.path.exists(save_dir+'/inout_stat.npz') == False:
        # extract raw features from video_dir and audio_dir
        inps, outps = loadRawFeature(vali_rate)
        
        # normalize them
        inps, outps, means, stds = normalizeData(inps, outps)

        # save them to save/inout_stat.npz   
        np.savez(save_dir+'inout_stat.npz', 
                 inps_mean=means[0], outps_mean=means[1],
                 inps_std=stds[0],   outps_std=stds[1])
        inout_data = {'inps':inps, 'outps':outps}
        with open(save_dir+'inout_data.pkl', 'wb') as f:
            pickle.dump(inout_data, f)
    
    # create work space for current pass
    if os.path.exists(save_dir + pass_id + '/') == False:
        os.mkdir(save_dir + pass_id + '/')
            
    # extract inputs and outputs from save/passId/inout_stat
    with open(save_dir+'inout_data.pkl', 'rb') as f:
        inout_data = pickle.load(f)
    inps, outps = inout_data['inps'], inout_data['outps']
    
    # deal with step delay
    new_inps  = {'training':[], 'validation':[]}
    new_outps = {'training':[], 'validation':[]}
    for key in new_inps.keys():
        for inp, outp in zip(inps[key], outps[key]):
            # throw away those which have less than <step_delay+seq_len> frames
            if inp.shape[0] - step_delay >= seq_len:
                new_inps[key].append(np.copy(inp[step_delay:, :]))
                new_outps[key].append(np.copy(outp[:(-step_delay if step_delay > 0 else None), :]))
    
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
        startfr = random.randint(0, inp.shape[0]-seq_len-1)
        x.append(np.copy(inp[startfr : startfr+seq_len]))
        y.append(np.copy(outp[startfr: startfr+seq_len]))
        
        # determine whether to move to next fragment of data
        nseq = math.ceil(inp.shape[0] / seq_len)
        if random.random() < (1 / nseq):
            # let X be the number of sequences extracted from inp,
            # then E(X) = nseq
            batch_pt[mode] = (batch_pt[mode] + 1) % nbatches[mode]
        
    return np.array(x), np.array(y)

def restoreState(sess, pass_id):
    '''restore network states
    ### Parameters
    sess       (Session) \\
    pass_id    (str)    name of this pass, including training and testing
    ### Return Values
    startEpoch (int)    we should continue training from this epoch \\
    saver      (Saver)  tf.train.Saver
    '''
    # define checkpoint saver
    saver = tf.train.Saver(tf.global_variables())

    # restore start epoch number and global variables
    last_ckpt = tf.train.latest_checkpoint(save_dir+pass_id)
    if last_ckpt is None:
        # no ckpt file yet
        startEpoch = 0
        print('train from the very beginning')
    else:
        startEpoch = int(last_ckpt.split('-')[-1])
        saver.restore(sess, last_ckpt)
        print('train from epoch', startEpoch+1)
            
    return startEpoch, saver
        
def reportBatch(pass_id, e, b, nepochs, nbatches, b_during, b_savef, tloss):
    '''summary after each training batch
    ### Parameters
    pass_id    (str)    name of this pass, including training and testing \\
    e          (int)    current epoch index \\
    b          (int)    current training batch index \\
    nepochs    (int)    total number of epochs \\
    nbatches   (int)    total number of training batches per epoch \\
    b_during   (float)  time elapse per training batch \\
    b_savef    (int)    frequency to save to progress.log \\
    tloss      (float)  training loss in current training batch
    '''
    # calculate eta-time
    cur_batches = e * nbatches + b
    eta_batches = nepochs*nbatches - 1 - cur_batches
    eta_time = round(eta_batches * b_during)
    m, s = divmod(eta_time, 60)
    h, m = divmod(m, 60)
    eta_str = "%d:%02d:%02d" % (h, m, s)
    
    # initialize log file
    if cur_batches == 0:
        with open(save_dir+pass_id+'/loss.log', 'w') as f:
            f.write("0 %f %f\n" % (tloss, tloss))
    
    # print & save batch report
    report = "epoch:%d/%d batch:%d/%d tloss:%f eta_time:%s" % (e+1, nepochs, b+1, nbatches, tloss, eta_str)
    if b == 0 or (b+1) % b_savef == 0 or b == nbatches - 1:
        with open(save_dir+pass_id+'/progress.log', 'w') as f:
            f.write(report)
#        print('progress.log at epoch %d/%d, batch %d/%d saved' % (e+1, nepochs, b+1, nbatches))
        print(report)

def reportEpoch(pass_id, sess, saver, e, nepochs, e_savef, trainLoss, validLoss):
    '''summary after each epoch
    ### Parameters
    pass_id    (str)     name of this pass, including training and testing \\
    sess       (Session) tf.Session \\
    saver      (Saver)   tf.train.Saver \\
    e          (int)     current epoch index \\
    nepochs    (int)     total number of epochs \\
    e_savef    (int)     frequency to save to checkpoint file \\
    trainLoss  (float)   final training loss in this epoch \\
    validLoss  (float)   average validation loss in this epoch
    '''
    # save checkpoint
    if (e+1) % e_savef == 0 or e == nepochs - 1:
        saver.save(sess, save_dir+pass_id+'/model', global_step=e+1)
        print('checkpoint at epoch %d/%d saved' % (e+1, nepochs))
        
    # report loss
    with open(save_dir+pass_id+'/loss.log', 'r') as f:
        records = f.readlines()[:e+1]
    records.append("%d %f %f\n" % (e+1, trainLoss, validLoss))
    with open(save_dir+pass_id+'/loss.log', 'w') as f:
        f.write(''.join(records))
    print('loss.log at epoch %d/%d saved' % (e+1, nepochs))
        
if __name__ == '__main__':
    print('Hello, World')