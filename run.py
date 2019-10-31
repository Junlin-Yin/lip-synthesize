# -*- coding: utf-8 -*-
import numpy as np
from model import Audio2Video
from loadstore import plotLoss, pred_dir, data_dir
from visual import formMp4, combine

# important parameters
args = {}
args['vr']         = 0.2   # validation set ratio
args['step_delay'] = 20    # step delay for LSTM (10ms per step)
args['dim_hidden'] = 60    # dimension of hidden layer and cell state
args['nlayers']    = 1     # number of LSTM layers
args['keep_prob']  = 0.8   # dropout keep probability
args['seq_len']    = 100   # sequence length (nframes per sequence)
args['batch_size'] = 100   # batch size (nseq per batch)
args['nepochs']    = 300   # number of epochs
args['grad_clip']  = 10    # gradient clipping threshold
args['lr']         = 1e-3  # initial learning rate
args['dr']         = 0.99  # learning rate's decay rate
args['b_savef']    = 50    # batch report save frequency
args['e_savef']    = 5     # epoch report save frequency

args['pass_id']    = 'L1-h60-d20-u'
args['argspath']   = None
args['showGraph']  = False
args['preprocess'] = False
args['outp_norm']  = False

predict   = True
testid = 'test037'
audiopath = pred_dir+testid+'.npy'
musicpath = pred_dir+testid+'.mp3'

def run():
    a2v = Audio2Video(args=args)
    if not predict:
        a2v.train()
        plotLoss(args['pass_id'])
    else:
        resdir = a2v.test(audiopath=audiopath)
        avi_path = None
        PCA_MAT = np.load(data_dir + 'PCA_MAT.npy')
        outp_path = formMp4(resdir, PCA_MAT, avi_path)
        print('Final results are successfully saved to path: %s' % outp_path)        

if __name__ == '__main__':
    run()
