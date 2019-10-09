# -*- coding: utf-8 -*-
from model import Audio2Video
from loadstore import plotLoss, pred_dir

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

args['pass_id']    = 'L1-h60-d20-n'
args['argspath']   = None
args['showGraph']  = False
args['preprocess'] = False
args['outp_norm']  = True

predict   = False
audiopath = pred_dir+'test037.npy'

def run():
    a2v = Audio2Video(args=args)
    if not predict:
        a2v.train()
        plotLoss(args['pass_id'])
    else:
        a2v.test(audiopath=audiopath)

if __name__ == '__main__':
    run()
