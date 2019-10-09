# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:28:05 2019

@author: xinzhu
"""

from subprocess import call
from loadstore import data_dir
import numpy as np
import bisect
import cv2
import os

base_dir = 'predict/'
mark_color = (255, 255, 255)
vfps = 30

def formMp4(mp4_path, PCA_MAT, size=(1280, 720), fps=vfps):   
    # load result data predicted by LSTM network
    data = np.load(mp4_path)
    outp, times, delay = data['outp'], data['times'], int(data['step_delay'])
    delay_times = times[delay+1:]
    period = times[1] - times[0]
    # delay_times.shape == outp.shape[0]
    
    # construct each frame
    filedir, filename = os.path.split(mp4_path)
    filename, _ = os.path.splitext(filename)
    writer = cv2.VideoWriter(filedir+'/'+filename+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    vnfr = int((delay_times[-1] - delay_times[0]) * vfps) 
    for i in range(vnfr):
        # interpolating and get the 20 landmark positions
        itime = i / vfps - delay_times[0]
        left = bisect.bisect_left(delay_times, itime)
        alpha = (itime - delay_times[left]) / period
        ldmks = outp[left,:] * (1-alpha) + outp[left+1,:] * alpha
        ldmks = np.matmul(PCA_MAT, ldmks).reshape((20, 2))
        
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for pt in ldmks:
            show_pt = pt*400 + (440, 160)
            show_pt = show_pt.astype(np.int)
            frame = cv2.circle(frame, tuple(show_pt), 3, mark_color, -1)
        writer.write(frame)
    return filedir+'/'+filename+'.avi'
        
def combine(mp4_path, mp3_path):
    bdir, namext = os.path.split(mp4_path)
    name, _ = os.path.splitext(namext)
    outp_path = bdir + '/' + name + '.mp4'
    command = 'ffmpeg -i ' + mp4_path + ' -i ' + mp3_path + ' -c:v copy -c:a aac -strict experimental ' + outp_path
    call(command)
    os.remove(mp4_path)
    return outp_path
    
if __name__ == '__main__':
    PCA_MAT = np.load(data_dir + 'PCA_MAT.npy')
    mp4_path = formMp4(base_dir+'test036_res.npz', PCA_MAT)
    outp_path = combine(mp4_path, base_dir+'test036.mp3')
    print('Final results are successfully saved to path: %s' % outp_path)