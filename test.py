import time
import numpy as np
import tensorflow as tf
from capsulenet import CapsuleNet
from Mybase.solver import Solver


def test():
    
    mdl = CapsuleNet(cls_num=10, reg=1e-4, typ=tf.float32)
    sov = Solver(mdl,
                 opm_cfg={
                     'lr_base':      0.001,
                     #'decay_rule':  'fixed',
                     'decay_rule':  'exponential',
                     'decay_rate':   0.5,
                     'decay_step':    50,
                     'staircase':    False,
                     'optim_rule':  'adam',
                     #'optim_rule':  'momentum',
                     #'momentum':     0.9,
                     #'use_nesterov': True
                 },
                 use_gpu     =   True, 
                 gpu_lst     =    '0',
                 bat_siz     =    100,
                 tra_num     =  50000,
                 val_num     =  10000,
                 epc_num     = 100000,
                 min_que_tra =  60000,
                 min_que_val =  10000,
                 prt_ena     =   True,
                 itr_per_prt =     20,
                 tst_num     =   None,
                 tst_shw     =   True,
                 tst_sav     =   True,
                 mdl_nam     = 'model.ckpt',
                 mdl_dir     = 'Mybase/Model',
                 log_dir     = 'Mybase/logdata',
                 dat_dir     = 'Mybase/datasets',
                 mov_ave_dca = 0.99)
    print('TRAINING...')
    sov.train()
    '''
    print('TESTING...')
    sov.test()
    sov.display_detections()
    #sov.show_loss_acc()
    '''
test()
