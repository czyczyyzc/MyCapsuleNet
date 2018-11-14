import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from .load_weights import *
from .optim_utils import *
from .capsule_utils.make_image import *

def get_data(fid):
    try:  
            a = pickle.load(fid)
            return 1, a
    except EOFError:  
            return 0, 0

def get_all_data(fid):
    data = []
    while(True):
        sig, dat = get_data(fid)
        if(sig == 0): break
        else:
            data.append(dat)
    return data

class Solver(object):
    
    def __init__(self, mdl, **kwargs):
        
        self.mdl         = mdl
        self.opm_cfg     = kwargs.pop('opm_cfg',       {})
        self.use_gpu     = kwargs.pop('use_gpu',     True)
        self.gpu_lst     = kwargs.pop('gpu_lst',      '0')
        self.gpu_num     = len(self.gpu_lst.split(',')) if self.use_gpu else 1
        self.mdl_dev     = '/gpu:%d'                    if self.use_gpu else '/cpu:%d'
        self.MDL_DEV     = 'GPU_%d'                     if self.use_gpu else 'CPU_%d'
        self.bat_siz     = kwargs.pop('bat_siz',        2)
        self.bat_siz_all = self.bat_siz                    * self.gpu_num
        self.tra_num     = kwargs.pop('tra_num',     8000) * self.gpu_num
        self.val_num     = kwargs.pop('val_num',       80) * self.gpu_num
        self.epc_num     = kwargs.pop('epc_num',       10)
        self.min_que_tra = kwargs.pop('min_que_tra', 5000) * self.gpu_num
        self.min_que_val = kwargs.pop('min_que_val', 1000) * self.gpu_num
        self.prt_ena     = kwargs.pop('prt_ena',     True)
        self.itr_per_prt = kwargs.pop('itr_per_prt',   20)
        self.tst_num     = kwargs.pop('tst_num',     None)
        self.tst_shw     = kwargs.pop('tst_shw',     True)
        self.tst_sav     = kwargs.pop('tst_sav',     True)
        self.mdl_nam     = kwargs.pop('mdl_nam',    'model.ckpt'     )
        self.mdl_dir     = kwargs.pop('mdl_dir',    'Mybase/Model'   )
        self.log_dir     = kwargs.pop('log_dir',    'Mybase/logdata' )
        self.dat_dir     = kwargs.pop('dat_dir',    'Mybase/datasets')
        self.mov_ave_dca = kwargs.pop('mov_ave_dca', 0.99)
        self.dat_dir_tra = self.dat_dir + '/train'
        self.dat_dir_val = self.dat_dir + '/val'
        self.dat_dir_tst = self.dat_dir + '/test'
        self.dat_dir_rst = self.dat_dir + '/result'
        self.log_dir_tra = self.log_dir + '/train'
        self.log_dir_val = self.log_dir + '/val'
        self.log_dir_tst = self.log_dir + '/test'
        self.epc_cnt     = 0
        
        os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_lst
        
        if len(kwargs) > 0:
            extra = ', '.join('%s' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)
    
    ###############################For CLASSIFY################################
    def _train_step(self, mtra=None, mtst=None, itr_per_epc=None, glb_stp=None):
        #将简单的运算放在CPU上，只有神经网络的训练过程放在GPU上
        with tf.device('/cpu:0'):
            
            GI_tra = GeneratorForMNIST(True,  self.dat_dir_tra, self.bat_siz_all, self.epc_num, self.min_que_tra)
            GI_val = GeneratorForMNIST(False, self.dat_dir_val, self.bat_siz_all, self.epc_num, self.min_que_val)
            imgs_tra, lbls_tra = GI_tra.get_input()
            imgs_val, lbls_val = GI_val.get_input()
            imgs = tf.cond(mtst, lambda: imgs_val, lambda: imgs_tra, strict=True)
            lbls = tf.cond(mtst, lambda: lbls_val, lambda: lbls_tra, strict=True)
            #with tf.name_scope('input_image'):
            #    tf.summary.image('input', X, 10)

            self.opm_cfg['decay_step'] =  self.opm_cfg['decay_step'] * itr_per_epc #decay
            tra_stp, lrn_rat = update_rule(self.opm_cfg, glb_stp)

            self.mdl.mod_tra = True
            self.mdl.glb_pol = True

            grds_lst = []
            loss_lst = []
            accs_lst = []
            for i in range(self.gpu_num):
                with tf.device(self.mdl_dev % i):
                    with tf.name_scope(self.MDL_DEV % i) as scp:
                        sta = i     * self.bat_siz
                        end = (i+1) * self.bat_siz
                        loss, accs = \
                            self.mdl.forward(imgs=imgs[sta:end], lbls=lbls[sta:end], mtra=mtra, scp=scp)

                        #在第一次声明变量之后，将控制变量重用的参数设置为True。这样可以让不同的GPU更新同一组参数
                        #注意tf.name_scope函数并不会影响tf.get_variable的命名空间
                        tf.get_variable_scope().reuse_variables()
                        #使用当前GPU计算所有变量的梯度
                        grds = tra_stp.compute_gradients(loss[0])
                        #print(grds)
                grds_lst.append(grds)
                loss_lst.append(loss)
                accs_lst.append(accs)
            '''
            with tf.variable_scope('average',  reuse = tf.AUTO_REUSE):
                mov_ave    = tf.train.ExponentialMovingAverage(self.mov_ave_dca, glb_stp)
                mov_ave_op = mov_ave.apply(tf.trainable_variables())
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mov_ave_op)
            '''
            with tf.variable_scope('optimize', reuse = tf.AUTO_REUSE):
                grds = average_gradients(grds_lst)
                upd_opas = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(upd_opas):
                    tra_opa = tra_stp.apply_gradients(grds, global_step=glb_stp)

            loss = tf.stack(loss_lst,   axis=0)
            loss = tf.reduce_mean(loss, axis=0)
            accs = tf.concat(accs_lst,  axis=0)
            #tf.summary.scalar('loss', loss)
            #tf.summary.scalar('acc', acc)
            #for grad, var in grads:
            #    if grad is not None:
            #        tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)
            #for var in tf.trainable_variables():
            #    tf.summary.histogram(var.op.name, var)
        return tra_opa, lrn_rat, loss, accs, lbls
    
    def concat(self, sess=None, fetches=None, feed_dict=None, itr_num=None, gen=None, tsrs=None, keps=None):
        
        rsts_lst = [[] for _ in range(len(fetches))]
        if keps != None:
            rsts_kep = [[] for _ in range(len(keps))]
        for _ in range(itr_num):
            if gen != None:
                feds = next(gen)
                for i, tsr in enumerate(tsrs):
                    feed_dict[tsr] = feds[i]
                for i, kep in enumerate(keps):
                    rsts_kep[i].append(feds[kep])
            rsts = sess.run(fetches, feed_dict=feed_dict)
            for i, rst in enumerate(rsts):
                rsts_lst[i].append(rst)
        for i, rst in enumerate(rsts_lst):
            rsts_lst[i] = np.concatenate(rst, axis=0)
        if keps != None:
            for i, rst in enumerate(rsts_kep):
                rsts_kep[i] = np.concatenate(rst, axis=0)
            return rsts_lst, rsts_kep
        else:
            return rsts_lst
    
    def merge(self, rsts=None, rst_nums=None):
        
        rst_imxs = []
        rsts_lst = [[] for _ in range(len(rsts))]
        for i, rst_num in enumerate(rst_nums): #batch
            rst_imxs.extend([i]*rst_num)
            for j, rst in enumerate(rsts):     #tensors
                rsts_lst[j].append(rst[i][:rst_num])
        rst_imxs = np.asarray(rst_imxs, dtype=np.int32)
        for i, rst in enumerate(rsts_lst):
            rsts_lst[i] = np.concatenate(rst, axis=0)
        return rsts_lst, rst_imxs
    
    #####################################For CLASSIFY#####################################
    def train(self):
        
        itr_per_epc = max(self.tra_num // self.bat_siz_all, 1)
        if self.tra_num % self.bat_siz_all != 0:
            itr_per_epc += 1
        tra_itr_num = self.epc_num * itr_per_epc
        
        val_itr_num = max(self.val_num // self.bat_siz_all, 1)
        if self.val_num % self.bat_siz_all != 0:
            val_itr_num += 1

        tf.reset_default_graph()
        
        mtra = tf.placeholder(dtype=tf.bool, name='train')
        mtst = tf.placeholder(dtype=tf.bool, name='test' )
        
        glb_stp = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
        
        tra_opa, lrn_rat, loss, accs, lbls = \
            self._train_step(mtra, mtst, itr_per_epc, glb_stp)
        
        #with tf.device(self.mdl_dev % 0):
            #TODO
            
        #tf.summary.scalar('loss', loss)
        #summary_op   = tf.summary.merge_all()
        #summary_loss = tf.summary.merge(loss)
        #writer       = tf.summary.FileWriter(LOG_PATH, sess.graph, flush_secs=5) #tf.get_default_graph()    
        #gpu_options  = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        #config       = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        #config       = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, device_count={'CPU': 2}, \
        #                              inter_op_parallelism_threads=16, intra_op_parallelism_threads=16)
        config        = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            
            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            #coord   = tf.train.Coordinator()
            #threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            saver = tf.train.Saver()
            ckpt  = tf.train.get_checkpoint_state(self.mdl_dir)
            if ckpt and ckpt.model_checkpoint_path:
                #var = tf.global_variables()
                #var = [v for v in var if 'layers_module1_0/' in v.name or 'layers_module1_1/' in v.name]
                #var = [v for v in var if 'average/' not in v.name and 'optimize/' not in v.name]
                #var_ave = tf.train.ExponentialMovingAverage(self.mv_ave_decay, glb_stp)
                #var = var_ave.variables_to_restore()
                #saver = tf.train.Saver(var)
                saver.restore(sess, ckpt.model_checkpoint_path)
                saver = tf.train.Saver()
            
            with open(os.path.join(self.log_dir_tra, 'accs'), 'ab') as fid_tra_accs, \
                 open(os.path.join(self.log_dir_tra, 'loss'), 'ab') as fid_tra_loss, \
                 open(os.path.join(self.log_dir_val, 'accs'), 'ab') as fid_val_accs, \
                 open(os.path.join(self.log_dir_val, 'loss'), 'ab') as fid_val_loss:
                
                tra_loss_lst = []
                for t in range(tra_itr_num):
                    epc_end = (t + 1) % itr_per_epc == 0
                    itr_sta = (t == 0)
                    itr_end = (t == tra_itr_num - 1)
                    if epc_end:
                        self.epc_cnt += 1
                        
                    #_, summary, loss1, = sess.run([train_op, summary_op, loss], feed_dict = {mtrain: True})
                    #writer.add_summary(summary, global_step=glb_stp.eval())
                    _, tra_loss = sess.run([tra_opa, loss], feed_dict={mtra: True, mtst: False})
                    tra_loss_lst.append(tra_loss)
                    
                    if self.prt_ena and t % self.itr_per_prt == 0:
                        tra_loss_tmp = np.mean(tra_loss_lst, axis=0)
                        tra_loss_lst = []
                        print('(Iteration %d / %d) losses: %s' % (t + 1, tra_itr_num, str(tra_loss_tmp)))
                        
                    #if itr_sta or itr_end or epc_end:
                    if itr_end or epc_end: 
                        saver.save(sess, os.path.join(self.mdl_dir, self.mdl_nam), global_step=glb_stp)
                        
                        fetches   = [accs[tf.newaxis], loss[tf.newaxis, :]]
                        feed_dict = {mtra: False, mtst: False}
                        tra_accs, tra_loss = self.concat(sess, fetches, feed_dict, val_itr_num)
                        tra_accs  = np.mean(tra_accs, axis=0)
                        tra_loss  = np.mean(tra_loss, axis=0)
                        fetches   = [accs[tf.newaxis], loss[tf.newaxis, :]]
                        feed_dict = {mtra: False, mtst: True }
                        val_accs, val_loss = self.concat(sess, fetches, feed_dict, val_itr_num)
                        val_accs  = np.mean(val_accs, axis=0)
                        val_loss  = np.mean(val_loss, axis=0)
                        
                        pickle.dump(tra_accs, fid_tra_accs, pickle.HIGHEST_PROTOCOL)
                        pickle.dump(tra_loss, fid_tra_loss, pickle.HIGHEST_PROTOCOL)
                        pickle.dump(val_accs, fid_val_accs, pickle.HIGHEST_PROTOCOL)
                        pickle.dump(val_loss, fid_val_loss, pickle.HIGHEST_PROTOCOL)
                        if self.prt_ena:
                            print('(Epoch %d / %d) tra_accs: %f, val_accs: %f, lrn_rate:%f \n tra_loss: %s \n val_loss: %s' \
                                  % (self.epc_cnt, self.epc_num, tra_accs, val_accs, lrn_rat.eval(), str(tra_loss), str(val_loss)))
            #coord.request_stop()
            #coord.join(threads)
    
    def show_loss_acc(self):

        with open(os.path.join(LOG_PATH1, 'loss'), 'rb') as fid_train_loss, \
             open(os.path.join(LOG_PATH1, 'mAP'), 'rb') as fid_train_mAP, \
             open(os.path.join(LOG_PATH2, 'mAP'), 'rb') as fid_val_mAP:
                    
            loss_history      = get_all_data(fid_train_loss)
            train_acc_history = get_all_data(fid_train_mAP)
            val_acc_history   = get_all_data(fid_val_mAP)

            plt.figure(1)

            plt.subplot(2, 1, 1)
            plt.title('Training loss')
            plt.xlabel('Iteration')

            plt.subplot(2, 1, 2)
            plt.title('accuracy')
            plt.xlabel('Epoch')
            
            #plt.subplot(3, 1, 3)
            #plt.title('Validation accuracy')
            #plt.xlabel('Epoch')
            
            plt.subplot(2, 1, 1)
            plt.plot(loss_history, 'o')

            plt.subplot(2, 1, 2)
            plt.plot(train_acc_history, '-o', label='train_acc')
            plt.plot(val_acc_history, '-o', label='val_acc')

            for i in [1, 2]:
                plt.subplot(2, 1, i)
                plt.legend(loc='upper center', ncol=4)

                plt.gcf().set_size_inches(15, 15)
            
            plt.show()
