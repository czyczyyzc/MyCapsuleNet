import numpy as np
import tensorflow as tf

from Mybase import layers
from Mybase.layers import *
from Mybase.layers_utils import *
from Mybase.losses import *


class CapsuleNet(object):
    
    def __init__(self, cls_num=10, reg=1e-4, typ=tf.float32):
        
        self.cls_num  = cls_num #class number
        self.x_dim    = 8
        self.v_dim    = 16
        self.reg      = reg     #regularization
        self.typ      = typ     #dtype
        
        self.mod_tra  = True    #mode training
        self.glb_pol  = False   #global pooling
        
    def squash(self, x=None, layer=0, eps=1e-7):
        
        with tf.variable_scope('squash_'+str(layer)) as scope:
            squa = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
            sqrt = tf.sqrt(squa + eps)
            x    = squa / (1.0 + squa) * x / sqrt
            print_activations(x)
        return x
    
    def project(self, x=None, layer=0, reuse=False, trainable=True):
        x_shp = get_shape(x)                                           #[img_num, 1152, 8]
        with tf.variable_scope('project_'+str(layer), reuse=reuse) as scope:
            w = tf.get_variable(name='weights', shape=x_shp[1:3]+[self.cls_num,self.v_dim], dtype=self.typ, \
                                #initializer=tf.initializers.random_normal(mean=0.0, stddev=0.01), \
                                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True),
                                regularizer=tf.contrib.layers.l2_regularizer(self.reg), \
                                trainable=trainable)                   #(1152, 8, 10, 16)
            u = tf.einsum('ijk,jkmn->ijmn', x, w)                      #(img_num, 1152, 10, 16)
            print_activations(u)
        return u
    
    def route(self, u=None, layer=0, r=3):
        u_shp = get_shape(u)                                           #[img_num, 1152, 10, 16]
        with tf.variable_scope('route_'+str(layer)) as scope:
            
            b = tf.zeros(shape=u_shp[:-1]+[1], dtype=tf.float32)       #(img_num, 1152, 10,  1)
            def cond(i, u, b):
                c = tf.less(i, r)
                return c
            def body(i, u, b):
                c = tf.nn.softmax(b, axis=-2)                          #(img_num, 1152, 10,  1) 每个输入cap预测输出cap的概率 
                s = u * c                                              #(img_num, 1152, 10, 16)
                s = tf.reduce_sum(s, axis=1, keepdims=True)            #(img_num,    1, 10, 16)
                v = self.squash(s, 0)                                  #(img_num,    1, 10, 16)
                b = b + tf.reduce_sum(u*v, axis=-1, keepdims=True)     #(img_num, 1152, 10,  1)
                return [i+1, u, b]
            i = tf.constant(0)
            [i, u, b] = tf.while_loop(cond, body, loop_vars=[i, u, b], shape_invariants=None, \
                                      parallel_iterations=1, back_prop=True, swap_memory=True)
            
            c = tf.nn.softmax(b, axis=-2)                              #(img_num, 1152, 10,  1) 每个输入cap预测输出cap的概率 
            s = u * c                                                  #(img_num, 1152, 10, 16)
            s = tf.reduce_sum(s, axis=1, keepdims=True)                #(img_num,    1, 10, 16)
            v = self.squash(s, 1)                                      #(img_num,    1, 10, 16)
            v = tf.squeeze(v, axis=[1])                                #(img_num,   10, 16)
            print_activations(v)
        return v
        
    def margin_loss(self, v, y, layer=0, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        with tf.variable_scope('margin_loss_'+str(layer)) as scope:
            y  = tf.one_hot(y, depth=self.cls_num, dtype=tf.float32)   #(img_num, 10)
            v  = tf.norm(v, ord='euclidean', axis=-1, keepdims=False)  #(img_num, 10)
            fp = tf.square(tf.maximum(0., m_plus-v ))
            fn = tf.square(tf.maximum(0., v-m_minus))
            L  = y * fp + lambda_ * (1.0 - y) * fn
            L  = tf.reduce_mean(tf.reduce_sum(L, axis=-1))
            print_activations(L)
        return L
    
    def recons_loss(self, v, x, y, layer=0):
        x_shp = get_shape(x)
        with tf.variable_scope('recons_loss_'+str(layer)) as scope:
            x           = tf.reshape(x, [x_shp[0], -1])                       #(img_num, 784)
            x_shp       = get_shape(x)
            y           = tf.one_hot(y, depth=self.cls_num, dtype=tf.float32) #(img_num, 10)
            v           = v * tf.expand_dims(y, axis=-1)                      #(img_num, 10, 16)
            v           = tf.reshape(v, [x_shp[0], -1])                       #(img_num, 160)
            p           = {}
            p['com']    = {'reg':self.reg, 'wscale':0.01, 'dtype':self.typ, 'reuse':False, 'is_train':self.mod_tra, 'trainable':True}
            p['relu']   = {'alpha':-0.1}
            p['affine'] = {'dim':512,      'use_bias':True}
            v           = affine_relu1(v, 0, p)
            p['affine'] = {'dim':1024,     'use_bias':True}
            v           = affine_relu1(v, 1, p)
            p['affine'] = {'dim':x_shp[1], 'use_bias':True}
            v           = affine_sigmoid1(v, 0, p)
            L           = tf.reduce_sum(tf.square(x - v))
            print_activations(L)
        return L
    
    def total_loss(self, v, x, y, layer=0, alpha=0.0005):
        with tf.variable_scope('total_loss_'+str(layer)) as scope:
            L0 = self.margin_loss(v, y, 0)
            L1 = self.recons_loss(v, x, y, 0)
            L  = L0 + alpha * L1
            print_activations(L)
        return L
    
    def accuracy(self, v, y, layer=0):
        with tf.variable_scope('accuracy_'+str(layer)) as scope:
            v   = tf.norm(v, ord='euclidean', axis=-1, keepdims=False)   #(img_num, 10)
            v   = tf.cast(tf.argmax(v, axis=-1), dtype=tf.int32)         #(img_num)
            acc = tf.cast(tf.equal(v, y), tf.float32)                    #(img_num)
            acc = tf.reduce_mean(acc, keepdims=False)                    #(1)
            print_activations(acc)
        return acc
    
    def forward(self, imgs=None, lbls=None, mtra=None, scp=None): 
        
        img_shp = imgs.get_shape().as_list()
        img_num, img_hgt, img_wdh = img_shp[0], img_shp[1], img_shp[2]
        img_shp = np.stack([img_hgt, img_wdh], axis=0)
        #####################Common Parameters!############################
        com_pams = {
            'com':   {'reg':self.reg, 'wscale':0.01, 'dtype':self.typ, 'reuse':False, 'is_train':self.mod_tra, 'trainable':True},
            'bn':    {'eps':1e-5, 'decay':0.9997}, #0.9997
            'relu':  {'alpha':-0.1},
            'conv':  {'number':256,'shape':[9,9],'rate':[1,1],'stride':[1,1],'padding':'VALID','use_bias':True},
            'glb_pool':  {'axis':  [1, 2]},
            'reshape':   {'shape': [img_num, -1]},
            'squeeze':   {'axis':  [1, 2]},
            'transpose': {'perm':  [0, 3, 1, 2, 4]},
            'affine':    {'dim':   self.cls_num*self.v_dim, 'use_bias':False},
            'dropout':   {'keep_p': 0.75, 'shape': None},
            #'bilstm':   {'num_h': self.fet_dep//2, 'num_o': None, 'fbias': 1.0, 'tmajr': False},
            #'concat':   {'axis': 0},
            #'split':    {'axis': 0, 'number': img_num},
        }
        
        opas = {'op':[{'op':'conv_relu1', 'loop':1, 'params':{}},
                      {'op':'conv_relu1', 'loop':1, 'params':{'conv':{'stride':[2, 2]}}},
                      {'op':'reshape1',   'loop':1, 'params':{'reshape':{'shape':[img_num, -1, self.x_dim]}}},
                     ], 'loop':1}
        x    = layers_module1(imgs, 0, com_pams, opas, mtra)
        x    = self.squash(x, 0)
        u    = self.project(x, 0)
        v    = self.route(u, 0)
        accs = self.accuracy(v, lbls, 0)
        
        los_dat = self.total_loss(v, imgs, lbls, 0)
        los_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        los     = los_dat + los_reg * 0.0
        loss    = tf.stack([los, los_dat, los_reg], axis=0)
        return loss, accs