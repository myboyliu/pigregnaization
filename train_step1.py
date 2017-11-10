# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 19:27:29 2017

@author: lilhope
"""

import os
import mxnet as mx

import numpy as np
import argparse
import time
import logging
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet import autograd as ag

logging.basicConfig(level=logging.INFO)
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--data_root',type=str,default=os.path.join(os.getcwd(),'..','data'),
                        help='data root dir')
 
    parser.add_argument('--train-data', type=str, default='pig_train',
                        help='training record file to use, required for imagenet.')
    parser.add_argument('--val-data', type=str, default='pig_test',
                        help='validation record file to use, required for imagenet.')
   
    parser.add_argument('--batch-size', type=int, default=12,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU ids to be used')
  
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate. default is 0.01.')
    parser.add_argument('-momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
   
    parser.add_argument('--seed', type=int, default=233,
                        help='random seed to use. Default=223.')
    parser.add_argument('--mode', type=str,default='hybrid',
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--model', type=str, default='resnet18_v1',
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--use_thumbnail',default=False,
                        help='use thumbnail or not in resnet. default is false.')
    parser.add_argument('--batch-norm', default=True,
                        help='enable batch normalization or not in vgg. default is true.')
    parser.add_argument('--use-pretrained',default=False,
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--log-interval', type=int, default=50, help='Number of batches to wait before logging.')
  
    args = parser.parse_args()
    return args
def train_net():
    args = parse_args()
    logging.info(args)
    mx.random.seed(args.seed)
    #
    batch_size = args.batch_size
    gpus=args.gpus
    if gpus:
    	ctx = [mx.gpu(int(i)) for i in gpus.split(',')]
    else:
	ctx = [mx.cpu()]
    model_name=args.model
    kwargs = {'ctx': ctx, 'pretrained': args.use_pretrained, 'classes': 30}
    if model_name.startswith('resnet'):
   	 kwargs['thumbnail'] = args.use_thumbnail
    if model_name.startswith('vgg'):
	 kwargs['batch_norm'] = args.batch_norm
    net = models.get_model(args.model,**kwargs)
    train_rec = os.path.join(args.data_root,args.train_data+'.rec')
    #train_list = os.path.join(args.data_root,args.train-data+'.lst')
    val_rec = os.path.join(args.data_root,args.val_data+'.rec')
    print(train_rec)
    print(val_rec)
    #val_rec = os.path.join(args.data_root,args.val-data+'.lst')
   
    train_data = mx.io.ImageRecordIter(path_imgrec=train_rec,#the target record file
                                       data_shape=(3,640,1000),#output shape,640*1000 region will crop from the data
                                       batch_size=batch_size,#batch size
                                       shuffle=True,
                                       mean_r=117.218,mean_g=118.462,mean_b=130.496,
                                       std_r=49.835,std_g=50.261,std_b=51.662,
                                       rand_mirror=True)# the only data agumentation params
    val_data = mx.io.ImageRecordIter(path_imgrec=val_rec,
                                     data_shape=(3,640,1000),
                                     batch_size=batch_size,
                                     mean_r=117.218,mean_g=118.462,mead_b=130.396,
                                     std_r=49.825,std_g=50.262,std_b=51.662)
   
    #train_data, val_data = mnist_iterator(batch_size, (1, 32, 32))
    if args.mode=='hybrid':
        net.initialize(mx.init.Xavier(magnitude=2),ctx=ctx)
        trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum},
                            kvstore = args.kvstore)
        metric = mx.metric.Accuracy()
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        epochs = args.epochs
        for epoch in range(epochs):
            tic = time.time()
       	    train_data.reset()
            metric.reset()
            btic = time.time()
            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
                outputs = []
                Ls = []
                with ag.record():
                    for x, y in zip(data, label):
                        z = net(x)
                        L = loss(z, y)
                        #print(y)
                        # store the loss and do backward after we have done forward
                        # on all GPUs for better speed on multiple GPUs.
                        Ls.append(L)
                        outputs.append(z)
                    for L in Ls:
                        L.backward()
                trainer.step(batch.data[0].shape[0])
	        #print(outputs)
                metric.update(label, outputs)
                if args.log_interval and not (i+1)%args.log_interval:
                    name, acc = metric.get()
                    logging.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f'%(
                               epoch, i, batch_size/(time.time()-btic), name, acc))
                    btic = time.time()

            name, acc = metric.get()
            logging.info('[Epoch %d] training: %s=%f'%(epoch, name, acc))
            logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
            #name, val_acc = test(net,val_data,ctx)
            val_data.reset()
            for batch in val_data:
		data  = gluon.utils.split_and_load(batch.data[0],ctx_list=ctx,batch_axis=0)
		label = gluon.utils.split_and_load(batch.label[0],ctx_list=ctx,batch_axis=0)
		outputs = []
		for x in data:
		    outputs.append(net(x))
		metric.update(label,outputs)
	    name, val_acc = metric.get()
		
            logging.info('[Epoch %d] validation: %s=%f'%(epoch, name, val_acc))
            net.save_params('image-classifier-%s-%d.params'%(args.model, epoch))
def test(net,val_data,ctx):
    metric = mx.metric.Accuracy()
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        #print(outputs)
        metric.update(label, outputs)
    return metric.get()
train_net()    
