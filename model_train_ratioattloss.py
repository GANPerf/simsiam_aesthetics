# -*- coding: utf-8 -*-
from __future__ import division, print_function

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from scipy import stats
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from loss_def import *
from model_def import *
from util import *

seed = 0

torch.manual_seed( seed )

use_gpu = True
#use_gpu =False









max_epoch = 10
BATCH_SIZE = 64
C1, C2,C3, weight_decay = 1,1, 1, 1e-5
lr_model, lr_D = 1e-4, 5e-5





# 载入数据
train_set = MyDataset( '../AADB Database input', 'train' )
train_loader = DataLoader( train_set, batch_size=BATCH_SIZE, shuffle=True )

valid_set = MyDataset( '../AADB Database input', 'valid' )
valid_loader = DataLoader( valid_set, batch_size=BATCH_SIZE, shuffle=True )

test_set = MyDataset( '../AADB Database input', 'test' )
test_loader = DataLoader( test_set, batch_size=BATCH_SIZE, shuffle=True )









model = MyModel()
param_file = 'model_param.pkl'
model.load_state_dict( torch.load( param_file ) )
criterion_MSE = nn.MSELoss()
criterion_ATT=ATTLoss()
criterion_RANK=RANKLoss()
if use_gpu:
    model.cuda()
    criterion_MSE.cuda()
    criterion_RANK.cuda()
    criterion_ATT.cuda()



# 根据http://pytorch.org/docs/master/optim.html的建议
# 在创建optimizer之前调用函数model.cuda()
optimizer = optim.Adam( filter( lambda p: p.requires_grad, model.parameters() ), lr=lr_model, weight_decay=weight_decay )
scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'max', patience=5 )  # 学习率衰减





















print( '\n-------------- Training Phase --------------' )


record = pd.DataFrame( [], columns=['epoch', 'train_mse', 'train_rank_spearman',
    'valid_mse', 'valid_rank_spearman', 'test_mse', 'test_rank_spearman'] )

corr=np.array([0.340,0.569,0.697,0.416,0.574,0.204,0.542,0.084,0.359,0.117,0.518])
corr=torch.FloatTensor(corr)
var_corr=Variable(corr)
if(use_gpu):
    var_corr=var_corr.cuda()


for epoch in range( max_epoch ):

    print( '\nEpoch %d' % epoch )
    t0 = time.time()

    values = [epoch]




    ################ Training on Train Set ################

    y_train = np.zeros( 0, dtype=float )
    pred_train = np.zeros( 0, dtype=float )

    for i, data in enumerate( train_loader, 0 ):

        # ------------ Prapare Variables ------------
        batch_x, batch_y = data
       
        att=np.array(batch_y[:,1:12])
        att=np.where(att>0.5,1,0)
        att=torch.FloatTensor(att)
        att=Variable(att)
        
        score=np.array(batch_y[:,0])
        score=np.where(score>0.5,1,0)
        score=torch.FloatTensor(score)
        score=Variable(score)

        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        
        if use_gpu:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            att=att.cuda()
            score=score.cuda()
       

        real_labels = sample_real_labels( batch_x.size(0), '../AADB Database input/label/train_label.csv' )
        real_labels = Variable(real_labels)

        label_one = Variable( torch.ones( batch_x.size(0) ) )      # 1 for real
        label_zero = Variable( torch.zeros( batch_x.size(0) ) )    # 0 for fake

        if use_gpu:
            real_labels = real_labels.cuda()
            label_one = label_one.cuda()
            label_zero = label_zero.cuda()




     






        # ------------ Training model ------------
        model.train()  # unlock model
        optimizer.zero_grad()


        fake_labels = model( batch_x )
        fake_score=fake_labels[:,0]
        real_score=batch_y[:,0]
        
        term3=0
        for j in range(11):
            atti=att[:,j]
            tag=torch.abs(1-atti-score)  #atti=0,1;score=0,1
            term3+=C3 * var_corr[j]*criterion_ATT(fake_score,atti,tag)

        term2 = C2 * criterion_RANK(fake_score,real_score)
        term1 = C1 * criterion_MSE( fake_labels, batch_y )

        loss = term1 +term2+term3


        loss.backward()
        optimizer.step()







        # ------------ Preparation for Evaluation on Train Set ------------
        fake_labels = fake_labels[:, 0].squeeze()            # output选择第0列
        batch_y = batch_y[:, 0].squeeze()          # batch_y选择第0列

        fake_labels = fake_labels.cpu().data.numpy() if use_gpu else fake_labels.data.numpy()
        batch_y = batch_y.cpu().data.numpy() if use_gpu else batch_y.data.numpy()

        pred_train = np.concatenate( [pred_train, fake_labels] )
        y_train = np.concatenate( [y_train, batch_y] )






        if i % 10 == 0:
            print( '\tbatch_i=%d\tloss_model=%f [%f, %f,%f]' % ( i, loss, term1, term2,term3 ) )







    ################ Evaluation on Train Set ################

    mse = mean_squared_error( y_train, pred_train )
    rank_spearman = stats.spearmanr( get_rank(y_train), get_rank(pred_train) )[0]
    values.append( mse )
    values.append( rank_spearman )
    print( 'Train Set\tmse=%f, rank_spearman=%f' % ( mse, rank_spearman ) )

    





    ################ Evaluation on Valid/Test Set ################

    model.eval()  # evaluation mode

    mse, rank_spearman = model_eval( model, valid_loader, use_gpu )
    values.append( mse )
    values.append( rank_spearman )
    print( 'Valid Set\tmse=%f, rank_spearman=%f' % ( mse, rank_spearman ) )


    scheduler.step( rank_spearman )  # 执行学习率衰减


    mse, rank_spearman = model_eval( model, test_loader, use_gpu )
    values.append( mse )
    values.append( rank_spearman )
    print( 'Test Set\tmse=%f, rank_spearman=%f' % ( mse, rank_spearman ) )


    print( 'Done in %.2fs' % ( time.time() - t0 ) )






    ################ Writing Record ################

    temp = pd.DataFrame( [values], columns=['epoch', 'train_mse', 'train_rank_spearman',
        'valid_mse', 'valid_rank_spearman', 'test_mse', 'test_rank_spearman'] )
    record = record.append( temp )






record.to_csv( 'record_test.csv', index=False )
#torch.save( model.state_dict(), 'model_param.pkl' )
