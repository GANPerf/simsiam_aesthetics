# -*- coding: utf-8 -*-
from __future__ import print_function, division
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from PIL import Image
from scipy import stats
from sklearn.metrics import mean_squared_error

from model_def import *
from util import *

seed = 0

torch.manual_seed( seed )

use_gpu = True









max_epoch = 50
BATCH_SIZE = 64
C1, C2, weight_decay = 1, 0, 1e-5
lr_model, lr_D = 1e-4, 5e-5





# 载入数据
train_set = MyDataset( '../AADB Database input', 'train' )
train_loader = DataLoader( train_set, batch_size=BATCH_SIZE, shuffle=True )

valid_set = MyDataset( '../AADB Database input', 'valid' )
valid_loader = DataLoader( valid_set, batch_size=BATCH_SIZE, shuffle=True )

test_set = MyDataset( '../AADB Database input', 'test' )
test_loader = DataLoader( test_set, batch_size=BATCH_SIZE, shuffle=True )









model = MyModel()
criterion_MSE = nn.MSELoss()

if use_gpu:
    model.cuda()
    criterion_MSE.cuda()


# 根据http://pytorch.org/docs/master/optim.html的建议
# 在创建optimizer之前调用函数model.cuda()
optimizer = optim.Adam( filter( lambda p: p.requires_grad, model.parameters() ), lr=lr_model, weight_decay=weight_decay )
scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'max', patience=5 )  # 学习率衰减










model_D = Model_D()
criterion_BCE = nn.BCELoss()

if use_gpu:
    model_D.cuda()
    criterion_BCE.cuda()

optimizer_D = optim.Adam( model_D.parameters(), lr=lr_D, weight_decay=weight_decay )








print( '\n-------------- Training Phase --------------' )


record = pd.DataFrame( [], columns=['epoch', 'train_mse', 'train_rank_spearman',
    'valid_mse', 'valid_rank_spearman', 'test_mse', 'test_rank_spearman'] )




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
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
    
        if use_gpu:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
       

        real_labels = sample_real_labels( batch_x.size(0), '../AADB Database input/label/train_label.csv' )
        real_labels = Variable(real_labels)

        label_one = Variable( torch.ones( batch_x.size(0) ) )      # 1 for real
        label_zero = Variable( torch.zeros( batch_x.size(0) ) )    # 0 for fake

        if use_gpu:
            real_labels = real_labels.cuda()
            label_one = label_one.cuda()
            label_zero = label_zero.cuda()




        # ------------ Training model_D ------------
        optimizer_D.zero_grad()
        model.eval()  # lock model


        # real label=1 for model_D
        output_real = model_D( real_labels )
        loss_real = criterion_BCE( output_real.squeeze(), label_one )
        acc_real = ( output_real >= 0.5 ).data.float().mean()

        # fake label=0 for model_D
        fake_labels = model( batch_x ).detach()
        output_fake = model_D( fake_labels )
        loss_fake = criterion_BCE( output_fake.squeeze(), label_zero )        
        acc_fake = ( output_fake < 0.5 ).data.float().mean()

        # 汇总real和fake
        loss_D = loss_real + loss_fake
        acc_D = ( acc_real + acc_fake ) / 2

        loss_D.backward()
        optimizer_D.step()






        # ------------ Training model ------------
        model.train()  # unlock model
        optimizer.zero_grad()


        fake_labels = model( batch_x )
        output = model_D( fake_labels )

        term1 = C1 * criterion_MSE( fake_labels, batch_y )
        term2 = C2 * criterion_BCE( output.squeeze(), label_one )

        loss = term1 + term2


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
            print( '\tbatch_i=%d\tloss_model=%f [%f, %f], loss_D=%f, acc_D=%f' % ( i, loss, term1, term2, loss_D, acc_D ) )







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






record.to_csv( 'record.csv', index=False )
torch.save( model.state_dict(), 'model_param.pkl' )





