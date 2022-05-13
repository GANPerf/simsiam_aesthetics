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
#use_gpu =False









max_epoch = 100
BATCH_SIZE = 34
C1, C2,C3, weight_decay = 1,0.1, 0, 1e-5
lr_model, lr_D = 1e-4, 5e-5





# 载入数据
train_set = MyDataset( '../AADBDatabaseinput', 'train' )
train_loader = DataLoader( train_set, batch_size=BATCH_SIZE, shuffle=True )

valid_set = MyDataset( '../AADBDatabaseinput', 'valid' )
valid_loader = DataLoader( valid_set, batch_size=BATCH_SIZE, shuffle=True )

test_set = MyDataset( '../AADBDatabaseinput', 'test' )
test_loader = DataLoader( test_set, batch_size=BATCH_SIZE, shuffle=True )









model = MyModel()

criterion_CE = nn.CrossEntropyLoss()
if use_gpu:
    model.cuda()
    criterion_CE.cuda()



# 根据http://pytorch.org/docs/master/optim.html的建议
# 在创建optimizer之前调用函数model.cuda()
optimizer = optim.Adam( filter( lambda p: p.requires_grad, model.parameters() ), lr=lr_model, weight_decay=weight_decay )
scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'max', patience=5 )  # 学习率衰减


#param_file='E:\yangyang\科研\第四个任务\师弟代码\Workspace Image Aesthetic Assessment\MyTest ResNet multi-label+GAN\model_param.pkl'
#model.load_state_dict( torch.load( param_file ) )











print( '\n-------------- Training Phase --------------' )


record = pd.DataFrame( [], columns=['epoch', 'train_acc',
        'valid_accuracy', 'test_accuracy'] )


for epoch in range( max_epoch ):

    print( '\nEpoch %d' % epoch )
    t0 = time.time()

    values = [epoch]




    ################ Training on Train Set ################


    for i, data in enumerate( train_loader, 0 ):

        # ------------ Prapare Variables ------------
        batch_x, batch_y = data
      
        '''
        att=np.array(batch_y[:,1:12])
        att=np.where(att>0.5,1,0)
        att=torch.FloatTensor(att)
        att=Variable(att)
        '''


        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        
        if use_gpu:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
       


        # ------------ Training model ------------
        model.train()  # unlock model
        optimizer.zero_grad()


        fake_labels = model( batch_x )
        real_score=batch_y[:,0]
        real_score = real_score-1
        loss = criterion_CE(fake_labels.cuda(), real_score.type(torch.LongTensor).cuda())#type(torch.LongTensor)


        loss.backward()
        optimizer.step()
        







        # ------------ Preparation for Evaluation on Train Set ------------
        #fake_labels = fake_labels[:, 0].squeeze()            # output选择第0列
        batch_y = batch_y[:, 0].squeeze()          # batch_y选择第0列


        
        #fake_labels = fake_labels.cpu().data.numpy() if use_gpu else fake_labels.data.numpy()
        #batch_y = batch_y.cpu().data.numpy() if use_gpu else batch_y.data.numpy()
		
        start_test = True
        fake_labels = torch.softmax(fake_labels, dim=-1)
        if start_test:
                all_outputs = fake_labels.data.float()
                all_labels = real_score.data.float()
                start_test = False
        else:
            all_outputs = torch.cat((all_outputs, fake_labels.data.float()), 0)
            all_labels = torch.cat((all_labels , real_score.data.float()), 0)
        

    _, predict = torch.max(all_outputs, 1)
        
    ################ Evaluation on Train Set ################
    if epoch % 10 == 0:
			
        hit_num = (predict == all_labels).sum().item()
        sample_num = predict.size(0)
        print("epoch: {}; current acc: {}".format(epoch, hit_num / float(sample_num)))
		
    values.append(hit_num / float(sample_num))


    ################ Evaluation on Valid/Test Set ################

    model.eval()  # evaluation mode

    accuracy = model_eval( model, valid_loader, use_gpu )
    values.append( accuracy )
    print( 'Valid Set\taccuracy=%f' % accuracy )


    scheduler.step( accuracy )  # 执行学习率衰减


    accuracy = model_eval( model, test_loader, use_gpu )
    values.append( accuracy )

    print( 'Test Set\taccuracy=%f' % accuracy )


    print( 'Done in %.2fs' % ( time.time() - t0 ) )






    ################ Writing Record ################

    temp = pd.DataFrame( [values], columns=['epoch', 'train_acc',
        'valid_accuracy', 'test_accuracy'] )
    record = record.append( temp )




record.to_csv( 'record_test.csv', index=False )
torch.save( model.state_dict(), 'model_param.pkl' )





