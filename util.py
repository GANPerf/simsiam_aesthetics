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





def get_rank(x):

    arg = np.argsort(x)
    rank = np.zeros_like(x)

    for i in range( len(x) ):
        rank[ arg[i] ] = i

    return rank






class MyDataset(Dataset):

    def __init__(self, input_dir, dataset_str):

        self.input_dir = input_dir
        self.dataset_str = dataset_str
        self.df = pd.read_csv( '%s/label/%s_label.csv' % ( input_dir, dataset_str ) )

        if dataset_str == 'train':
            self.preprocess = transforms.Compose( [
                transforms.RandomSizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
            ] )
        else:
            self.preprocess = transforms.Compose( [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
            ] )

        '''
        # 将某些attr从[-1, 1]变换到[0, 1]
        attrs = ['BalacingElements', 'ColorHarmony', 'Content', 'DoF', 'Light', 'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor']
        self.df[attrs] = self.df[attrs] / 2.0 + 0.5
        '''




    def __getitem__(self, index):

        img_name = self.df.loc[index, 'image']
        img_path = '%s/image/%s' % ( self.input_dir, img_name )

        X = Image.open( img_path )

        if len( np.asarray(X).shape ) == 2:
            X = X.convert( 'RGB' )

        X = self.preprocess( X )

        y = self.df.iloc[index, 1:].values.astype(float)
        y = torch.from_numpy(y).float()
        #y = self.df.iloc[index]

        return X, y




    def __len__(self):
        return len( self.df )










def model_eval( model, data_loader, use_gpu ):


    for i, data in enumerate( data_loader, 0 ):

        batch_x, batch_y = data
        batch_x, batch_y = Variable(batch_x), batch_y[:,0]  # batch_y选择第0列

        if use_gpu:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        output = model( batch_x )
			
		
        start_test = True
        fake_labels = torch.softmax(output, dim=-1)
        if start_test:
            all_outputs = fake_labels.data.float()
            all_labels = batch_y.data.float()
            start_test = False
        else:
            all_outputs = torch.cat((all_outputs, fake_labels.data.float()), 0)
            all_labels = torch.cat((all_labels , batch_y.data.float()), 0)
	
    _, predict = torch.max(all_outputs, 1)
    predict = predict + 1
	
    #print(all_labels)
    #print("!!!!")
    #print(predict)


    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).item() / (float(all_labels.size()[0]))

    return accuracy









def sample_real_labels( batch_size, filename ):
    
    df = pd.read_csv( filename ).sample( batch_size )
    y = df.iloc[:, 1:].values
    y = torch.from_numpy(y).float()

    return y

