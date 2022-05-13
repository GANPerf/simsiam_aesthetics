import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

class  (_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(ATTLoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self,fake_score,att,tag):
        sig_score=func.sigmoid(fake_score-0.5)
        
        #att1=att.sum(1)
        #att0=11-att1
        
        positive=nn.Threshold(0, 0) #if x<=0,x=0 else x=x
        loss=att*(positive(1-2*sig_score))+(1-att)*(positive(2*sig_score-1))
        loss=loss*tag
        return loss.sum()/tag.sum()

class RANKLoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(RANKLoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self,fake_score,real_score):
        loss=0
        for i in range(real_score.size(0)):
            for j in range(real_score.size(0)):
                diff1=(real_score[i]-real_score[j])
                diff2=(fake_score[i]-fake_score[j])        
                loss+=(diff1-diff2)*(diff1-diff2)    
        return loss/(2*real_score.size(0))