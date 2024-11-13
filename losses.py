import torch

from fvcore.nn import sigmoid_focal_loss

import torch.nn.functional as F


def reduce_masked_mean(x, mask, dim=None, keepdim=False, EPS=1e-8):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    for (a,b) in zip(x.size(), mask.size()):
        # if not b==1:
        assert(a==b) # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x*mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS+torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer/denom
    return mean



def balanced_mse_loss(pred, gt, valid=None):
    pos_mask = gt.gt(0.5).float()
    neg_mask = gt.lt(0.5).float()
    if valid is None:
        valid = torch.ones_like(pos_mask)
    mse_loss = F.mse_loss(pred, gt, reduction='none')
    pos_loss = reduce_masked_mean(mse_loss, pos_mask*valid)
    neg_loss = reduce_masked_mean(mse_loss, neg_mask*valid)
    loss = (pos_loss + neg_loss)*0.5
    return loss


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight=2.13):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]).to('cuda'), reduction='none')
        
    def forward(self, ypred, ytgt, valid=None):

        loss = self.loss_fn(ypred, ytgt)
        if valid is not None:
            loss = reduce_masked_mean(loss, valid)
        else:
            loss = loss.mean()        
        return loss
    
class SigmoidFocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=-1.0,
        gamma=2.0,
        reduction='mean'
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, label, valid):
        loss = sigmoid_focal_loss(pred, label, self.alpha, self.gamma, self.reduction)
        loss = loss*valid
        
        return loss.mean() 
