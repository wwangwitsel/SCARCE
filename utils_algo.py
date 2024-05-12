import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## SCARCE loss
def logistic_loss(pred):
    negative_logistic = nn.LogSigmoid()
    logistic = -1. * negative_logistic(pred)
    return logistic

def SCARCE_loss(outputs, labels, device):
    comple_label_mat = torch.zeros_like(outputs)
    comple_label_mat[torch.arange(comple_label_mat.shape[0]), labels.long()] = 1
    comple_label_mat = comple_label_mat.to(device)
    pos_loss = logistic_loss(outputs)
    neg_loss = logistic_loss(-outputs)
    neg_data_mat = comple_label_mat.float()
    unlabel_data_mat = torch.ones_like(neg_data_mat)
    # calculate negative label loss of negative data
    neg_loss_neg_data_mat = neg_loss * neg_data_mat
    tmp1 = neg_data_mat.sum(dim=0)
    tmp1[tmp1 == 0.] = 1.
    neg_loss_neg_data_vec = neg_loss_neg_data_mat.sum(dim=0) / tmp1
    # calculate positive label loss of unlabeled data
    pos_loss_unlabel_data_mat = pos_loss * unlabel_data_mat
    tmp2 = unlabel_data_mat.sum(dim=0)
    tmp2[tmp2 == 0.] = 1.
    pos_loss_unlabel_data_vec = pos_loss_unlabel_data_mat.sum(dim=0) / tmp2
    # calculate positive label loss of negative data
    pos_loss_neg_data_mat = pos_loss * neg_data_mat
    pos_loss_neg_data_vec = pos_loss_neg_data_mat.sum(dim=0) / tmp1
    # calculate final loss
    prior_vec = 1. / outputs.shape[1] * torch.ones(outputs.shape[1])
    prior_vec = prior_vec.to(device)
    ccp = 1. - prior_vec
    loss1 = (ccp * neg_loss_neg_data_vec).sum()
    unmax_loss_vec = pos_loss_unlabel_data_vec - ccp * pos_loss_neg_data_vec
    max_loss_vec = torch.abs(unmax_loss_vec)
    loss2 = max_loss_vec.sum()
    loss = loss1 + loss2
    return loss

def accuracy_check(loader, model, device):
    sm = F.softmax
    total, num_samples = 0, 0
    for images, labels in loader:
        labels, images = labels.to(device), images.to(device)
        outputs = model(images)
        sm_outputs = sm(outputs, dim=1)
        _, predicted = torch.max(sm_outputs.data, 1)
        total += (predicted == labels).sum().item()
        num_samples += labels.size(0)
    return 100 * total / num_samples

def chosen_loss_c(f, K, labels, ccp, meta_method, device):
    class_loss_torch = None
    if meta_method=='SCARCE':
        final_loss = SCARCE_loss(outputs=f, labels=labels, device=device)
    return final_loss, class_loss_torch
