from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch import optim
from truenet_tumseg.truenet_tumorseg import (truenet_tumseg_loss_functions,
                                             truenet_tumseg_model, truenet_tumseg_train)
from truenet_tumseg.utils import truenet_tumseg_utils

################################################################################################
# Truenet_tumseg main training function
# Vaanathi Sundaresan
# 01-05-2021, Oxford
################################################################################################


def main(sub_name_dicts, tr_params, aug=True, save_cp=True, save_wei=True, save_case='last',
         verbose=True, dir_cp=None):
    """
    The main training function
    :param sub_name_dicts: list of dictionaries containing training filpaths
    :param tr_params: dictionary of training parameters
    :param aug: bool, perform data augmentation
    :param save_cp: bool, save checkpoints
    :param save_wei: bool, if False, the whole model will be saved
    :param save_case: str, condition for saving the checkpoint
    :param verbose: bool, display debug messages
    :param dir_cp: str, directory for saving model/weights
    :return: trained model
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    assert len(sub_name_dicts) >= 5, "Number of distinct subjects for training cannot be less than 5"
    
    optim_type = tr_params['Optimizer']  # adam, sgd
    milestones = tr_params['LR_Milestones']  # list of integers [1, N]
    gamma = tr_params['LR_red_factor']  # scalar (0,1)
    lrt = tr_params['Learning_rate']  # scalar (0,1)
    req_plane = tr_params['Acq_plane']  # string ('axial', 'sagittal', 'coronal', 'all')
    train_prop = tr_params['Train_prop']  # scale (0,1)
    nclass = tr_params['Nclass']

    modalities = tr_params['Num_modalities']
    if any(modalities) > 1:
        raise ValueError('Only 1 and 0 allowed. At least one of the modalities must be selected to be 1')
    nchannels = sum(modalities)

    if nclass == 2:
        criterion = truenet_tumseg_loss_functions.CombinedLoss()
    else:
        criterion = truenet_tumseg_loss_functions.CombinedMultiLoss(nclasses=nclass)
    
    if verbose:
        print('Found' + str(len(sub_name_dicts)) + 'subjects', flush=True)
        
    num_val_subs = max(int(len(sub_name_dicts) * (1-train_prop)), 1)
    train_name_dicts, val_name_dicts, val_ids = truenet_tumseg_utils.select_train_val_names(sub_name_dicts,
                                                                                            num_val_subs)
    if type(milestones) != list:
        milestones = [milestones]

    models = []
    if req_plane == 'all' or req_plane == 'axial':
        model_axial = truenet_tumseg_model.TrUENetTumSeg(n_channels=nchannels, n_classes=nclass, init_channels=64,
                                                         plane='axial')
        model_axial.to(device=device)
        model_axial = nn.DataParallel(model_axial)
        print('Total number of Axial model to train: ', str(sum([p.numel() for p in model_axial.parameters()])),
              flush=True)
        if optim_type == 'adam':
            epsilon = tr_params['Epsilon']
            optimizer_axial = optim.Adam(filter(lambda p: p.requires_grad, model_axial.parameters()), lr=lrt,
                                         eps=epsilon)
        elif optim_type == 'sgd':
            moment = tr_params['Momentum']
            optimizer_axial = optim.SGD(filter(lambda p: p.requires_grad, model_axial.parameters()), lr=lrt,
                                        momentum=moment)
        else:
            raise ValueError("Invalid optimiser choice provided! Valid options: 'adam', 'sgd'")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_axial, milestones, gamma=gamma, last_epoch=-1)
        model_axial = truenet_tumseg_train.train_truenet(train_name_dicts, val_name_dicts, model_axial,
                                                         criterion, optimizer_axial, scheduler, tr_params,
                                                         device, mode='axial', augment=aug, save_checkpoint=save_cp,
                                                         save_weights=save_wei, save_case=save_case, verbose=verbose,
                                                         dir_checkpoint=dir_cp)
        models.append(model_axial)
    if req_plane == 'all' or req_plane == 'sagittal':
        model_sagittal = truenet_tumseg_model.TrUENetTumSeg(n_channels=nchannels, n_classes=nclass,
                                                            init_channels=64, plane='sagittal')
        model_sagittal.to(device=device)
        model_sagittal = nn.DataParallel(model_sagittal)
        print('Total number of Sagittal model to train: ', str(sum([p.numel() for p in model_sagittal.parameters()])),
              flush=True)
        if optim_type == 'adam':
            epsilon = tr_params['Epsilon']
            optimizer_sagittal = optim.Adam(filter(lambda p: p.requires_grad, model_sagittal.parameters()), lr=lrt,
                                            eps=epsilon)
        elif optim_type == 'sgd':
            moment = tr_params['Momentum']
            optimizer_sagittal = optim.SGD(filter(lambda p: p.requires_grad, model_sagittal.parameters()), lr=lrt,
                                           momentum=moment)
        else:
            raise ValueError("Invalid optimiser choice provided! Valid options: 'adam', 'sgd'")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_sagittal, milestones, gamma=gamma, last_epoch=-1)
        model_sagittal = truenet_tumseg_train.train_truenet(train_name_dicts, val_name_dicts, model_sagittal,
                                                            criterion, optimizer_sagittal, scheduler, tr_params,
                                                            device, mode='sagittal', augment=aug,
                                                            save_checkpoint=save_cp, save_weights=save_wei,
                                                            save_case=save_case, verbose=verbose, dir_checkpoint=dir_cp)
        models.append(model_sagittal)
    if req_plane == 'all' or req_plane == 'coronal':
        model_coronal = truenet_tumseg_model.TrUENetTumSeg(n_channels=nchannels, n_classes=nclass,
                                                           init_channels=64, plane='coronal')
        model_coronal.to(device=device)
        model_coronal = nn.DataParallel(model_coronal)
        print('Total number of Coronal model to train: ', str(sum([p.numel() for p in model_coronal.parameters()])),
              flush=True)
        if optim_type == 'adam':
            epsilon = tr_params['Epsilon']
            optimizer_coronal = optim.Adam(filter(lambda p: p.requires_grad, model_coronal.parameters()), lr=lrt,
                                           eps=epsilon)
        elif optim_type == 'sgd':
            moment = tr_params['Momentum']
            optimizer_coronal = optim.SGD(filter(lambda p: p.requires_grad, model_coronal.parameters()), lr=lrt,
                                          momentum=moment)
        else:
            raise ValueError("Invalid optimiser choice provided! Valid options: 'adam', 'sgd'")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_coronal, milestones, gamma=gamma, last_epoch=-1)
        model_coronal = truenet_tumseg_train.train_truenet(train_name_dicts, val_name_dicts, model_coronal,
                                                           criterion, optimizer_coronal, scheduler, tr_params,
                                                           device, mode='coronal', augment=aug, save_checkpoint=save_cp,
                                                           save_weights=save_wei, save_case=save_case, verbose=verbose,
                                                           dir_checkpoint=dir_cp)
        models.append(model_coronal)
    if req_plane == 'all' or req_plane == 'tc':
        criterion = truenet_tumseg_loss_functions.CombinedLoss()
        model_tc = truenet_tumseg_model.TrUENetTumSeg(n_channels=nchannels, n_classes=2, init_channels=64,
                                                      plane='axial')
        model_tc.to(device=device)
        model_tc = nn.DataParallel(model_tc)
        print('Total number of Tumour core model to train: ', str(sum([p.numel() for p in model_tc.parameters()])),
              flush=True)
        if optim_type == 'adam':
            epsilon = tr_params['Epsilon']
            optimizer_tc = optim.Adam(filter(lambda p: p.requires_grad, model_tc.parameters()), lr=lrt, eps=epsilon)
        elif optim_type == 'sgd':
            moment = tr_params['Momentum']
            optimizer_tc = optim.SGD(filter(lambda p: p.requires_grad, model_tc.parameters()), lr=lrt, momentum=moment)
        else:
            raise ValueError("Invalid optimiser choice provided! Valid options: 'adam', 'sgd'")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_tc, milestones, gamma=gamma, last_epoch=-1)
        model_tc = truenet_tumseg_train.train_truenet(train_name_dicts, val_name_dicts, model_tc, criterion,
                                                      optimizer_tc, scheduler, tr_params, device, mode='tc',
                                                      augment=aug, save_checkpoint=save_cp, save_weights=save_wei,
                                                      save_case=save_case, verbose=verbose, dir_checkpoint=dir_cp)
        models.append(model_tc)
    return models


