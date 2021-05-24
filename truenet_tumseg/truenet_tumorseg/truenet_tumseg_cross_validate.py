from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import nibabel as nib
from truenet_tumseg.truenet_tumorseg import (truenet_tumseg_loss_functions,
                                             truenet_tumseg_model, truenet_tumseg_train,
                                             truenet_tumseg_evaluate, truenet_tumseg_data_postprocessing)
from truenet_tumseg.utils import (truenet_tumseg_utils)

################################################################################################
# Truenet cross-validation function
# Vaanathi Sundaresan
# 09-03-2021, Oxford
################################################################################################


def main(sub_name_dicts, cv_params, aug=True, intermediate=False, save_cp=False, save_wei=True,
         save_case='best', verbose=True, dir_cp=None, output_dir=None):
    """
    The main function for leave-one-out validation of Truenet
    :param sub_name_dicts: list of dictionaries containing subject filepaths
    :param cv_params: dictionary of LOO paramaters
    :param aug: bool, whether to do data augmentation
    :param intermediate: bool, whether to save intermediate results
    :param save_cp: bool, whether to save checkpoint
    :param save_wei: bool, whether to save weights alone or the full model
    :param save_case: str, condition for saving the CP
    :param verbose: bool, display debug messages
    :param dir_cp: str, filepath for saving the model
    :param output_dir: str, filepath for saving the output predictions
    """

    assert len(sub_name_dicts) >= 5, "Number of distinct subjects for Leave-one-out validation cannot be less than 5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nclass = cv_params['Nclass']

    modalities = cv_params['Num_modalities']
    if any(modalities) > 1:
        raise ValueError('Only 1 and 0 allowed. At least one of the modalities must be selected to be 1')
    nchannels = sum(modalities)

    if nclass == 2:
        criterion = truenet_tumseg_loss_functions.CombinedLoss()
    else:
        criterion = truenet_tumseg_loss_functions.CombinedMultiLoss(nclasses=nclass)

    optim_type = cv_params['Optimizer']  # adam, sgd
    milestones = cv_params['LR_Milestones']  # list of integers [1, N]
    gamma = cv_params['LR_red_factor']  # scalar (0,1)
    lrt = cv_params['Learning_rate']  # scalar (0,1)
    train_prop = cv_params['Train_prop']  # scalar (0,1)
    fold = cv_params['fold']  # scalar [1, N]
    res_fold = cv_params['res_fold']  # scalar [1, N]
    postproc = cv_params['Postproc']

    res_fold = res_fold - 1

    test_subs_per_fold = max(int(np.round(len(sub_name_dicts) / fold)), 1)

    if type(milestones) != list:
        milestones = [milestones]

    if verbose:
        print('Found' + str(len(sub_name_dicts)) + 'subjects', flush=True)

    for fld in range(res_fold, fold):
        model_axial = truenet_tumseg_model.TrUENetTumSeg(n_channels=nchannels, n_classes=nclass,
                                                         init_channels=64, plane='axial')
        model_sagittal = truenet_tumseg_model.TrUENetTumSeg(n_channels=nchannels, n_classes=nclass,
                                                            init_channels=64, plane='sagittal')
        model_coronal = truenet_tumseg_model.TrUENetTumSeg(n_channels=nchannels, n_classes=nclass,
                                                           init_channels=64, plane='coronal')

        model_axial.to(device=device)
        model_sagittal.to(device=device)
        model_coronal.to(device=device)
        model_axial = nn.DataParallel(model_axial)
        model_sagittal = nn.DataParallel(model_sagittal)
        model_coronal = nn.DataParallel(model_coronal)
        if postproc is True:
            model_tc = truenet_tumseg_model.TrUENetTumSeg(n_channels=nchannels, n_classes=2,
                                                          init_channels=64, plane='coronal')
            model_tc.to(device=device)
            model_tc = nn.DataParallel(model_tc)

        if optim_type == 'adam':
            epsilon = cv_params['Epsilon']
            optimizer_axial = optim.Adam(model_axial.parameters(), lr=lrt, eps=epsilon)
            optimizer_sagittal = optim.Adam(model_sagittal.parameters(), lr=lrt, eps=epsilon)
            optimizer_coronal = optim.Adam(model_coronal.parameters(), lr=lrt, eps=epsilon)
            if postproc is True:
                optimizer_tc = optim.Adam(model_tc.parameters(), lr=lrt, eps=epsilon)
        elif optim_type == 'sgd':
            moment = cv_params['Momentum']
            optimizer_axial = optim.SGD(model_axial.parameters(), lr=lrt, momentum=moment)
            optimizer_sagittal = optim.SGD(model_sagittal.parameters(), lr=lrt, momentum=moment)
            optimizer_coronal = optim.SGD(model_coronal.parameters(), lr=lrt, momentum=moment)
            if postproc is True:
                optimizer_tc = optim.Adam(model_tc.parameters(), lr=lrt, momentum=moment)
        else:
            raise ValueError("Invalid optimiser choice provided! Valid options: 'adam', 'sgd'")

        if verbose:
            print('Training models for fold ' + str(fld+1) + '...', flush=True)

        if fld == (fold - 1):
            test_ids = np.arange(fld * test_subs_per_fold, len(sub_name_dicts))
            test_sub_dicts = sub_name_dicts[test_ids]
        else:
            test_ids = np.arange(fld * test_subs_per_fold, (fld+1) * test_subs_per_fold)
            test_sub_dicts = sub_name_dicts[test_ids]

        rem_sub_ids = np.setdiff1d(np.arange(len(sub_name_dicts)), test_ids)
        rem_sub_name_dicts = [sub_name_dicts[idx] for idx in rem_sub_ids]
        num_val_subs = max(int(len(sub_name_dicts) * (1-train_prop)), 1)
        train_name_dicts, val_name_dicts, val_ids = truenet_tumseg_utils.select_train_val_names(rem_sub_name_dicts,
                                                                                                num_val_subs)
        if save_cp:
            dir_cp = os.path.join(dir_cp, 'fold' + str(fld+1) + '_models')
            os.mkdir(dir_cp)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_axial, milestones, gamma=gamma, last_epoch=-1)
        model_axial = truenet_tumseg_train.train_truenet(train_name_dicts, val_name_dicts, model_axial,
                                                         criterion, optimizer_axial, scheduler, cv_params,
                                                         device, mode='axial', augment=aug,
                                                         save_checkpoint=save_cp, save_weights=save_wei,
                                                         save_case=save_case, verbose=verbose, dir_checkpoint=dir_cp)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_sagittal, milestones, gamma=gamma, last_epoch=-1)
        model_sagittal = truenet_tumseg_train.train_truenet(train_name_dicts, val_name_dicts, model_axial,
                                                            criterion, optimizer_sagittal, scheduler,
                                                            cv_params, device, mode='sagittal', augment=aug,
                                                            save_checkpoint=save_cp, save_weights=save_wei,
                                                            save_case=save_case, verbose=verbose, dir_checkpoint=dir_cp)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_coronal, milestones, gamma=gamma, last_epoch=-1)
        model_coronal = truenet_tumseg_train.train_truenet(train_name_dicts, val_name_dicts, model_axial,
                                                           criterion, optimizer_coronal, scheduler, cv_params,
                                                           device, mode='coronal', augment=aug,
                                                           save_checkpoint=save_cp, save_weights=save_wei,
                                                           save_case=save_case, verbose=verbose, dir_checkpoint=dir_cp)

        if postproc is True:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer_tc, milestones, gamma=gamma, last_epoch=-1)
            model_tc = truenet_tumseg_train.train_truenet(train_name_dicts, val_name_dicts, model_axial,
                                                          criterion, optimizer_tc, scheduler, cv_params,
                                                          device, mode='tc', augment=aug,
                                                          save_checkpoint=save_cp, save_weights=save_wei,
                                                          save_case=save_case, verbose=verbose, dir_checkpoint=dir_cp)

        if verbose:
            print('Predicting outputs for subjects in fold ' + str(fld+1) + '...', flush=True)

        for sub in range(len(test_sub_dicts)):
            if verbose:
                print('Predicting for subject ' + str(sub + 1) + '...', flush=True)
            test_sub_dict = [test_sub_dicts[sub]]
            basename = test_sub_dict[0]['basename']
            probs_combined = []
            flair_path = test_sub_dict[0]['flair_path']
            flair_hdr = nib.load(flair_path).header
            probs_axial = truenet_tumseg_evaluate.evaluate_truenet(test_sub_dict, model_axial, cv_params,
                                                                   device, mode='axial', verbose=verbose)
            probs_axial = truenet_tumseg_data_postprocessing.resize_to_original_size(probs_axial, test_sub_dict,
                                                                                     plane='axial')
            probs_combined.append(probs_axial)

            if intermediate:
                save_path = os.path.join(output_dir, 'Predicted_probmap_truenet_' + basename + '_axial.nii.gz')
                preds_axial = truenet_tumseg_data_postprocessing.get_final_3dvolumes(probs_axial, test_sub_dict)
                if verbose:
                    print('Saving the intermediate Axial prediction ...', flush=True)

                newhdr = flair_hdr.copy()
                newobj = nib.nifti1.Nifti1Image(preds_axial, None, header=newhdr)
                nib.save(newobj, save_path)

            probs_sagittal = truenet_tumseg_evaluate.evaluate_truenet(test_sub_dict, model_sagittal, cv_params,
                                                                      device, mode='sagittal', verbose=verbose)
            probs_sagittal = truenet_tumseg_data_postprocessing.resize_to_original_size(probs_sagittal, test_sub_dict,
                                                                                        plane='sagittal')
            probs_combined.append(probs_sagittal)

            if intermediate:
                save_path = os.path.join(output_dir, 'Predicted_probmap_truenet_' + basename + '_sagittal.nii.gz')
                preds_sagittal = truenet_tumseg_data_postprocessing.get_final_3dvolumes(probs_sagittal, test_sub_dict)
                if verbose:
                    print('Saving the intermediate Sagittal prediction ...', flush=True)

                newhdr = flair_hdr.copy()
                newobj = nib.nifti1.Nifti1Image(preds_sagittal, None, header=newhdr)
                nib.save(newobj, save_path)

            probs_coronal = truenet_tumseg_evaluate.evaluate_truenet(test_sub_dict, model_coronal, cv_params,
                                                                     device, mode='coronal', verbose=verbose)
            probs_coronal = truenet_tumseg_data_postprocessing.resize_to_original_size(probs_coronal, test_sub_dict,
                                                                                       plane='coronal')
            probs_combined.append(probs_coronal)

            if intermediate:
                save_path = os.path.join(output_dir, 'Predicted_probmap_truenet_' + basename + '_coronal.nii.gz')
                preds_coronal = truenet_tumseg_data_postprocessing.get_final_3dvolumes(probs_coronal, test_sub_dict)
                if verbose:
                    print('Saving the intermediate Coronal prediction ...', flush=True)

                newhdr = flair_hdr.copy()
                newobj = nib.nifti1.Nifti1Image(preds_coronal, None, header=newhdr)
                nib.save(newobj, save_path)

            probs_combined = np.array(probs_combined)
            prob_mean = np.mean(probs_combined, axis=0)

            pred_mean = truenet_tumseg_data_postprocessing.get_final_3dvolumes(prob_mean, test_sub_dict)

            pred_output = np.argmax(prob_mean, axis=3)
            pred_output = truenet_tumseg_data_postprocessing.get_final_3dvolumes(pred_output, test_sub_dict)

            if postproc is False:
                save_path = os.path.join(output_dir, 'Predicted_probmap_truenet_' + basename + '_final.nii.gz')
                if verbose:
                    print('Saving the final prediction ...', flush=True)

                newhdr = flair_hdr.copy()
                newobj = nib.nifti1.Nifti1Image(pred_mean, None, header=newhdr)
                nib.save(newobj, save_path)

                save_path = os.path.join(output_dir, 'Predicted_output_truenet_' + basename + '_final.nii.gz')
                if verbose:
                    print('Saving the final output segmented map ...', flush=True)

                newhdr = flair_hdr.copy()
                newobj = nib.nifti1.Nifti1Image(pred_output, None, header=newhdr)
                nib.save(newobj, save_path)
            else:
                probs_tc = truenet_tumseg_evaluate.evaluate_truenet(test_sub_dict, model_tc, cv_params,
                                                                    device, mode='tc', verbose=verbose)
                probs_tc = truenet_tumseg_data_postprocessing.resize_to_original_size(probs_tc, test_sub_dict,
                                                                                      plane='tc')
                preds_tc = truenet_tumseg_data_postprocessing.get_final_3dvolumes(probs_tc, test_sub_dict)
                pred_final = truenet_tumseg_utils.post_processing_including_tc(pred_output, preds_tc)

                if intermediate:
                    save_path = os.path.join(output_dir, 'Predicted_probmap_truenet_' + basename + '_tc.nii.gz')
                    if verbose:
                        print('Saving the intermediate Tumour core prediction ...', flush=True)

                    newhdr = flair_hdr.copy()
                    newobj = nib.nifti1.Nifti1Image(preds_tc, None, header=newhdr)
                    nib.save(newobj, save_path)

                save_path = os.path.join(output_dir, 'Predicted_output_truenet_' + basename + '_final.nii.gz')
                if verbose:
                    print('Saving the final output segmented map ...', flush=True)

                newhdr = flair_hdr.copy()
                newobj = nib.nifti1.Nifti1Image(pred_final, None, header=newhdr)
                nib.save(newobj, save_path)

        if verbose:
            print('Fold ' + str(fld+1) + ': complete!', flush=True)

    if verbose:
        print('Cross-validation done!', flush=True)