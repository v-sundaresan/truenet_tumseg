from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import os
import nibabel as nib
from truenet_tumseg.truenet_tumorseg import (truenet_tumseg_model, truenet_tumseg_evaluate,
                                             truenet_tumseg_data_postprocessing)
from truenet_tumseg.utils import truenet_tumseg_utils

################################################################################################
# Truenet main test function
# Vaanathi Sundaresan
# 09-03-2021, Oxford
################################################################################################


def main(sub_name_dicts, eval_params, intermediate=False, model_dir=None,
         load_case='last', output_dir=None, verbose=False):
    '''
    The main function for testing Truenet
    :param sub_name_dicts: list of dictionaries containing subject filepaths
    :param eval_params: dictionary of evaluation parameters
    :param intermediate: bool, whether to save intermediate results
    :param model_dir: str, filepath containing the test model
    :param load_case: str, condition for loading the checkpoint
    :param output_dir: str, filepath for saving the output predictions
    :param verbose: bool, display debug messages
    '''
    assert len(sub_name_dicts) > 0, "There must be at least 1 subject for testing."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nclass = eval_params['Nclass']
    pretrained = eval_params['Pretrained']
    postproc = eval_params['Postproc']

    modalities = eval_params['Num_modalities']
    if any(modalities) > 1:
        raise ValueError('Only 1 and 0 allowed. At least one of the modalities must be selected to be 1')
    nchannels = sum(modalities)

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
                                                      init_channels=64, plane='tc')
        model_tc.to(device=device)
        model_tc = nn.DataParallel(model_tc)

    if pretrained:
        if load_case == 'last':
            model_path = os.path.join(model_dir, 'Truenet_model_beforeES_axial.pth')
            model_axial = truenet_tumseg_utils.loading_model(model_path, model_axial, mode='full_model')

            model_path = os.path.join(model_dir, 'Truenet_model_beforeES_sagittal.pth')
            model_sagittal = truenet_tumseg_utils.loading_model(model_path, model_sagittal, mode='full_model')

            model_path = os.path.join(model_dir, 'Truenet_model_beforeES_coronal.pth')
            model_coronal = truenet_tumseg_utils.loading_model(model_path, model_coronal, mode='full_model')

            if postproc is True:
                model_path = os.path.join(model_dir, 'Truenet_model_beforeES_tc.pth')
                model_tc = truenet_tumseg_utils.loading_model(model_path, model_tc, mode='full_model')
        elif load_case == 'best':
            model_path = os.path.join(model_dir, 'Truenet_model_bestdice_axial.pth')
            model_axial = truenet_tumseg_utils.loading_model(model_path, model_axial, mode='full_model')

            model_path = os.path.join(model_dir, 'Truenet_model_bestdice_sagittal.pth')
            model_sagittal = truenet_tumseg_utils.loading_model(model_path, model_sagittal, mode='full_model')

            model_path = os.path.join(model_dir, 'Truenet_model_bestdice_coronal.pth')
            model_coronal = truenet_tumseg_utils.loading_model(model_path, model_coronal, mode='full_model')

            if postproc is True:
                model_path = os.path.join(model_dir, 'Truenet_model_bestdice_tc.pth')
                model_tc = truenet_tumseg_utils.loading_model(model_path, model_tc, mode='full_model')
        elif load_case == 'everyN':
            cpn = eval_params['EveryN']
            try:
                model_path = os.path.join(model_dir, 'Truenet_model_epoch' + str(cpn) + '_axial.pth')
                model_axial = truenet_tumseg_utils.loading_model(model_path, model_axial, mode='full_model')

                model_path = os.path.join(model_dir, 'Truenet_model_epoch' + str(cpn) + '_sagittal.pth')
                model_sagittal = truenet_tumseg_utils.loading_model(model_path, model_sagittal, mode='full_model')

                model_path = os.path.join(model_dir, 'Truenet_model_epoch' + str(cpn) + '_coronal.pth')
                model_coronal = truenet_tumseg_utils.loading_model(model_path, model_coronal, mode='full_model')

                if postproc is True:
                    model_path = os.path.join(model_dir, 'Truenet_model_epoch' + str(cpn) + '_tc.pth')
                    model_tc = truenet_tumseg_utils.loading_model(model_path, model_tc, mode='full_model')
            except ImportError:
                raise ImportError(
                    'Incorrect N value provided for the available pretrained models')
        else:
            raise ValueError("Invalid saving condition provided! Valid options: best, specific, last")
    else:
        model_name = eval_params['Modelname']

        try:
            model_path = os.path.join(model_dir, model_name + '_axial.pth')
            model_axial = truenet_tumseg_utils.loading_model(model_path, model_axial)

            model_path = os.path.join(model_dir, model_name + '_sagittal.pth')
            model_sagittal = truenet_tumseg_utils.loading_model(model_path, model_sagittal)

            model_path = os.path.join(model_dir, model_name + '_coronal.pth')
            model_coronal = truenet_tumseg_utils.loading_model(model_path, model_coronal)

            if postproc is True:
                model_path = os.path.join(model_dir, model_name + '_tc.pth')
                model_tc = truenet_tumseg_utils.loading_model(model_path, model_tc)
        except:
            try:
                model_path = os.path.join(model_dir, model_name + '_axial.pth')
                model_axial = truenet_tumseg_utils.loading_model(model_path, model_axial, mode='full_model')

                model_path = os.path.join(model_dir, model_name + '_sagittal.pth')
                model_sagittal = truenet_tumseg_utils.loading_model(model_path, model_sagittal, mode='full_model')

                model_path = os.path.join(model_dir, model_name + '_coronal.pth')
                model_coronal = truenet_tumseg_utils.loading_model(model_path, model_coronal, mode='full_model')

                if postproc is True:
                    model_path = os.path.join(model_dir, model_name + '_tc.pth')
                    model_tc = truenet_tumseg_utils.loading_model(model_path, model_tc, mode='full_model')
            except ImportError:
                if postproc is True:
                    raise ImportError('In directory ' + model_dir + ', ' + model_name + '_axial.pth or ' +
                                      model_name + '_sagittal.pth or ' + model_name + '_coronal.pth or ' +
                                      model_name + '_tc.pth does not appear to be a valid model file')
                else:
                    raise ImportError('In directory ' + model_dir + ', ' + model_name + '_axial.pth or' +
                                      model_name + '_sagittal.pth or' + model_name + '_coronal.pth ' +
                                      'does not appear to be a valid model file')

    if verbose:
        print('Found' + str(len(sub_name_dicts)) + 'subjects', flush=True)
    for sub in range(len(sub_name_dicts)):
        if verbose:
            print('Predicting output for subject ' + str(sub+1) + '...', flush=True)
            
        test_sub_dict = [sub_name_dicts[sub]]
        basename = test_sub_dict[0]['basename']
        
        probs_combined = []
        flair_path = test_sub_dict[0]['flair_path']
        flair_hdr = nib.load(flair_path).header
        probs_axial = truenet_tumseg_evaluate.evaluate_truenet(test_sub_dict, model_axial, eval_params, device,
                                                               mode='axial', verbose=verbose)
        probs_axial = truenet_tumseg_data_postprocessing.resize_to_original_size(probs_axial, test_sub_dict,
                                                                                 plane='axial')
        probs_combined.append(probs_axial)
        
        if intermediate:
            save_path = os.path.join(output_dir, 'Predicted_probmap_truenet_' + basename + '_axial.nii.gz')
            preds_axial = truenet_tumseg_data_postprocessing.get_final_3dvolumes(np.argmax(probs_axial, axis=3), test_sub_dict)
            if verbose:
                print('Saving the intermediate Axial prediction ...', flush=True)
            
            newhdr = flair_hdr.copy()
            newobj = nib.nifti1.Nifti1Image(preds_axial, None, header=newhdr)
            nib.save(newobj, save_path)            
        
        probs_sagittal = truenet_tumseg_evaluate.evaluate_truenet(test_sub_dict, model_sagittal, eval_params,
                                                                  device, mode='sagittal', verbose=verbose)
        probs_sagittal = truenet_tumseg_data_postprocessing.resize_to_original_size(probs_sagittal, test_sub_dict,
                                                                                    plane='sagittal')
        probs_combined.append(probs_sagittal)
        
        if intermediate:
            save_path = os.path.join(output_dir, 'Predicted_probmap_truenet_' + basename + '_sagittal.nii.gz')
            preds_sagittal = truenet_tumseg_data_postprocessing.get_final_3dvolumes(np.argmax(probs_sagittal, axis=3), test_sub_dict)
            if verbose:
                print('Saving the intermediate Sagittal prediction ...', flush=True)
            
            newhdr = flair_hdr.copy()
            newobj = nib.nifti1.Nifti1Image(preds_sagittal, None, header=newhdr)
            nib.save(newobj, save_path) 
        
        probs_coronal = truenet_tumseg_evaluate.evaluate_truenet(test_sub_dict, model_coronal, eval_params,
                                                                 device, mode='coronal', verbose=verbose)
        probs_coronal = truenet_tumseg_data_postprocessing.resize_to_original_size(probs_coronal, test_sub_dict,
                                                                                   plane='coronal')
        probs_combined.append(probs_coronal)
        
        if intermediate:
            save_path = os.path.join(output_dir, 'Predicted_probmap_truenet_' + basename + '_coronal.nii.gz')
            preds_coronal = truenet_tumseg_data_postprocessing.get_final_3dvolumes(np.argmax(probs_coronal, axis=3), test_sub_dict)
            if verbose:
                print('Saving the intermediate Coronal prediction ...', flush=True)
            
            newhdr = flair_hdr.copy()
            newobj = nib.nifti1.Nifti1Image(preds_coronal, None, header=newhdr)
            nib.save(newobj, save_path) 
            
        probs_combined = np.array(probs_combined)
        prob_mean = np.mean(probs_combined, axis=0)
        # pred_mean = truenet_tumseg_data_postprocessing.get_final_3dvolumes(prob_mean, test_sub_dict)

        pred_output = np.argmax(prob_mean, axis=3)
        pred_output = truenet_tumseg_data_postprocessing.get_final_3dvolumes(pred_output, test_sub_dict)

        if postproc is False:
            # save_path = os.path.join(output_dir, 'Predicted_probmap_truenet_' + basename + '_final.nii.gz')
            # if verbose:
            #     print('Saving the final prediction ...', flush=True)
            #
            # newhdr = flair_hdr.copy()
            # newobj = nib.nifti1.Nifti1Image(pred_mean, None, header=newhdr)
            # nib.save(newobj, save_path)

            save_path = os.path.join(output_dir, 'Predicted_output_truenet_' + basename + '_final.nii.gz')
            if verbose:
                print('Saving the final output segmented map ...', flush=True)

            newhdr = flair_hdr.copy()
            newobj = nib.nifti1.Nifti1Image(pred_output, None, header=newhdr)
            nib.save(newobj, save_path)
        else:
            probs_tc = truenet_tumseg_evaluate.evaluate_truenet(test_sub_dict, model_tc, eval_params,
                                                                device, mode='tc', verbose=verbose)
            probs_tc = truenet_tumseg_data_postprocessing.resize_to_original_size(probs_tc, test_sub_dict,
                                                                                  plane='tc')
            preds_tc = truenet_tumseg_data_postprocessing.get_final_3dvolumes(np.argmax(probs_tc, axis=3), test_sub_dict)
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
        print('Testing complete for all subjects!', flush=True)
