from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from truenet_tumseg.truenet_tumorseg import truenet_tumseg_data_preprocessing
# from skimage.transform import resize
import nibabel as nib

################################################################################################
# Truenet data postprocessing function
# Vaanathi Sundaresan
# 10-03-2021, Oxford
################################################################################################


def resize_to_original_size(probs, testpathdicts, plane='axial'):
    """
    :param probs: predicted 4d probability maps (N x H x W x Classes)
    :param testpathdicts: list of dictionaries containing test image datapaths
    :param plane: Acquisition plane
    :return: 3D probability maps with cropped dimensions.
    """
    overall_prob = np.array([])
    testpath = testpathdicts[0]
    modal_paths = []
    if testpath['flair_path'] is not None:
        modal_paths.append(testpath['flair_path'])

    if testpath['t1_path'] is not None:
        modal_paths.append(testpath['t1_path'])

    if testpath['t1ce_path'] is not None:
        modal_paths.append(testpath['t1ce_path'])

    if testpath['t2_path'] is not None:
        modal_paths.append(testpath['t2_path'])

    if testpath['other_path'] is not None:
        modal_paths.append(testpath['other_path'])
    data = nib.load(modal_paths[0]).get_data().astype(float)
    _, coords = truenet_tumseg_data_preprocessing.tight_crop_data(data)
    reqd_dims = [192, 192, 160]
    if plane == 'axial' or plane == 'tc':
        probs_sub = probs[:reqd_dims[2], :, :, :]
        overall_prob = np.concatenate((overall_prob, probs_sub), axis=0) if overall_prob.size else probs_sub
    elif plane == 'sagittal':
        probs_sub = probs[:reqd_dims[0], :, :, :]
        prob_specific_sub = probs_sub.transpose(2, 0, 1, 3)
        overall_prob = np.concatenate((overall_prob, prob_specific_sub),
                                      axis=0) if overall_prob.size else prob_specific_sub
    elif plane == 'coronal':
        probs_sub = probs[:reqd_dims[1], :, :, :]
        prob_specific_sub = probs_sub.transpose(2, 1, 0, 3)
        overall_prob = np.concatenate((overall_prob, prob_specific_sub),
                                      axis=0) if overall_prob.size else prob_specific_sub
    return overall_prob


def get_final_3dvolumes(volume3d, testpathdicts):
    """
    :param volume3d: 3D probability maps
    :param testpathdicts: 3D probability maps in original dimensions
    :return:
    """
    volume3d = np.tile(volume3d, (1, 1, 1, 1))
    print('starting from here for debugging!')
    print(volume3d.shape)
    volume4d = volume3d.transpose(1, 2, 3, 0)
    print(volume4d.shape)
    st = 0
    testpath = testpathdicts[0]

    modal_paths = []
    if testpath['flair_path'] is not None:
        modal_paths.append(testpath['flair_path'])

    if testpath['t1_path'] is not None:
        modal_paths.append(testpath['t1_path'])

    if testpath['t1ce_path'] is not None:
        modal_paths.append(testpath['t1ce_path'])

    if testpath['t2_path'] is not None:
        modal_paths.append(testpath['t2_path'])

    if testpath['other_path'] is not None:
        modal_paths.append(testpath['other_path'])
    data = nib.load(modal_paths[0]).get_data().astype(float)

    volume3d = 0 * data
    print(volume3d.shape)
    _, coords = truenet_tumseg_data_preprocessing.tight_crop_data(data)
    row_cent = coords[1] // 2 + coords[0]
    col_cent = coords[3] // 2 + coords[2]
    stack_cent = coords[5] // 2 + coords[4]
    rowstart = np.amax([row_cent - 96, 0])
    rowend = np.amin([row_cent + 96, data.shape[0]])
    colstart = np.amax([col_cent - 96, 0])
    colend = np.amin([col_cent + 96, data.shape[1]])
    stackstart = np.amax([stack_cent - 80, 0])
    stackend = np.amin([stack_cent + 80, data.shape[2]])
    data_sub = data[rowstart:rowend, colstart:colend, stackstart:stackend]
    print(data.shape)
    print(data_sub.shape)
    required_stacks = volume4d[st:st+data_sub.shape[2], :data_sub.shape[0], :data_sub.shape[1], 0].transpose(1, 2, 0)
    print(required_stacks.shape)
    volume3d[rowstart:rowend, colstart:colend, stackstart:stackend] = required_stacks
    print(volume3d.shape)
    return volume3d


def resize_testdata_to_original_size(probs, testpathdicts, plane='axial'):
    """
    :param probs: predicted 4d probability maps (N x H x W x Classes)
    :param testpathdicts: list of dictionaries containing test image datapaths
    :param plane: Acquisition plane
    :return: 3D probability maps with cropped dimensions.
    """
    overall_prob = np.array([])
    testpath = testpathdicts[0]
    modal_paths = []
    if testpath['flair_path'] is not None:
        modal_paths.append(testpath['flair_path'])

    if testpath['t1_path'] is not None:
        modal_paths.append(testpath['t1_path'])

    if testpath['t1ce_path'] is not None:
        modal_paths.append(testpath['t1ce_path'])

    if testpath['t2_path'] is not None:
        modal_paths.append(testpath['t2_path'])

    if testpath['other_path'] is not None:
        modal_paths.append(testpath['other_path'])
    data = nib.load(modal_paths[0]).get_data().astype(float)
    _, coords = truenet_tumseg_data_preprocessing.tight_crop_data(data)
    reqd_dims = [coords[1], coords[3], coords[5]]
    if plane == 'axial' or plane == 'tc':
        probs_sub = probs[:reqd_dims[2], :, :, :]
        overall_prob = np.concatenate((overall_prob, probs_sub), axis=0) if overall_prob.size else probs_sub
    elif plane == 'sagittal':
        probs_sub = probs[:reqd_dims[0], :, :, :]
        prob_specific_sub = probs_sub.transpose(2, 0, 1, 3)
        overall_prob = np.concatenate((overall_prob, prob_specific_sub),
                                      axis=0) if overall_prob.size else prob_specific_sub
    elif plane == 'coronal':
        probs_sub = probs[:reqd_dims[1], :, :, :]
        prob_specific_sub = probs_sub.transpose(2, 1, 0, 3)
        overall_prob = np.concatenate((overall_prob, prob_specific_sub),
                                      axis=0) if overall_prob.size else prob_specific_sub
    return overall_prob


def get_final_testdata_3dvolumes(volume3d, testpathdicts):
    """
    :param volume3d: 3D probability maps
    :param testpathdicts: 3D probability maps in original dimensions
    :return:
    """
    volume3d = np.tile(volume3d, (1, 1, 1, 1))
    print('starting from here for debugging!')
    print(volume3d.shape)
    volume4d = volume3d.transpose(1, 2, 3, 0)
    print(volume4d.shape)
    st = 0
    testpath = testpathdicts[0]

    modal_paths = []
    if testpath['flair_path'] is not None:
        modal_paths.append(testpath['flair_path'])

    if testpath['t1_path'] is not None:
        modal_paths.append(testpath['t1_path'])

    if testpath['t1ce_path'] is not None:
        modal_paths.append(testpath['t1ce_path'])

    if testpath['t2_path'] is not None:
        modal_paths.append(testpath['t2_path'])

    if testpath['other_path'] is not None:
        modal_paths.append(testpath['other_path'])
    data = nib.load(modal_paths[0]).get_data().astype(float)

    volume3d = 0 * data
    print(volume3d.shape)
    _, coords = truenet_tumseg_data_preprocessing.tight_crop_data(data)
    # row_cent = coords[1] // 2 + coords[0]
    # col_cent = coords[3] // 2 + coords[2]
    # stack_cent = coords[5] // 2 + coords[4]
    rowstart = np.amax([coords[0], 0])
    rowend = np.amin([coords[0] + coords[1], data.shape[0]])
    colstart = np.amax([coords[2], 0])
    colend = np.amin([coords[2] + coords[3], data.shape[1]])
    stackstart = np.amax([coords[4], 0])
    stackend = np.amin([coords[4] + coords[5], data.shape[2]])
    data_sub = data[rowstart:rowend, colstart:colend, stackstart:stackend]
    print(data.shape)
    print(data_sub.shape)
    required_stacks = volume4d[st:st+data_sub.shape[2], :data_sub.shape[0], :data_sub.shape[1], 0].transpose(1, 2, 0)
    print(required_stacks.shape)
    volume3d[rowstart:rowend, colstart:colend, stackstart:stackend] = required_stacks
    print(volume3d.shape)
    return volume3d
