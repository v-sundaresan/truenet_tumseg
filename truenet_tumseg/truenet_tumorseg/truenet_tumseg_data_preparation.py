from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from truenet_tumseg.truenet_tumorseg import (truenet_tumseg_augmentation, truenet_tumseg_data_preprocessing)
# from skimage.transform import resize
import nibabel as nib

#=========================================================================================
# Truenet data preparation function
# Vaanathi Sundaresan
# 10-03-2021, Oxford
#=========================================================================================


def create_data_array(names, plane='axial'):
    """
    Create the input stack of 2D slices reshaped to required dimensions
    :param names: list of dictionaries containing filepaths
    :param plane: acquisition plane
    :return: dictionary of input arrays
    """
    labels = np.array([])
    data_final = []
    for i in range(len(names)): 
        array_loaded = load_and_crop_data(names[i])
        data_list = array_loaded['data_cropped']
        labels_sub1 = array_loaded['label_cropped']

        if plane == 'axial' or plane == 'tc':
            for mo in range(len(data_list)):
                data_sub1 = data_list[mo]
                data_sub1 = data_sub1.transpose(2, 0, 1)
                try:
                    data = data_final[mo]
                except:
                    data = np.array([])
                data = np.concatenate((data, data_sub1), axis=0) if data.size else data_sub1
                if i == 0:
                    data_final.append(data)
                else:
                    data_final[mo] = data

            labels_sub1 = labels_sub1.transpose(2, 0, 1)
            labels = np.concatenate((labels, labels_sub1), axis=0) if labels.size else labels_sub1
        elif plane == 'sagittal':
            for mo in range(len(data_list)):
                data_sub1 = data_list[mo]
                try:
                    data = data_final[mo]
                except:
                    data = np.array([])
                data = np.concatenate((data, data_sub1), axis=0) if data.size else data_sub1
                if i == 0:
                    data_final.append(data)
                else:
                    data_final[mo] = data

            labels = np.concatenate((labels, labels_sub1), axis=0) if labels.size else labels_sub1
        elif plane == 'coronal':
            for mo in range(len(data_list)):
                data_sub1 = data_list[mo]
                data_sub1 = data_sub1.transpose(1, 0, 2)
                try:
                    data = data_final[mo]
                except:
                    data = np.array([])
                data = np.concatenate((data, data_sub1), axis=0) if data.size else data_sub1
                if i == 0:
                    data_final.append(data)
                else:
                    data_final[mo] = data

            labels_sub1 = labels_sub1.transpose(1, 0, 2)
            labels = np.concatenate((labels, labels_sub1), axis=0) if labels.size else labels_sub1

    input_data = {'flair': data_final, 'label': labels}
    return input_data


def load_and_crop_data(data_path):
    """
    Loads and crops the input data and distance maps (if required)
    :param data_path: dictionary of filepaths
    :param weighted: bool, whether to apply spatial weights in loss function
    :return: dictionary containing cropped arrays
    """
    modal_paths = []
    if data_path['flair_path'] is not None:
        modal_paths.append(data_path['flair_path'])

    if data_path['t1_path'] is not None:
        modal_paths.append(data_path['t1_path'])

    if data_path['t1ce_path'] is not None:
        modal_paths.append(data_path['t1ce_path'])

    if data_path['t2_path'] is not None:
        modal_paths.append(data_path['t2_path'])

    if data_path['other_path'] is not None:
        modal_paths.append(data_path['other_path'])

    lab_path = data_path['gt_path']

    data_sub_org = nib.load(modal_paths[0]).get_data().astype(float)
    _, coords = truenet_tumseg_data_preprocessing.tight_crop_data(data_sub_org)

    row_cent = coords[1] // 2 + coords[0]
    col_cent = coords[3] // 2 + coords[2]
    stack_cent = coords[5] // 2 + coords[4]
    rowstart = np.amax([row_cent - 96, 0])
    rowend = np.amin([row_cent + 96, data_sub_org.shape[0]])
    colstart = np.amax([col_cent - 96, 0])
    colend = np.amin([col_cent + 96, data_sub_org.shape[1]])
    stackstart = np.amax([stack_cent - 80, 0])
    stackend = np.amin([stack_cent + 80, data_sub_org.shape[2]])

    cropped_data = []
    for mo in range(len(modal_paths)):
        data_sub_org = nib.load(modal_paths[mo]).get_data().astype(float)
        data_sub1 = np.zeros([192, 192, 160])
        data_sub_piece = truenet_tumseg_data_preprocessing.preprocess_data_gauss(
            data_sub_org[rowstart:rowend, colstart:colend, stackstart:stackend])
        data_sub1[:data_sub_piece.shape[0], :data_sub_piece.shape[1], :data_sub_piece.shape[2]] = data_sub_piece
        cropped_data.append(data_sub1)

    labels_sub = nib.load(lab_path).get_data().astype(float) 
    labels_sub1 = np.zeros([192, 192, 160])
    labels_sub_piece = labels_sub[rowstart:rowend, colstart:colend, stackstart:stackend]
    labels_sub1[:labels_sub_piece.shape[0], :labels_sub_piece.shape[1], :labels_sub_piece.shape[2]] = labels_sub_piece

    loaded_array = {'data_cropped': cropped_data,
                    'label_cropped': labels_sub1
                    }
    
    return loaded_array


def get_slices_from_data_with_aug(loaded_data_array, af=2, plane='axial', test=0):
    '''
    getting the final stack of slices after data augmentation (if chosen) to form datasets.
    :param loaded_data_array: dictionary of reshaped input arrays
    :param af: int, augmentation factor
    :param plane: str, acquisition plane
    :param test: binary, if test == 1, no data sugmentation will be applied
    :return:
    '''
    data = loaded_data_array['flair']
    labels = loaded_data_array['label']
    labels = (labels == 1).astype(float)

    if plane == 'sagittal':
        aug_factor = af
    elif plane == 'coronal':
        aug_factor = af
    elif plane == 'axial' or plane == 'tc':
        aug_factor = af+1

    if len(data == 1):
        if test == 0:
            data1, labels = perform_augmentation1(data[0], labels, af=aug_factor)
            data[0] = data1
    elif len(data == 2):
        if test == 0:
            data1, data2, labels = perform_augmentation2(data[0], data[1], labels, af=aug_factor)
            data = [data1, data2]
    elif len(data == 3):
        if test == 0:
            data1, data2, data3, labels = \
                perform_augmentation3(data[0], data[1], data[2], labels, af=aug_factor)
            data = [data1, data2, data3]
    elif len(data == 4):
        if test == 0:
            data1, data2, data3, data4, labels = \
                perform_augmentation4(data[0], data[1], data[2], data[3], labels, af=aug_factor)
            data = [data1, data2, data3, data4]
    elif len(data == 5):
        if test == 0:
            data1, data2, data3, data4, data5, labels = \
                perform_augmentation5(data[0], data[1], data[2], data[3], data[4], labels, af=aug_factor)
            data = [data1, data2, data3, data4, data5]

    data_aug = np.array([])
    for mo in range(len(data)):
        data_aug = np.concatenate((data_aug, data[mo]), axis=-1) if data_aug.size else data[mo]
    data2d = [data_aug, labels]
    return data2d


def perform_augmentation1(otr, otr_labs, af=2):
    '''
    Perform augmentation on input images (without distance maps)
    :param otr: mod1 4D [N, H, W, 1]
    :param otr_labs: manual mask 3D [N, H, W]
    :param af: int, augmentation factor
    :return: augmented images (same dims as above) N = N + (N * af)
    '''
    augmented_img_list = []
    augmented_mseg_list = []
    for i in range(0,af):
        for id in range(otr.shape[0]):
            image = otr[id, :, :]
            manmask = otr_labs[id, :, :]
            augmented_img, augmented_manseg = truenet_tumseg_augmentation.augment1(image, manmask)
            augmented_img_list.append(augmented_img)
            augmented_mseg_list.append(augmented_manseg)
    augmented_img = np.array(augmented_img_list)
    augmented_mseg = np.array(augmented_mseg_list)
    augmented_img = np.reshape(augmented_img, [-1, otr.shape[1], otr.shape[2]])
    augmented_mseg = np.reshape(augmented_mseg, [-1, otr.shape[1], otr.shape[2]])
    augmented_img = np.tile(augmented_img, (1, 1, 1, 1))
    augmented_imgs = augmented_img.transpose(1, 2, 3, 0)
    otr_aug = np.concatenate((otr, augmented_imgs), axis=0)
    otr_labs = np.concatenate((otr_labs, augmented_mseg), axis=0)
    return otr_aug, otr_labs


def perform_augmentation2(otr, otr1, otr_labs, af=2):
    '''
    Perform augmentation on input images (without distance maps)
    :param otr: mod1 4D [N, H, W, 1]
    :param otr1: mod2 4D [N, H, W, 1]
    :param otr_labs: manual mask 3D [N, H, W]
    :param af: int, augmentation factor
    :return: augmented images (same dims as above) N = N + (N * af)
    '''
    augmented_img_list = []
    augmented_img1_list = []
    augmented_mseg_list = []
    for i in range(0, af):
        for id in range(otr.shape[0]):
            image = otr[id, :, :]
            image1 = otr1[id, :, :]
            manmask = otr_labs[id, :, :]
            augmented_img, augmented_img1, augmented_manseg = truenet_tumseg_augmentation.augment2(image, image1, manmask)
            augmented_img_list.append(augmented_img)
            augmented_img1_list.append(augmented_img1)
            augmented_mseg_list.append(augmented_manseg)
    augmented_img = np.array(augmented_img_list)
    augmented_img1 = np.array(augmented_img1_list)
    augmented_mseg = np.array(augmented_mseg_list)
    augmented_img = np.reshape(augmented_img, [-1, otr.shape[1], otr.shape[2]])
    augmented_img1 = np.reshape(augmented_img1, [-1, otr.shape[1], otr.shape[2]])
    augmented_mseg = np.reshape(augmented_mseg, [-1, otr.shape[1], otr.shape[2]])
    augmented_img = np.tile(augmented_img, (1, 1, 1, 1))
    augmented_imgs = augmented_img.transpose(1, 2, 3, 0)
    augmented_img1 = np.tile(augmented_img1, (1, 1, 1, 1))
    augmented_imgs1 = augmented_img1.transpose(1, 2, 3, 0)
    otr_aug = np.concatenate((otr, augmented_imgs), axis=0)
    otr_aug1 = np.concatenate((otr1, augmented_imgs1), axis=0)
    otr_labs = np.concatenate((otr_labs, augmented_mseg), axis=0)
    return otr_aug, otr_aug1, otr_labs


def perform_augmentation3(otr, otr1, otr2, otr_labs, af=2):
    '''
    Perform augmentation on input images (without distance maps)
    :param otr: mod1 4D [N, H, W, 1]
    :param otr1: mod2 4D [N, H, W, 1]
    :param otr2: mod3 4D [N, H, W, 1]
    :param otr_labs: manual mask 3D [N, H, W]
    :param af: int, augmentation factor
    :return: augmented images (same dims as above) N = N + (N * af)
    '''
    augmented_img_list = []
    augmented_img1_list = []
    augmented_img2_list = []
    augmented_mseg_list = []
    for i in range(0, af):
        for id in range(otr.shape[0]):
            image = otr[id, :, :]
            image1 = otr1[id, :, :]
            image2 = otr2[id, :, :]
            manmask = otr_labs[id, :, :]
            augmented_img, augmented_img1, augmented_img2, augmented_manseg = \
                truenet_tumseg_augmentation.augment3(image, image1, image2, manmask)
            augmented_img_list.append(augmented_img)
            augmented_img1_list.append(augmented_img1)
            augmented_img2_list.append(augmented_img2)
            augmented_mseg_list.append(augmented_manseg)
    augmented_img = np.array(augmented_img_list)
    augmented_img1 = np.array(augmented_img1_list)
    augmented_img2 = np.array(augmented_img2_list)
    augmented_mseg = np.array(augmented_mseg_list)
    augmented_img = np.reshape(augmented_img, [-1, otr.shape[1], otr.shape[2]])
    augmented_img1 = np.reshape(augmented_img1, [-1, otr.shape[1], otr.shape[2]])
    augmented_img2 = np.reshape(augmented_img2, [-1, otr.shape[1], otr.shape[2]])
    augmented_mseg = np.reshape(augmented_mseg, [-1, otr.shape[1], otr.shape[2]])
    augmented_img = np.tile(augmented_img, (1, 1, 1, 1))
    augmented_imgs = augmented_img.transpose(1, 2, 3, 0)
    augmented_img1 = np.tile(augmented_img1, (1, 1, 1, 1))
    augmented_imgs1 = augmented_img1.transpose(1, 2, 3, 0)
    augmented_img2 = np.tile(augmented_img2, (1, 1, 1, 1))
    augmented_imgs2 = augmented_img2.transpose(1, 2, 3, 0)
    otr_aug = np.concatenate((otr, augmented_imgs), axis=0)
    otr_aug1 = np.concatenate((otr1, augmented_imgs1), axis=0)
    otr_aug2 = np.concatenate((otr2, augmented_imgs2), axis=0)
    otr_labs = np.concatenate((otr_labs, augmented_mseg), axis=0)
    return otr_aug, otr_aug1, otr_aug2, otr_labs


def perform_augmentation4(otr, otr1, otr2, otr3, otr_labs, af=2):
    '''
    Perform augmentation on input images (without distance maps)
    :param otr: mod1 4D [N, H, W, 1]
    :param otr1: mod2 4D [N, H, W, 1]
    :param otr2: mod3 4D [N, H, W, 1]
    :param otr3: mod4 4D [N, H, W, 1]
    :param otr_labs: manual mask 3D [N, H, W]
    :param af: int, augmentation factor
    :return: augmented images (same dims as above) N = N + (N * af)
    '''
    augmented_img_list = []
    augmented_img1_list = []
    augmented_img2_list = []
    augmented_img3_list = []
    augmented_mseg_list = []
    for i in range(0, af):
        for id in range(otr.shape[0]):
            image = otr[id, :, :]
            image1 = otr1[id, :, :]
            image2 = otr2[id, :, :]
            image3 = otr3[id, :, :]
            manmask = otr_labs[id, :, :]
            augmented_img, augmented_img1, augmented_img2, augmented_img3, augmented_manseg = \
                truenet_tumseg_augmentation.augment4(image, image1, image2, image3, manmask)
            augmented_img_list.append(augmented_img)
            augmented_img1_list.append(augmented_img1)
            augmented_img2_list.append(augmented_img2)
            augmented_img3_list.append(augmented_img3)
            augmented_mseg_list.append(augmented_manseg)
    augmented_img = np.array(augmented_img_list)
    augmented_img1 = np.array(augmented_img1_list)
    augmented_img2 = np.array(augmented_img2_list)
    augmented_img3 = np.array(augmented_img3_list)
    augmented_mseg = np.array(augmented_mseg_list)
    augmented_img = np.reshape(augmented_img, [-1, otr.shape[1], otr.shape[2]])
    augmented_img1 = np.reshape(augmented_img1, [-1, otr.shape[1], otr.shape[2]])
    augmented_img2 = np.reshape(augmented_img2, [-1, otr.shape[1], otr.shape[2]])
    augmented_img3 = np.reshape(augmented_img3, [-1, otr.shape[1], otr.shape[2]])
    augmented_mseg = np.reshape(augmented_mseg, [-1, otr.shape[1], otr.shape[2]])
    augmented_img = np.tile(augmented_img, (1, 1, 1, 1))
    augmented_imgs = augmented_img.transpose(1, 2, 3, 0)
    augmented_img1 = np.tile(augmented_img1, (1, 1, 1, 1))
    augmented_imgs1 = augmented_img1.transpose(1, 2, 3, 0)
    augmented_img2 = np.tile(augmented_img2, (1, 1, 1, 1))
    augmented_imgs2 = augmented_img2.transpose(1, 2, 3, 0)
    augmented_img3 = np.tile(augmented_img3, (1, 1, 1, 1))
    augmented_imgs3 = augmented_img3.transpose(1, 2, 3, 0)
    otr_aug = np.concatenate((otr, augmented_imgs), axis=0)
    otr_aug1 = np.concatenate((otr1, augmented_imgs1), axis=0)
    otr_aug2 = np.concatenate((otr2, augmented_imgs2), axis=0)
    otr_aug3 = np.concatenate((otr3, augmented_imgs3), axis=0)
    otr_labs = np.concatenate((otr_labs, augmented_mseg), axis=0)
    return otr_aug, otr_aug1, otr_aug2, otr_aug3, otr_labs


def perform_augmentation5(otr, otr1, otr2, otr3, otr4, otr_labs, af=2):
    '''
    Perform augmentation on input images (without distance maps)
    :param otr: mod1 4D [N, H, W, 1]
    :param otr1: mod2 4D [N, H, W, 1]
    :param otr2: mod3 4D [N, H, W, 1]
    :param otr3: mod4 4D [N, H, W, 1]
    :param otr4: mod4 4D [N, H, W, 1]
    :param otr_labs: manual mask 3D [N, H, W]
    :param af: int, augmentation factor
    :return: augmented images (same dims as above) N = N + (N * af)
    '''
    augmented_img_list = []
    augmented_img1_list = []
    augmented_img2_list = []
    augmented_img3_list = []
    augmented_img4_list = []
    augmented_mseg_list = []
    for i in range(0, af):
        for id in range(otr.shape[0]):
            image = otr[id, :, :]
            image1 = otr1[id, :, :]
            image2 = otr2[id, :, :]
            image3 = otr3[id, :, :]
            image4 = otr4[id, :, :]
            manmask = otr_labs[id, :, :]
            augmented_img, augmented_img1, augmented_img2, augmented_img3, augmented_img4, augmented_manseg = \
                truenet_tumseg_augmentation.augment5(image, image1, image2, image3, image4, manmask)
            augmented_img_list.append(augmented_img)
            augmented_img1_list.append(augmented_img1)
            augmented_img2_list.append(augmented_img2)
            augmented_img3_list.append(augmented_img3)
            augmented_img4_list.append(augmented_img4)
            augmented_mseg_list.append(augmented_manseg)
    augmented_img = np.array(augmented_img_list)
    augmented_img1 = np.array(augmented_img1_list)
    augmented_img2 = np.array(augmented_img2_list)
    augmented_img3 = np.array(augmented_img3_list)
    augmented_img4 = np.array(augmented_img4_list)
    augmented_mseg = np.array(augmented_mseg_list)
    augmented_img = np.reshape(augmented_img, [-1, otr.shape[1], otr.shape[2]])
    augmented_img1 = np.reshape(augmented_img1, [-1, otr.shape[1], otr.shape[2]])
    augmented_img2 = np.reshape(augmented_img2, [-1, otr.shape[1], otr.shape[2]])
    augmented_img3 = np.reshape(augmented_img3, [-1, otr.shape[1], otr.shape[2]])
    augmented_img4 = np.reshape(augmented_img4, [-1, otr.shape[1], otr.shape[2]])
    augmented_mseg = np.reshape(augmented_mseg, [-1, otr.shape[1], otr.shape[2]])
    augmented_img = np.tile(augmented_img, (1, 1, 1, 1))
    augmented_imgs = augmented_img.transpose(1, 2, 3, 0)
    augmented_img1 = np.tile(augmented_img1, (1, 1, 1, 1))
    augmented_imgs1 = augmented_img1.transpose(1, 2, 3, 0)
    augmented_img2 = np.tile(augmented_img2, (1, 1, 1, 1))
    augmented_imgs2 = augmented_img2.transpose(1, 2, 3, 0)
    augmented_img3 = np.tile(augmented_img3, (1, 1, 1, 1))
    augmented_imgs3 = augmented_img3.transpose(1, 2, 3, 0)
    augmented_img4 = np.tile(augmented_img4, (1, 1, 1, 1))
    augmented_imgs4 = augmented_img4.transpose(1, 2, 3, 0)
    otr_aug = np.concatenate((otr, augmented_imgs), axis=0)
    otr_aug1 = np.concatenate((otr1, augmented_imgs1), axis=0)
    otr_aug2 = np.concatenate((otr2, augmented_imgs2), axis=0)
    otr_aug3 = np.concatenate((otr3, augmented_imgs3), axis=0)
    otr_aug4 = np.concatenate((otr4, augmented_imgs4), axis=0)
    otr_labs = np.concatenate((otr_labs, augmented_mseg), axis=0)
    return otr_aug, otr_aug1, otr_aug2, otr_aug3, otr_aug4, otr_labs


def create_test_data_array(names, plane='axial'):
    """
    Create the input stack of 2D slices reshaped to required dimensions
    :param names: list of dictionaries containing filepaths
    :param plane: acquisition plane
    :return: dictionary of input arrays
    """
    data_final = []
    for i in range(len(names)):  
        array_loaded = load_and_crop_test_data(names[i])
        data_list = array_loaded['data_cropped']

        if plane == 'axial' or plane == 'tc':
            for mo in range(len(data_list)):
                data_sub1 = data_list[mo]
                data_sub1 = data_sub1.transpose(2, 0, 1)
                try:
                    data = data_final[mo]
                except:
                    data = np.array([])
                data = np.concatenate((data, data_sub1), axis=0) if data.size else data_sub1
                if i == 0:
                    data_final.append(data)
                else:
                    data_final[mo] = data
        elif plane == 'sagittal':
            for mo in range(len(data_list)):
                data_sub1 = data_list[mo]
                try:
                    data = data_final[mo]
                except:
                    data = np.array([])
                data = np.concatenate((data, data_sub1), axis=0) if data.size else data_sub1
                if i == 0:
                    data_final.append(data)
                else:
                    data_final[mo] = data
        elif plane == 'coronal':
            for mo in range(len(data_list)):
                data_sub1 = data_list[mo]
                data_sub1 = data_sub1.transpose(1, 0, 2)
                try:
                    data = data_final[mo]
                except:
                    data = np.array([])
                data = np.concatenate((data, data_sub1), axis=0) if data.size else data_sub1
                if i == 0:
                    data_final.append(data)
                else:
                    data_final[mo] = data

    data = np.array([])
    for mo in range(len(data_final)):
        tmp = np.tile(data_final[mo], (1, 1, 1, 1))
        tmp = tmp.transpose(1, 2, 3, 0)
        data = np.concatenate((data, tmp), axis=-1) if data.size else tmp
    data2d = [data]
    return data2d


def load_and_crop_test_data(data_path):
    """
    Loads and crops the input data
    :param data_path: dictionary of filepaths
    :return: dictionary containing cropped arrays
    """
    modal_paths = []
    if data_path['flair_path'] is not None:
        modal_paths.append(data_path['flair_path'])

    if data_path['t1_path'] is not None:
        modal_paths.append(data_path['t1_path'])

    if data_path['t1ce_path'] is not None:
        modal_paths.append(data_path['t1ce_path'])

    if data_path['t2_path'] is not None:
        modal_paths.append(data_path['t2_path'])

    if data_path['other_path'] is not None:
        modal_paths.append(data_path['other_path'])

    data_sub_org = nib.load(modal_paths[0]).get_data().astype(float)
    _, coords = truenet_tumseg_data_preprocessing.tight_crop_data(data_sub_org)
    row_cent = coords[1] // 2 + coords[0]
    col_cent = coords[3] // 2 + coords[2]
    stack_cent = coords[5] // 2 + coords[4]
    rowstart = np.amax([row_cent - 96, 0])
    rowend = np.amin([row_cent + 96, data_sub_org.shape[0]])
    colstart = np.amax([col_cent - 96, 0])
    colend = np.amin([col_cent + 96, data_sub_org.shape[1]])
    stackstart = np.amax([stack_cent - 80, 0])
    stackend = np.amin([stack_cent + 80, data_sub_org.shape[2]])
    cropped_data = []
    for mo in range(len(modal_paths)):
        data_sub_org = nib.load(modal_paths[mo]).get_data().astype(float)
        data_sub1 = np.zeros([192, 192, 160])
        data_sub_piece = truenet_tumseg_data_preprocessing.preprocess_data_gauss(
            data_sub_org[rowstart:rowend, colstart:colend, stackstart:stackend])
        data_sub1[:data_sub_piece.shape[0], :data_sub_piece.shape[1], :data_sub_piece.shape[2]] = data_sub_piece
        cropped_data.append(data_sub1)

    loaded_array = {'data_cropped': cropped_data}
    
    return loaded_array


