from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate
from skimage.util import random_noise

#=========================================================================================
# Truenet augmentations function
# Vaanathi Sundaresan
# 11-03-2021, Oxford
#=========================================================================================


##########################################################################################
# Define transformations with distance maps
##########################################################################################

def translate1(image, label):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    offsetx = random.randint(-15, 15)
    offsety = random.randint(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    translated_im = shift(image, (offsetx, offsety), order=order, mode='nearest')
    translated_label = shift(label, (offsetx, offsety), order=order, mode='nearest')
    return translated_im, translated_label


def translate2(image, image1, label):
    """
    :param image: mod1
    :param image1: mod2
    :param label: manual mask
    :return:
    """
    offsetx = random.randint(-15, 15)
    offsety = random.randint(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    translated_im = shift(image, (offsetx, offsety), order=order, mode='nearest')
    translated_im1 = shift(image1, (offsetx, offsety), order=order, mode='nearest')
    translated_label = shift(label, (offsetx, offsety), order=order, mode='nearest')
    return translated_im, translated_im1, translated_label


def translate3(image, image1, image2, label):
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param label: manual mask
    :return:
    """
    offsetx = random.randint(-15, 15)
    offsety = random.randint(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    translated_im = shift(image, (offsetx, offsety), order=order, mode='nearest')
    translated_im1 = shift(image1, (offsetx, offsety), order=order, mode='nearest')
    translated_im2 = shift(image2, (offsetx, offsety), order=order, mode='nearest')
    translated_label = shift(label, (offsetx, offsety), order=order, mode='nearest')
    return translated_im, translated_im1, translated_im2, translated_label


def translate4(image, image1, image2, image3, label):
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param image3: mod4
    :param label: manual mask
    :return:
    """
    offsetx = random.randint(-15, 15)
    offsety = random.randint(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    translated_im = shift(image, (offsetx, offsety), order=order, mode='nearest')
    translated_im1 = shift(image1, (offsetx, offsety), order=order, mode='nearest')
    translated_im2 = shift(image2, (offsetx, offsety), order=order, mode='nearest')
    translated_im3 = shift(image3, (offsetx, offsety), order=order, mode='nearest')
    translated_label = shift(label, (offsetx, offsety), order=order, mode='nearest')
    return translated_im, translated_im1, translated_im2, translated_im3, translated_label


def translate5(image, image1, image2, image3, image4, label):
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param image3: mod4
    :param image4: mod5
    :param label: manual mask
    :return:
    """
    offsetx = random.randint(-15, 15)
    offsety = random.randint(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    translated_im = shift(image, (offsetx, offsety), order=order, mode='nearest')
    translated_im1 = shift(image1, (offsetx, offsety), order=order, mode='nearest')
    translated_im2 = shift(image2, (offsetx, offsety), order=order, mode='nearest')
    translated_im3 = shift(image3, (offsetx, offsety), order=order, mode='nearest')
    translated_im4 = shift(image4, (offsetx, offsety), order=order, mode='nearest')
    translated_label = shift(label, (offsetx, offsety), order=order, mode='nearest')
    return translated_im, translated_im1, translated_im2, translated_im3, translated_im4, translated_label


def rotate1(image, label):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    theta = random.uniform(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    new_img = rotate(image, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = rotate(label, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = (new_lab > 0.5).astype(float)
    return new_img, new_lab


def rotate2(image, image1, label):
    """
    :param image: mod1
    :param image1: mod2
    :param label: manual mask
    :return:
    """
    theta = random.uniform(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    new_img = rotate(image, float(theta), reshape=False, order=order, mode='nearest')
    new_img1 = rotate(image1, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = rotate(label, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = (new_lab > 0.5).astype(float)
    return new_img, new_img1, new_lab


def rotate3(image, image1, image2, label):
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param label: manual mask
    :return:
    """
    theta = random.uniform(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    new_img = rotate(image, float(theta), reshape=False, order=order, mode='nearest')
    new_img1 = rotate(image1, float(theta), reshape=False, order=order, mode='nearest')
    new_img2 = rotate(image2, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = rotate(label, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = (new_lab > 0.5).astype(float)
    return new_img, new_img1, new_img2, new_lab


def rotate4(image, image1, image2, image3, label):
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param image3: mod4
    :param label: manual mask
    :return:
    """
    theta = random.uniform(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    new_img = rotate(image, float(theta), reshape=False, order=order, mode='nearest')
    new_img1 = rotate(image1, float(theta), reshape=False, order=order, mode='nearest')
    new_img2 = rotate(image2, float(theta), reshape=False, order=order, mode='nearest')
    new_img3 = rotate(image3, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = rotate(label, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = (new_lab > 0.5).astype(float)
    return new_img, new_img1, new_img2, new_img3, new_lab


def rotate5(image, image1, image2, image3, image4, label):
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param image3: mod4
    :param image4: mod5
    :param label: manual mask
    :return:
    """
    theta = random.uniform(-15, 15)
    is_seg = False
    order = 0 if is_seg is True else 5
    new_img = rotate(image, float(theta), reshape=False, order=order, mode='nearest')
    new_img1 = rotate(image1, float(theta), reshape=False, order=order, mode='nearest')
    new_img2 = rotate(image2, float(theta), reshape=False, order=order, mode='nearest')
    new_img3 = rotate(image3, float(theta), reshape=False, order=order, mode='nearest')
    new_img4 = rotate(image4, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = rotate(label, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = (new_lab > 0.5).astype(float)
    return new_img, new_img1, new_img2, new_img3, new_img4, new_lab


def blur1(image, label):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    sigma = random.uniform(0.1, 0.2)
    new_img = gaussian_filter(image, sigma)
    return new_img, label


def blur2(image, image1, label):
    """
    :param image: mod1
    :param image1: mod2
    :param label: manual mask
    :return:
    """
    sigma = random.uniform(0.1, 0.2)
    new_img = gaussian_filter(image, sigma)
    new_img1 = gaussian_filter(image1, sigma)
    return new_img, new_img1, label


def blur3(image, image1, image2, label):
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param label: manual mask
    :return:
    """
    sigma = random.uniform(0.1, 0.2)
    new_img = gaussian_filter(image, sigma)
    new_img1 = gaussian_filter(image1, sigma)
    new_img2 = gaussian_filter(image2, sigma)
    return new_img, new_img1, new_img2, label


def blur4(image, image1, image2, image3, label):
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param image3: mod4
    :param label: manual mask
    :return:
    """
    sigma = random.uniform(0.1, 0.2)
    new_img = gaussian_filter(image, sigma)
    new_img1 = gaussian_filter(image1, sigma)
    new_img2 = gaussian_filter(image2, sigma)
    new_img3 = gaussian_filter(image3, sigma)
    return new_img, new_img1, new_img2, new_img3, label


def blur5(image, image1, image2, image3, image4, label):
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param image3: mod4
    :param image4: mod5
    :param label: manual mask
    :return:
    """
    sigma = random.uniform(0.1, 0.2)
    new_img = gaussian_filter(image, sigma)
    new_img1 = gaussian_filter(image1, sigma)
    new_img2 = gaussian_filter(image2, sigma)
    new_img3 = gaussian_filter(image3, sigma)
    new_img4 = gaussian_filter(image4, sigma)
    return new_img, new_img1, new_img2, new_img3, new_img4, label


def add_noise1(image, label):
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    new_img = random_noise(image, clip=False)
    return new_img, label


def add_noise2(image, image1, label):
    """
    :param image: mod1
    :param image1: mod2
    :param label: manual mask
    :return:
    """
    new_img = random_noise(image, clip=False)
    new_img1 = random_noise(image1, clip=False)
    return new_img, new_img1, label


def add_noise3(image, image1, image2, label):
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param label: manual mask
    :return:
    """
    new_img = random_noise(image, clip=False)
    new_img1 = random_noise(image1, clip=False)
    new_img2 = random_noise(image2, clip=False)
    return new_img, new_img1, new_img2, label


def add_noise4(image, image1, image2, image3, label):
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param image3: mod4
    :param label: manual mask
    :return:
    """
    new_img = random_noise(image, clip=False)
    new_img1 = random_noise(image1, clip=False)
    new_img2 = random_noise(image2, clip=False)
    new_img3 = random_noise(image3, clip=False)
    return new_img, new_img1, new_img2, new_img3, label


def add_noise5(image, image1, image2, image3, image4, label):
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param image3: mod4
    :param image4: mod5
    :param label: manual mask
    :return:
    """
    new_img = random_noise(image, clip=False)
    new_img1 = random_noise(image1, clip=False)
    new_img2 = random_noise(image2, clip=False)
    new_img3 = random_noise(image3, clip=False)
    new_img4 = random_noise(image4, clip=False)
    return new_img, new_img1, new_img2, new_img3, new_img4, label

##########################################################################################
# Define transformations with 1 modality
##########################################################################################


def augment1(image, label):
    # Applies a random number of the possible transformations to the inputs.
    """
    :param image: mod1
    :param label: manual mask
    :return:
    """
    if len(image.shape) == 2:
        # Add to the available transformations any functions you want to be applied
        available_transformations = {'noise': add_noise1, 'translate': translate1,
                                     'rotate': rotate1, 'blur': blur1}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformed_image = None
        transformed_label = None

        while num_transformations <= num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image, transformed_label = available_transformations[key](image, label)
            num_transformations += 1
        return transformed_image, transformed_label
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 2d')

##########################################################################################
# Define transformations with 2 modalities
##########################################################################################


def augment2(image, image1, label):
    # Applies a random number of the possible transformations to the inputs.
    """
    :param image: mod1
    :param image1: mod2
    :param label: manual mask
    :return:
    """
    if len(image.shape) == 2:
        # Add to the available transformations any functions you want to be applied
        available_transformations = {'noise': add_noise2, 'translate': translate2,
                                     'rotate': rotate2, 'blur': blur2}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformed_image = None
        transformed_image1 = None
        transformed_label = None

        while num_transformations <= num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image, transformed_image1, transformed_label = available_transformations[key](image, image1,
                                                                                                      label)
            num_transformations += 1
        return transformed_image, transformed_image1, transformed_label
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 2d')

##########################################################################################
# Define transformations with 3 modalities
##########################################################################################


def augment3(image, image1, image2, label):
    # Applies a random number of the possible transformations to the inputs.
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param label: manual mask
    :return:
    """
    if len(image.shape) == 2:
        # Add to the available transformations any functions you want to be applied
        available_transformations = {'noise': add_noise3, 'translate': translate3,
                                     'rotate': rotate3, 'blur': blur3}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformed_image = None
        transformed_image1 = None
        transformed_image2 = None
        transformed_label = None

        while num_transformations <= num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image, transformed_image1, transformed_image2, transformed_label = \
                available_transformations[key](image, image1, image2, label)
            num_transformations += 1
        return transformed_image, transformed_image1, transformed_image2, transformed_label
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 2d')


##########################################################################################
# Define transformations with 4 modalities
##########################################################################################


def augment4(image, image1, image2, image3, label):
    # Applies a random number of the possible transformations to the inputs.
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param image3: mod4
    :param label: manual mask
    :return:
    """
    if len(image.shape) == 2:
        # Add to the available transformations any functions you want to be applied
        available_transformations = {'noise': add_noise4, 'translate': translate4,
                                     'rotate': rotate4, 'blur': blur4}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformed_image = None
        transformed_image1 = None
        transformed_image2 = None
        transformed_image3 = None
        transformed_label = None

        while num_transformations <= num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image, transformed_image1, transformed_image2, transformed_image3, transformed_label = \
                available_transformations[key](image, image1, image2, image3, label)
            num_transformations += 1
        return transformed_image, transformed_image1, transformed_image2, transformed_image3, transformed_label
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 2d')

##########################################################################################
# Define transformations with 5 modalities
##########################################################################################


def augment5(image, image1, image2, image3, image4, label):
    # Applies a random number of the possible transformations to the inputs.
    """
    :param image: mod1
    :param image1: mod2
    :param image2: mod3
    :param image3: mod4
    :param image4: mod5
    :param label: manual mask
    :return:
    """
    if len(image.shape) == 2:
        # Add to the available transformations any functions you want to be applied
        available_transformations = {'noise': add_noise5, 'translate': translate5,
                                     'rotate': rotate5, 'blur': blur5}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformed_image = None
        transformed_image1 = None
        transformed_image2 = None
        transformed_image3 = None
        transformed_image4 = None
        transformed_label = None

        while num_transformations <= num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image, transformed_image1, transformed_image2, transformed_image3, transformed_image4, \
                transformed_label = available_transformations[key](image, image1, image2, image3, image4, label)
            num_transformations += 1
        return transformed_image, transformed_image1, transformed_image2, transformed_image3, transformed_image4, transformed_label
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 2d')
