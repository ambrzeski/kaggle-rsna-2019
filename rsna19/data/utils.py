import os
from pathlib import Path

import cv2
import json
import pickle
from math import atan

import nibabel as nib
import numpy as np
import pandas as pd
import skimage
import transforms3d
from scipy import ndimage
import time
from contextlib import contextmanager


from rsna19.configs.base_config import BaseConfig

DICOM_TAGS_DF_PATH = '/kolos/m2/ct/data/rsna/df.pkl'
HU_AIR = -1000
SEG_CLASSES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "non-classified", "any"]
SEG_MASKS_HOME = "/kolos/ssd/ct-m2/"


def load_dicom_tags():
    with open(DICOM_TAGS_DF_PATH, 'rb') as f:
        df = pickle.load(f)

    return df


def load_labels():
    labels = pd.read_csv(BaseConfig.labels_path)
    labels[['SOPInstanceUID', 'Disease']] = labels.ID.str.rsplit("_", 1, expand=True)
    labels = labels[['SOPInstanceUID', 'Disease', 'Label']]
    labels = pd.pivot_table(labels, index="SOPInstanceUID", columns="Disease", values="Label")

    return labels


def load_df_with_labels_and_dicom_tags():
    tags = load_dicom_tags()
    labels = load_labels()

    return labels.merge(tags, on='SOPInstanceUID', how='outer')


def normalize_train(image, min_hu_value=-1000, max_hu_value=1000):
    """normalize hu values to -1 to 1 range"""
    image[image < min_hu_value] = min_hu_value
    image[image > max_hu_value] = max_hu_value
    image = (image - min_hu_value) / ((max_hu_value - min_hu_value) / 2) - 1
    return image


def load_scan_2dc(middle_img_path, slices_indices, slice_size, padded_size=None):
    slices_image = np.zeros((len(slices_indices), slice_size, slice_size))
    for slice_idx, img_num in enumerate(slices_indices):

        if img_num < 0 or img_num > (len(os.listdir(middle_img_path.parent)) - 1):
            slice_img = np.full((slice_size, slice_size), HU_AIR)
        else:
            slice_img = np.load(middle_img_path.parent.joinpath('{:03d}.npy'.format(img_num)))

        if slice_img.shape != (slice_size, slice_size):
            slice_img = cv2.resize(np.int16(slice_img), (slice_size, slice_size),
                                   interpolation=cv2.INTER_AREA)

        slices_image[slice_idx] = slice_img

    if padded_size is not None:
        margin = (padded_size - slice_size) // 2
        slices_image = np.pad(slices_image, ((0, 0), (margin, margin), (margin, margin)), mode='constant',
                              constant_values=HU_AIR)

    return slices_image


def load_seg_masks_2dc(middle_img_path, slices_indices, slice_size):
    masks_image = np.zeros((len(slices_indices), slice_size, slice_size), dtype=np.float32)

    for slice_idx, img_num in enumerate(slices_indices):

        if img_num < 0 or img_num > (len(os.listdir(middle_img_path.parent)) - 1):
            mask_img = np.zeros((slice_size, slice_size), dtype=np.uint8)
        else:
            exam_id = middle_img_path.parts[-3]
            mask_path = Path(SEG_MASKS_HOME)/"data/rsna/train"/exam_id/"masks/cropped400/any/"/'{:03d}.png'.format(img_num)
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                print("Error loading mask: ", mask_path)
                mask_img = np.zeros((slice_size, slice_size), dtype=np.uint8)

        if mask_img.shape != (slice_size, slice_size):
            mask_img = cv2.resize(mask_img, (slice_size, slice_size), interpolation=cv2.INTER_CUBIC)

        mask_img = np.float32(mask_img) / 255.0

        masks_image[slice_idx] = mask_img

    return masks_image


def crop_scan(scan, dest_shape, x, y, pad_value):
    dest_shape = np.array(dest_shape)
    center = np.array([y, x], dtype=np.int32)
    corner0 = center - dest_shape // 2
    corner1 = corner0 + dest_shape

    corner0_clipped = np.maximum(corner0, 0)
    corner1_clipped = np.minimum(corner1, scan.shape[1:])
    margin0 = np.abs(corner0 - corner0_clipped)
    margin1 = np.abs(corner1 - corner1_clipped)

    scan_cropped = np.zeros((scan.shape[0], dest_shape[0], dest_shape[1]), dtype=scan.dtype) + pad_value
    crop = scan[:, corner0_clipped[0]:corner1_clipped[0], corner0_clipped[1]:corner1_clipped[1]]
    scan_cropped[:, margin0[0]:dest_shape[0] - margin1[0], margin0[1]:dest_shape[1] - margin1[1]] = crop

    return scan_cropped


def load_seg(path):
    seg = nib.load(path).get_data()
    seg = seg.transpose(2, 1, 0)
    return seg


def transform_seg(seg, y2, z2, spacing, dest_shape):
    # shear params in vtk_image work with uniform spacing
    # for images with non-uniform z spacing, z needs to be scaled before computing shear angle
    scaled_z2 = z2 * (spacing[2] / spacing[1])
    rad_tilt = atan(scaled_z2 / y2)

    M = transforms3d.shears.sadn2aff(rad_tilt, [0, 1, 0], [-1, 0, 0], [seg.shape[0] / 2, 0, 0])
    seg = ndimage.affine_transform(seg, M, order=0)

    # pad image in y axis, so that the shape matches vtk_image output
    y_margin = (dest_shape[1] - seg.shape[1]) // 2
    seg_padded = np.zeros(dest_shape, dtype=seg.dtype)
    seg_padded[:, y_margin:y_margin + seg.shape[1], :] = seg

    return seg_padded


def load_seg_3d(seg_path, meta_path):
    seg = load_seg(seg_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    spacing = meta['spacing']
    x1, y1, z1, x2, y2, z2 = meta['image_orientation']
    crop_x = meta['crop_x']
    crop_y = meta['crop_y']
    pre_crop_shape = meta['pre_crop_shape']
    out_shape = meta['out_shape']

    seg_transformed = transform_seg(seg, y2, z2, spacing, pre_crop_shape)
    seg_transformed = crop_scan(seg_transformed, out_shape[1:], crop_x, crop_y, 0)

    return seg_transformed


def load_seg_slice(seg_path, meta_path, slice_num, slice_size):
    seg = load_seg_3d(seg_path, meta_path)
    seg = seg[slice_num]

    if seg.shape != (slice_size, slice_size):
        seg = skimage.transform.resize(np.float32(seg), (slice_size, slice_size),
                                       order=0, anti_aliasing=False).astype(seg.dtype)

    return seg


def draw_seg(img, seg, draw_any=False):
    colors = {
        0: (255, 237, 0),
        1: (212, 36, 0),
        2: (173, 102, 108),
        3: (0, 48, 114),
        4: (74, 87, 50),
        5: (66, 233, 245)
    }

    if draw_any:
        colors[6] = (66, 233, 245)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
    for class_, color in colors.items():
        img[seg[class_] > 0.5] += color
        img[seg[class_] > 0.5] /= 2

    step = 25
    counter = 0

    for class_, color in colors.items():
        if img[seg[class_] > 0.5].sum() > 0:
            cv2.circle(img, (step, step // 2 + counter * step),
                       step // 2, color, -1)
            cv2.putText(img, SEG_CLASSES[class_], (2 * step, step // 2 + counter * step),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, color)
            counter += 1

    return np.uint8(img)


def draw_labels(img, labels):
    """
    Draw colored image labels on image.
    :param img: image to draw labels on
    :param labels: list/array of 5 boolean or 0-1 values, e.g.: [0, 1, 0, 1, 1]
    """
    class_colors = {
        "epidural": (255, 237, 0),
        "intraparenchymal": (212, 36, 0),
        "intraventricular": (173, 102, 108),
        "subarachnoid": (0, 48, 114),
        "subdural": (74, 87, 50)
    }
    step = 25
    counter = 0

    for i, c in enumerate(class_colors.keys()):
        if labels[i]:
            cv2.circle(img, (step, step // 2 + counter * step), step // 2, class_colors[c], -1)
            cv2.putText(img, c, (2 * step, step // 2 + counter * step), cv2.FONT_HERSHEY_SIMPLEX, .5, class_colors[c])
            counter += 1


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))