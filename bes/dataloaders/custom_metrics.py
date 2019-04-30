import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
import numpy as np


SMOOTH = 1e-12


def iou_score_batch(gt, pr):
    ''' function to calculate the mean IOU for a batch, calculating only for
    the positive class
    '''
    batch_size = gt.shape[0]
    average_per_batch = batch_size
    IOU_total = 0;
    for i in range(batch_size):
        # calculate the ground truth coverage per sample
        gt_mask = gt[i,:]
        H, W, C = gt_mask.shape
        coverage_gt = np.sum(gt_mask) / (H * W) 
        coverage_pr = np.sum(pr[i,:]) / (H * W)  
        if coverage_gt == 0 and coverage_pr == 0:
            IOU_total += 1.0
        elif coverage_gt == 0 and coverage_pr != 0:
        	IOU_total += 0.0     
        else:
            IOU_total += jaccard_similarity_margo(gt_mask, pr[i,:])

    return IOU_total/average_per_batch




def binary_accuracy_score_batch(gt, pr):
    ''' function to calculate the mean IOU for a batch, calculating only for
    the positive class
    K.mean(K.equal(y_true, K.round(y_pred))) -- in Keras
    '''
    batch_size = gt.shape[0]
    average_per_batch = batch_size
    accuracy_total = 0;
    for i in range(batch_size):
        # calculate the ground truth coverage per sample
        gt_mask = gt[i,:].astype(int)
        pr_mask = pr[i,:].astype(int)
        
        accuracy = (gt_mask  == np.round(pr_mask)).all(axis=(0,2)).mean() # tolerance just in case, not really needed coz the threshold before
        accuracy_total += accuracy
    return accuracy_total/average_per_batch




def jaccard_similarity_margo(gt, pr, smooth=SMOOTH):
    intersection = np.sum(np.sum(gt * pr))
    union = np.sum(np.sum(gt + pr)) - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou


def iou_score(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
    and is defined as the size of the intersection divided by the size of the union of the sample sets:

    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        IoU/Jaccard score in range [0, 1]

    .. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index

    """
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    intersection = K.sum(gt * pr, axis=axes)
    union = K.sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    if per_image:
        iou = K.mean(iou, axis=0)

    # weighted mean per class
    iou = K.mean(iou * class_weights)

    return iou
