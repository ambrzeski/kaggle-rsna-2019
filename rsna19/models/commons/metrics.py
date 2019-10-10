import numpy as np
import sklearn.metrics


"""
Multilabel classification metrics
"""


def accuracy(prediction, gt, threshold=0.5, per_class=False):
    """
    Expects arrays of shape: (batch, class_count). Array will be squeezed to handle single element dimensions.
    """
    prediction, gt = _preprocess(prediction, gt)
    tps, tns, fps, fns = confusion_matrix(prediction, gt, threshold)
    acc = 1.0 * (tps + tns) / (tps + tns + fps + fns)
    if per_class:
        return acc
    return acc.mean()


def sensitivity(prediction, gt, threshold=0.5, per_class=False):
    """
    Expects arrays of shape: (batch, class_count). Array will be squeezed to handle single element dimensions.
    """
    prediction, gt = _preprocess(prediction, gt)
    tps, tns, fps, fns = confusion_matrix(prediction, gt, threshold)
    with np.errstate(divide='ignore', invalid='ignore'):
        sens = np.nan_to_num(tps / (tps + fns))
    if per_class:
        return sens
    return sens.mean()


def specificity(prediction, gt, threshold=0.5, per_class=False):
    """
    Expects arrays of shape: (batch, class_count). Array will be squeezed to handle single element dimensions.
    """
    prediction, gt = _preprocess(prediction, gt)
    tps, tns, fps, fns = confusion_matrix(prediction, gt, threshold)
    with np.errstate(divide='ignore', invalid='ignore'):
        spec = np.nan_to_num(tns / (tns + fps))
    if per_class:
        return spec
    return spec.mean()


def precision(prediction, gt, threshold=0.5, per_class=False):
    """
    Expects arrays of shape: (batch, class_count). Array will be squeezed to handle single element dimensions.
    """
    prediction, gt = _preprocess(prediction, gt)
    tps, tns, fps, fns = confusion_matrix(prediction, gt, threshold)
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.nan_to_num(tps / (tps + fps))
    if per_class:
        return prec
    return prec.mean()


def recall(prediction, gt, threshold=0.5, per_class=False):
    """
    Expects arrays of shape: (batch, class_count). Array will be squeezed to handle single element dimensions.
    """
    return sensitivity(prediction, gt, threshold, per_class)


def f1score(prediction, gt, threshold=0.5, per_class=False):
    """
    Expects arrays of shape: (batch, class_count). Array will be squeezed to handle single element dimensions.
    """
    prec = precision(prediction, gt, threshold, per_class=True)
    rec = recall(prediction, gt, threshold, per_class=True)
    beta = 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.nan_to_num((1. + (beta ** 2)) * prec * rec / ((beta ** 2) * prec + rec))
    if per_class:
        return f1
    return f1.mean()


def f1score_spec(prediction, gt, threshold=0.5, per_class=False):
    """
    Expects arrays of shape: (batch, class_count). Array will be squeezed to handle single element dimensions.
    """
    sens = sensitivity(prediction, gt, threshold, per_class=True)
    spec = specificity(prediction, gt, threshold, per_class=True)
    beta = 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.nan_to_num((1. + (beta ** 2)) * sens * spec / ((beta ** 2) * spec + sens))
    if per_class:
        return f1
    return f1.mean()


def confusion_matrix(prediction, gt, threshold=0.5):
    """
    Expects arrays of shape: (batch, class_count). Array will be squeezed to handle single element dimensions.
    """
    prediction, gt = _preprocess(prediction, gt)
    tps = np.float32(sum(np.logical_and(gt == 1.0, prediction >= threshold)))
    fns = np.float32(sum(np.logical_and(gt == 1.0, prediction < threshold)))
    tns = np.float32(sum(np.logical_and(gt == 0.0, prediction < threshold)))
    fps = np.float32(sum(np.logical_and(gt == 0.0, prediction >= threshold)))
    return tps, tns, fps, fns


def roc_auc(prediction, gt, *args, **kwargs):
    """
    Expects arrays of shape: (batch, class_count). Array will be squeezed to handle single element dimensions.
    """
    prediction, gt = _preprocess(prediction, gt)
    num_classes = gt.shape[-1]
    aucs = []
    for i in range(num_classes):
        auc = sklearn.metrics.roc_auc_score(gt[:, i], prediction[:, i])
        aucs.append(auc)
    return np.array(aucs)


def best_fscore_spec_thresh(prediction, gt):
    max_fscore = np.zeros((prediction.shape[1]))
    max_fscore_thresholds = np.zeros((prediction.shape[1]))
    for threshold in np.linspace(0.01, 1, 100):
        curr_fscore = f1score_spec(prediction, gt, threshold, per_class=True)
        for i in range(0, prediction.shape[1]):
            if curr_fscore[i] > max_fscore[i]:
                max_fscore[i] = curr_fscore[i]
                max_fscore_thresholds[i] = threshold

    return max_fscore_thresholds


def _preprocess(prediction, gt):
    if prediction.ndim > 2:
        prediction = np.squeeze(prediction)
    if gt.ndim > 2:
        gt = np.squeeze(gt)
    return prediction, gt
