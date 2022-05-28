import cv2 as cv
import numpy as np
import pywt
import torch
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops

def _abstract_features(tensor_image, model, device):
    model.to(device)
    if device == torch.device("cpu"):
        tensor_image = tensor_image.cpu()
    else:
        tensor_image = tensor_image.cuda()
    return np.array(model(tensor_image).cpu().detach()).flatten()


def _wavelet_trans_features(grey_np_img, n_level=3):
    re = []  # 第n层所有节点的分解系数
    grey_np_img = np.array(grey_np_img, copy=True)
    wp = pywt.WaveletPacket2D(
        data=grey_np_img, wavelet='db1', mode='symmetric', maxlevel=n_level)
    for p in [n.path for nodes in wp.get_level(n_level, 'freq') for n in nodes]:
        re = np.append(re, np.array(
            [float(pow(np.linalg.norm(wp[p].data, ord=None), 2))]), axis=0)
    return re


def _comatrix_features(grey_np_img):
    ftrs = []
    grey_np_img = np.array(grey_np_img, copy=True)
    compress_gray = np.digitize(grey_np_img, np.linspace(0, 255, 64))
    comatrix = graycomatrix(compress_gray, np.linspace(10, 20, num=4),
                            [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                            256, symmetric=True, normed=True)
    for prop in {'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}:
        ftrs.append(graycoprops(comatrix, prop).flatten())

    return np.array(ftrs).flatten()


def _remove_unnecessary_region(mask):
    mask = np.array(mask, copy=True)
    n, labels = cv.connectedComponents(mask[:,:,0])
    edge_unique = np.unique(labels[[0, -1], :][:, [0, -1]])[1:]
    for i in edge_unique:
        mask[np.where(labels == i)] = 0
    return mask


def _get_area(img, hist, remove_unnecessary=False):
    hsvt = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsvt], [0, 1], hist, [0, 180, 0, 256], 1)

    # Now convolute with circular disc
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    cv.filter2D(dst, -1, disc, dst)

    # threshold and binary AND
    _, thresh = cv.threshold(dst, 0, 255, 0)
    thresh = cv.merge((thresh, thresh, thresh))
    if remove_unnecessary:
        thresh = _remove_unnecessary_region(thresh)
    return np.where(thresh[:, :, 0])[0].shape[0]


def _color_ratio_feature(im, n_hist, c_hist, b_hist, rm_unnecessary=False):
    '''rm_unnecessary: Blank areas connected to edges are not considered.'''
    n_area = _get_area(np.array(im.copy()), n_hist)
    c_area = _get_area(np.array(im.copy()), c_hist)
    b_area = _get_area(np.array(im.copy()), b_hist, rm_unnecessary)
    if c_area == 0: c_area = 1
    return np.array([n_area/c_area, b_area/c_area])


def get_flatten_rating_feature(im, index, transforms, encoder_model, device, n_hist, c_hist, b_hist):
    # print(f"{index} started.")
    im = im.convert('RGB')
    grey_np_im = np.array(im.copy().convert('L'))
    tensor_im = transforms(im.copy()).unsqueeze(0).to(device)
    ftr = []

    ftr = np.append(ftr, _abstract_features(tensor_im, encoder_model, device)) 
    # 小秘密: 由于数据量小, 其实加上抽象特征效果更差
    ftr = np.append(ftr, _comatrix_features(grey_np_im))
    ftr = np.append(ftr, _color_ratio_feature(im, n_hist, c_hist, b_hist, rm_unnecessary=(index=='e')))
    # print(f"{index} completed.")
    return ftr

