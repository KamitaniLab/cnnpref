#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''generate preferred image for the target uints'''

# Author: Shen Guo-Hua <shen-gh@atr.jp>

__author__ = 'sgh'

# version: v1

# import
import os
import numpy as np
import PIL.Image
import caffe
from datetime import datetime

from .utils import img_preprocess, img_deprocess, normalise_img, p_norm, TV_norm, image_norm, gaussian_blur, clip_extreme_pixel, clip_small_norm_pixel, clip_small_contribution_pixel

# main function
def generate_image(net, layer, feature_mask,
                   feature_weight = 1.,
                   initial_image = None,
                   iter_n = 200,
                   lr_start = 2., lr_end = 1e-10,
                   momentum_start = 0.9, momentum_end = 0.9,
                   decay_start = 0.1, decay_end = 0.1,
                   grad_normalize = True,
                   image_jitter = True, jitter_size = 32,
                   image_blur = True, sigma_start = 2., sigma_end = 0.5,
                   use_p_norm_reg = False, p = 3, lamda_start = 0.5, lamda_end = 0.5,
                   use_TV_norm_reg = False, TVbeta = 2, TVlamda_start = 0.5, TVlamda_end = 0.5,
                   clip_extreme = False, clip_extreme_every = 4, e_pct_start = 1, e_pct_end = 1,
                   clip_small_norm = False, clip_small_norm_every = 4, n_pct_start = 5., n_pct_end = 5.,
                   clip_small_contribution = False, clip_small_contribution_every = 4, c_pct_start = 5., c_pct_end = 5.,
                   disp_every = 1,
                   save_intermediate = False, save_intermediate_every = 1, save_intermediate_path = None
                   ):
    
    '''Generate preferred image for the target uints using gradient descent with momentum.
    
    Parameters
    ----------
    net: caffe.Classifier or caffe.Net object
        CNN model coresponding to the target CNN features.
    layer: str
        The name of the layer for the target units.
    feature_mask: ndarray
        The mask used to select the target units.
        The shape of the mask should be the same as that of the CNN features in that layer.
        The values of the mask array are binary, (1: target uint; 0: irrelevant unit)
    Optional Parameters
    ----------
    feature_weight: float or ndarray
        The weight for each target unit.
        If it is scalar, the scalar will be used as the universal weight for all units.
        If it is numpy array, it allows to specify different weights for different uints.
    initial_image: ndarray
        Initial image for the optimization.
        Use random noise as initial image by setting to None.
    iter_n: int
        The total number of iterations.
    lr_start: float
        The learning rate at start of the optimization.
        The learning rate will linearly decrease from lr_start to lr_end during the optimization.
    lr_end: float
        The learning rate at end of the optimization.
        The learning rate will linearly decrease from lr_start to lr_end during the optimization.
    momentum_start: float
        The momentum (gradient descend with momentum) at start of the optimization.
        The momentum will linearly decrease from momentum_start to momentum_end during the optimization.
    momentum_end: float
        The momentum (gradient descend with momentum) at the end of the optimization.
        The momentum will linearly decrease from momentum_start to momentum_end during the optimization.
    decay_start: float
        The decay rate of the image pixels at start of the optimization.
        The decay rate will linearly decrease from decay_start to decay_end during the optimization.
    decay_end: float
        The decay rate of the image pixels at the end of the optimization.
        The decay rate will linearly decrease from decay_start to decay_end during the optimization.
    grad_normalize: bool
        Normalise the gradient or not for each iteration.
    image_jitter: bool
        Use image jittering or not.
        If true, randomly shift the intermediate reconstructed image for each iteration.
    jitter_size: int
        image jittering in number of pixels.
    image_blur: bool
        Use image smoothing or not.
        If true, smoothing the image for each iteration.
    sigma_start: float
        The size of the gaussian filter for image smoothing at start of the optimization.
        The sigma will linearly decrease from sigma_start to sigma_end during the optimization.
    sigma_end: float
        The size of the gaussian filter for image smoothing at the end of the optimization.
        The sigma will linearly decrease from sigma_start to sigma_end during the optimization.
    use_p_norm_reg: bool
        Use p-norm loss for image or not as regularization term.
    p: float
        The order of the p-norm loss of image
    lamda_start: float
        The weight for p-norm loss at start of the optimization.
        The lamda will linearly decrease from lamda_start to lamda_end during the optimization.
    lamda_end: float
        The weight for p-norm loss at the end of the optimization.
        The lamda will linearly decrease from lamda_start to lamda_end during the optimization.
    use_TV_norm_reg: bool
        Use TV-norm or not as regularization term.
    TVbeta: float
        The order of the TV-norm.
    TVlamda_start: float
        The weight for TV-norm regularization term at start of the optimization.
        The TVlamda will linearly decrease from TVlamda_start to TVlamda_end during the optimization.
    TVlamda_end: float
        The weight for TV-norm regularization term at the end of the optimization.
        The TVlamda will linearly decrease from TVlamda_start to TVlamda_end during the optimization.
    clip_extreme: bool
        Clip or not the pixels with extreme high or low value.
    clip_extreme_every: int
        Clip the pixels with extreme value every n iterations.
    e_pct_start: float
        the percentage of pixels to be clipped at start of the optimization.
        The percentage will linearly decrease from e_pct_start to e_pct_end during the optimization.
    e_pct_end: float
        the percentage of pixels to be clipped at the end of the optimization.
        The percentage will linearly decrease from e_pct_start to e_pct_end during the optimization.
    clip_small_norm: bool
        Clip or not the pixels with small norm of RGB valuse.
    clip_small_norm_every: int
        Clip the pixels with small norm every n iterations
    n_pct_start: float
        The percentage of pixels to be clipped at start of the optimization.
        The percentage will linearly decrease from n_pct_start to n_pct_end during the optimization.
    n_pct_end: float
        The percentage of pixels to be clipped at start of the optimization.
        The percentage will linearly decrease from n_pct_start to n_pct_end during the optimization.
    clip_small_contribution: bool
        Clip or not the pixels with small contribution: norm of RGB channels of (img*grad).
    clip_small_contribution_every: int
        Clip the pixels with small contribution every n iterations.
    c_pct_start: float
        The percentage of pixels to be clipped at start of the optimization.
        The percentage will linearly decrease from c_pct_start to c_pct_end during the optimization.
    c_pct_end: float
        The percentage of pixels to be clipped at the end of the optimization.
        The percentage will linearly decrease from c_pct_start to c_pct_end during the optimization.
    disp_every: int
        Display the optimization information for every n iterations.
    save_intermediate: bool
        Save the intermediate reconstruction or not.
    save_intermediate_every: int
        Save the intermediate reconstruction for every n iterations.
    save_intermediate_path: str
        The path to save the intermediate reconstruction.
    
    Returns
    -------
    img: ndarray
        The preferred image [227x227x3].
    '''
    
    # make save dir
    if save_intermediate:
        if save_intermediate_path is None:
            save_intermediate_path = os.path.join('.','prefer_img_gd_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
        if not os.path.exists(save_intermediate_path):
            os.makedirs(save_intermediate_path)
    
    # image size
    img_size = net.blobs['data'].data.shape[-3:]
    
    # image mean
    img_mean = net.transformer.mean['data']
    
    # image norm
    noise_img = np.random.randint(0,256,(img_size[1],img_size[2],img_size[0]))
    img_norm0 = np.linalg.norm(noise_img)
    img_norm0 = img_norm0/2.
    
    # initial image
    if initial_image is None:
        initial_image = np.random.randint(0,256,(img_size[1],img_size[2],img_size[0]))
    if save_intermediate:
        save_name = 'initial_image.jpg'
        PIL.Image.fromarray(np.uint8(initial_image)).save(os.path.join(save_intermediate_path,save_name))
    
    # iteration for gradient descent
    img = initial_image.copy()
    img = img_preprocess(img,img_mean)
    delta_img = np.zeros_like(img)
    feat_grad = np.zeros_like(net.blobs[layer].diff[0])
    feat_grad[feature_mask==1] = -1.  # here we use gradient descent, so the gradient is negative, in order to make the target units have high positive activation;
    feat_grad = feat_grad * feature_weight
    for t in xrange(iter_n):
        # parameters
        lr = lr_start + t * (lr_end - lr_start) / iter_n
        momentum = momentum_start + t * (momentum_end - momentum_start) / iter_n
        decay = decay_start + t * (decay_end - decay_start) / iter_n
        sigma = sigma_start + t * (sigma_end - sigma_start) / iter_n
        
        # shift
        if image_jitter:
            ox, oy = np.random.randint(-jitter_size, jitter_size+1, 2)
            img = np.roll(np.roll(img, ox, -1), oy, -2)
            delta_img = np.roll(np.roll(delta_img, ox, -1), oy, -2)
        
        # forward
        net.blobs['data'].data[0] = img.copy()
        net.forward(end=layer)
        feat = net.blobs[layer].data[0][feature_mask==1]
        feat_abs_mean = np.mean(np.abs(feat))
        
        # backward for net
        net.blobs[layer].diff[0] = feat_grad.copy()
        net.backward(start=layer)
        net.blobs[layer].diff.fill(0.)
        grad = net.blobs['data'].diff[0].copy()
        
        # normalize gradient
        if grad_normalize:
            grad_mean = np.abs(grad).mean()
            if grad_mean>0:
                grad = grad / grad_mean
        
        # gradient with momentum
        delta_img = delta_img * momentum + grad
        
        # p norm regularization
        if use_p_norm_reg:
            lamda = lamda_start + t * (lamda_end - lamda_start) / iter_n
            loss_r, grad_r = p_norm(img, p)
            loss_r = loss_r / (img_norm0 ** 2)
            grad_r = grad_r / (img_norm0 ** 2)
            if grad_normalize:
                grad_mean = np.abs(grad_r).mean()
                if grad_mean>0:
                    grad_r = grad_r / grad_mean
            err = err + lamda * loss_r
            delta_img = delta_img + lamda * grad_r
        
        # TV norm regularization
        if use_TV_norm_reg:
            TVlamda = TVlamda_start + t * (TVlamda_end - TVlamda_start) / iter_n
            loss_r, grad_r = TV_norm(img, opts['TVbeta'])
            loss_r = loss_r / (img_norm0 ** 2)
            grad_r = grad_r / (img_norm0 ** 2)
            if grad_normalize:
                grad_mean = np.abs(grad_r).mean()
                if grad_mean>0:
                    grad_r = grad_r / grad_mean
            err = err + TVlamda * loss_r
            delta_img = delta_img + TVlamda * grad_r
        
        # image update
        img = img - lr * delta_img
        
        # clip pixels with extreme value
        if clip_extreme and (t+1)%clip_extreme_every==0:
            e_pct = e_pct_start + t * (e_pct_end - e_pct_start) / iter_n
            img = clip_extreme_pixel(img,e_pct)
        
        # clip pixels with small norm
        if clip_small_norm and (t+1)%clip_small_norm_every==0:
            n_pct = n_pct_start + t * (n_pct_end - n_pct_start) / iter_n
            img = clip_small_norm_pixel(img,n_pct)
        
        # clip pixels with small contribution
        if clip_small_contribution and (t+1)%clip_small_contribution_every==0:
            c_pct = c_pct_start + t * (c_pct_end - c_pct_start) / iter_n
            img = clip_small_contribution_pixel(img,grad,c_pct)
        
        # unshift
        if image_jitter:
            img = np.roll(np.roll(img, -ox, -1), -oy, -2)
            delta_img = delta_img - grad
            delta_img = np.roll(np.roll(delta_img, -ox, -1), -oy, -2)
            delta_img = delta_img + grad
        
        # L_2 decay
        img = (1-decay) * img
        
        # gaussian blur
        if image_blur:
            img = gaussian_blur(img,sigma)
        
        # disp info
        if (t+1)%disp_every==0:
            print('iter=%d; mean(abs(feat))=%g;'%(t+1,feat_abs_mean))
        
        # save image
        if save_intermediate and ((t+1)%save_intermediate_every==0):
            save_name = '%05d.jpg'%(t+1)
            PIL.Image.fromarray(normalise_img(img_deprocess(img,img_mean))).save(os.path.join(save_intermediate_path,save_name))
            #print(img.dtype)
    
    # return img
    return img_deprocess(img,img_mean)

