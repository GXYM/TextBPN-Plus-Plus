import sys
import scipy.io as sio
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import pylab as plt
from matplotlib import cm
import os

def label2color(label):

    label = label.astype(np.uint16)
    
    height, width = label.shape
    color3u = np.zeros((height, width, 3), dtype=np.uint8)
    unique_labels = np.unique(label)

    if unique_labels[-1] >= 2**24:       
        raise RuntimeError('Error: label overflow!')

    for i in range(len(unique_labels)):
    
        binary = '{:024b}'.format(unique_labels[i])
        # r g b 3*8 24
        r = int(binary[::3][::-1], 2)
        g = int(binary[1::3][::-1], 2)
        b = int(binary[2::3][::-1], 2)

        color3u[label == unique_labels[i]] = np.array([r, g, b])

    return color3u


def vis_direction_field(gt_flux):

    norm_gt = np.sqrt(gt_flux[1, :, :] ** 2 + gt_flux[0, :, :] ** 2)
    angle_gt = 180 / math.pi * np.arctan2(gt_flux[1, :, :], gt_flux[0, :, :])

    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(121)
    ax1.set_title('Norm_gt')
    ax1.set_autoscale_on(True)
    im1 = ax1.imshow(norm_gt, cmap=cm.jet)
    plt.colorbar(im1, shrink=0.5)

    ax2 = fig.add_subplot(122)
    ax2.set_title('Angle_gt')
    ax2.set_autoscale_on(True)
    im2 = ax2.imshow(angle_gt, cmap=cm.jet)
    plt.colorbar(im2, shrink=0.5)

    plt.savefig('1.png')
    plt.close(fig)


def vis_flux(vis_image, pred_flux, gt_flux, gt_mask, image_name, save_dir):

    vis_image = vis_image.data.cpu().numpy()[0, ...]
    pred_flux = pred_flux.data.cpu().numpy()[0, ...]
    gt_flux = gt_flux.data.cpu().numpy()[0, ...]
    gt_mask = gt_mask.data.cpu().numpy()[0, ...]
    
    image_name = image_name[0]

    norm_pred = np.sqrt(pred_flux[1,:,:]**2 + pred_flux[0,:,:]**2)
    angle_pred = 180/math.pi*np.arctan2(pred_flux[1,:,:], pred_flux[0,:,:])

    norm_gt = np.sqrt(gt_flux[1,:,:]**2 + gt_flux[0,:,:]**2)
    angle_gt = 180/math.pi*np.arctan2(gt_flux[1,:,:], gt_flux[0,:,:])

    fig = plt.figure(figsize=(10,6))

    ax0 = fig.add_subplot(231)
    ax0.imshow(vis_image[:,:,::-1])

    ax1 = fig.add_subplot(232)
    ax1.set_title('Norm_gt')
    ax1.set_autoscale_on(True)
    im1 = ax1.imshow(norm_gt, cmap=cm.jet)
    plt.colorbar(im1,shrink=0.5)

    ax2 = fig.add_subplot(233)
    ax2.set_title('Angle_gt')
    ax2.set_autoscale_on(True)
    im2 = ax2.imshow(angle_gt, cmap=cm.jet)
    plt.colorbar(im2, shrink=0.5)

    ax5 = fig.add_subplot(234)
    color_mask = label2color(gt_mask)
    ax5.imshow(color_mask)

    ax4 = fig.add_subplot(235)
    ax4.set_title('Norm_pred')
    ax4.set_autoscale_on(True)
    im4 = ax4.imshow(norm_pred, cmap=cm.jet)
    plt.colorbar(im4,shrink=0.5)

    ax5 = fig.add_subplot(236)
    ax5.set_title('Angle_pred')
    ax5.set_autoscale_on(True)
    im5 = ax5.imshow(angle_pred, cmap=cm.jet)
    plt.colorbar(im5, shrink=0.5)

    plt.savefig(save_dir + image_name + '.png')
    plt.close(fig)
