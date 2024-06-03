import os
import cv2
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
import time
import torch
from torchvision import transforms as TR
import numpy as np
import matplotlib.pyplot as plt


def sharpen(img, sigma=100):
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

    return usm


# --- read options ---#
opt = config.read_arguments(train=False, eval=True)

save_path = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter, 'gen_label')
if not os.path.isdir(save_path):
    os.mkdir(save_path)

# --- create dataloader ---#
_, dataloader_val = dataloaders.get_dataloaders(opt)

# --- create utils ---#
image_saver = utils.results_saver(opt)

# --- create models ---#
model = models.DP_GAN_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

# --- iterate over dataset ---#
for idx, data_i in enumerate(dataloader_val):
    if idx == 1:
        break
    print('name', data_i['name'], data_i['label'].size())
    preds, score = model(data_i['image'], None, 'eval', None)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # D net output channel: 0 - fake, 1 - background, 2 - RI, 3 - RO.
    preds = preds.cpu().numpy()
    for idx, pred in enumerate(preds):
        name = data_i['name'][idx].split('.')[0]

        thres = 0.5
        c_maps = list()
        for channel in pred:
            c_map = (channel - channel.min()) / (channel.max() - channel.min())
            thres_idx = np.argwhere(c_map >= thres)
            new_c_map = np.zeros(c_map.shape, np.uint8)
            # new_c_map[thres_idx[:, 0], thres_idx[:, 1]] = 255
            new_c_map = (c_map * 255).astype(np.uint8)
            # new_c_map = sharpen(new_c_map, 5)
            c_maps.append(new_c_map)

        fig = plt.figure(figsize=(20, 10))
        plt.subplot(231)
        plt.imshow((((data_i['image'][idx].cpu().numpy().transpose((1, 2, 0)) * 0.5) + 0.5) * 255).astype(np.uint8))
        plt.subplot(232)
        plt.imshow(data_i['label'][idx].cpu().numpy().transpose((1, 2, 0)).astype(np.uint8))
        plt.subplot(233)
        plt.imshow(c_maps[0])
        plt.subplot(234)
        plt.imshow(c_maps[1])
        plt.subplot(235)

        plt.imshow(c_maps[2])

        # ri = c_maps[2]
        # contours, hierarchy = cv2.findContours(ri, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # ri_contour = cv2.drawContours(ri, contours, -1, 255, thickness=2)
        # plt.imshow(ri_contour)

        plt.subplot(236)
        plt.imshow(c_maps[3])
        plt.savefig(os.path.join(save_path, name), dpi=600)
        # for c_idx, channel in enumerate(i):
        #     print(c_idx, channel.min(), channel.max())
    # print('pred', pred.shape)
