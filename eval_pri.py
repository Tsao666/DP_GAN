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
from PIL import Image
import pandas as pd


def sharpen(img, sigma=100):
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

    return usm


def path2tensor(path):
    image = Image.open(path).convert('RGB')
    image_tensor = image.resize((512, 256), Image.BICUBIC)
    image_tensor = np.transpose(image_tensor, (2, 0, 1))
    image_tensor = np.expand_dims(np.asarray(image_tensor), 0)
    image_tensor = image_tensor.astype(np.float32) / 255.
    image_tensor = (image_tensor - 0.5) / 0.5
    return np.asarray(image), torch.from_numpy(image_tensor)


# --- read options ---#
comparing_1 = '20240521ImproveMaskSampleC1.4SampleStep1000'
comparing_2 = '20240521ImproveMaskSampleC1SampleStep1000'
comparing_3 = '/app/DP_GAN/private2/UAV_instance_512_z128_400/best/image'
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
prefixes = ['fake', 'background', 'RI', 'RO']
row_list = list()
row_dict = {p: list() for p in prefixes}
for idx, path in enumerate(os.listdir(os.path.join(opt.results_dir, opt.name, opt.ckpt_iter, 'image'))):
    # if idx == 1:
    #     break
    # print(path)
    img_1, image_tensor_1 = path2tensor(os.path.join(opt.results_dir, opt.name, opt.ckpt_iter, 'image', path))
    img_2, image_tensor_2 = path2tensor(os.path.join(comparing_1, path))
    img_3, image_tensor_3 = path2tensor(os.path.join(comparing_2, path))
    img_4, image_tensor_4 = path2tensor(os.path.join(comparing_3, path))

    preds_1, score_1 = model(image_tensor_1, None, 'eval', None)
    preds_2, score_2 = model(image_tensor_2, None, 'eval', None)
    preds_3, score_3 = model(image_tensor_3, None, 'eval', None)
    preds_4, score_4 = model(image_tensor_4, None, 'eval', None)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # D net output channel: 0 - fake, 1 - background, 2 - RI, 3 - RO.
    preds_1 = preds_1.cpu().numpy()[0]
    preds_2 = preds_2.cpu().numpy()[0]
    preds_3 = preds_3.cpu().numpy()[0]
    preds_4 = preds_4.cpu().numpy()[0]
    name = path.split('.')[0]

    row = dict()
    c_maps_1 = list()
    c_maps_2 = list()
    c_maps_3 = list()
    c_maps_4 = list()
    vals_1 = list()
    vals_2 = list()
    vals_3 = list()
    vals_4 = list()
    for prefix, channel_1, channel_2, channel_3, channel_4 in zip(prefixes, preds_1, preds_2, preds_3, preds_4):
        c_map_1 = (channel_1 - channel_1.min()) / (channel_1.max() - channel_1.min())
        c_map_2 = (channel_2 - channel_2.min()) / (channel_2.max() - channel_2.min())
        c_map_3 = (channel_3 - channel_3.min()) / (channel_3.max() - channel_3.min())
        c_map_4 = (channel_4 - channel_4.min()) / (channel_4.max() - channel_4.min())
        c_maps_1.append(c_map_1)
        c_maps_2.append(c_map_2)
        c_maps_3.append(c_map_3)
        c_maps_4.append(c_map_4)
        vals_1.append(channel_1.mean())
        vals_2.append(channel_2.mean())
        vals_3.append(channel_3.mean())
        vals_4.append(channel_4.mean())
        row[prefix] = 'Henry' if channel_1.mean() >= channel_2.mean() else 'Tony'
        row_dict[prefix].append({'name': name, 'Henry1': channel_1.mean(), 'Henry2': channel_4.mean(), 'Tony1': channel_2.mean(), 'Tony2': channel_3.mean()})
    row_list.append({'name': name, **row})

    # fig = plt.figure(figsize=(20, 8))
    # plt.subplot(3, 5, 1)
    # plt.title('Henry')
    # plt.imshow(img_1)
    # plt.subplot(3, 5, 6)
    # plt.title('Tony1')
    # plt.imshow(img_2)
    # plt.subplot(3, 5, 11)
    # plt.title('Tony2')
    # plt.imshow(img_3)
    # for i in range(4):
    #     plt.subplot(3, 5, i + 2)
    #     plt.title(f'{prefixes[i]}: {vals_1[i]:.4f}')
    #     plt.imshow(c_maps_1[i])
    #     plt.subplot(3, 5, i + 5 + 2)
    #     plt.title(f'{prefixes[i]}: {vals_2[i]:.4f}')
    #     plt.imshow(c_maps_2[i])
    #     plt.subplot(3, 5, i + 10 + 2)
    #     plt.title(f'{prefixes[i]}: {vals_3[i]:.4f}')
    #     plt.imshow(c_maps_3[i])
    # plt.savefig(os.path.join(save_path, name), dpi=100)
    # plt.close(fig)
# pd.DataFrame(row_list).to_csv('20240527-higher table.csv', index=False)
{pd.DataFrame(v).to_csv(f'20240528-{k} table.csv', index=False) for k, v in row_dict.items()}
