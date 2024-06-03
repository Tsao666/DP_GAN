import random
import torch
from torchvision import transforms as TR
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd


class UAVDataset(torch.utils.data.Dataset):
    def __init__(self, opt, competition, set_name, target=None):
        # 428 * 240
        self.load_size = 512
        opt.crop_size = 512
        opt.aspect_ratio = 2.0
        opt.label_nc = 3
        opt.semantic_nc = 3  # label_nc
        # every value in label pixel means:
        # 0: grass
        # 1: RI
        # 2: RO

        self.opt = opt
        self.competition = competition
        self.set_name = set_name  # train, valid, test, full-train, full-valid, public, private
        self.target = target
        self.competition_id = 1 if self.competition == '34' else 2
        self.load_size = 512 if self.opt.phase == "test" or self.set_name in ['valid', 'test', 'public', 'private'] else 620
        self.images, self.labels, self.names = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx]).convert('L')
        height, width = label.height, label.width
        pixel_val = 2 if 'RO' in self.names[idx] else 1

        label_array = np.array(label)
        new_label = np.zeros((height, width), dtype=np.uint8)
        idx_255 = np.argwhere(label_array == 255)

        # mode: LR
        # for r in range(height):
        #     indices = idx_255[idx_255[:, 0] == r]
        #     if len(indices) == 1:
        #         new_label[r, indices[0, 1]] = pixel_val
        #     if len(indices) >= 2:
        #         new_label[r, indices[:, 1].min(): indices[:, 1].max() + 1] = pixel_val

        # mode: TB
        # for c in range(width):
        #     indices = idx_255[idx_255[:, 1] == c]
        #     if len(indices) == 1:
        #         new_label[indices[0, 0], c] = pixel_val
        #     if len(indices) >= 2:
        #         new_label[indices[:, 0].min(): indices[:, 0].max() + 1, c] = pixel_val

        # mode: contour
        contours, hierarchy = cv2.findContours(label_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_label = cv2.drawContours(new_label, contours, -1, pixel_val, -1)

        label = Image.fromarray(new_label)

        image, label = self.transforms(image, label)
        label = label * 255
        return {"image": image, "label": label, "name": self.names[idx]}

    def list_images(self):
        if self.set_name in ['public', 'private']:
            folder_name = 'label_img' if self.competition == '34' else 'img'
            path1 = f'/datasets/UAV/{self.competition}_Competition {self.competition_id}_public testing dataset/{folder_name}'
            path2 = f'/datasets/UAV/{self.competition}_Competition {self.competition_id}_Private Test Dataset/{folder_name}'
            names1 = os.listdir(path1)
            names2 = os.listdir(path2)
            if self.competition == '34':
                images = ['empty.jpg'] * len(names1 + names2)
                labels = [f'{path1}/{name}' for name in names1] + [f'{path2}/{name}' for name in names2]
            else:
                images = [f'{path1}/{name}' for name in names1] + [f'{path2}/{name}' for name in names2]
                labels = ['empty.png'] * len(names1 + names2)
            return images, labels, names1 + names2

        rgb_path = f'/datasets/UAV/{self.competition}_Competition {self.competition_id}_Training dataset/Training dataset/img'
        gray_path = f'/datasets/UAV/{self.competition}_Competition {self.competition_id}_Training dataset/Training dataset/label_img'
        df = pd.read_csv(f'/app/Generative-AI-Navigation-Information-Competition-for-UAV-Reconnaissance-in-Natural-Environments/data/train_{self.competition}.csv')
        df['data_path'] = df['data'].apply(lambda x: f'{rgb_path}/{x}')
        df['label_path'] = df['label'].apply(lambda x: f'{gray_path}/{x}')
        if self.set_name in ['full-train', 'full-valid']:
            df = self.assign_split_subset(df)
            df = df[df['subset'].eq(self.set_name.split('-')[1])]
        elif self.set_name in ['train', 'valid']:
            df = df[df['set_name'].eq('train')]
            df = self.assign_split_subset(df)
            df = df[df['subset'].eq(self.set_name)]
        else:
            df = df[df['set_name'].eq('test')]
        if self.competition == '34':
            names = df['data'].to_list()
        else:
            names = df['label'].to_list()
        images = df['data_path'].to_list()
        labels = df['label_path'].to_list()
        return images, labels, names

    def assign_split_subset(self, df):
        mask = df.groupby('class').apply(lambda x: x.sample(frac=.9, random_state=42), include_groups=False).reset_index()['level_1'].tolist()
        df['subset'] = None
        df.loc[mask, 'subset'] = 'train'
        df['subset'] = df['subset'].fillna('valid')
        return df

    def transforms(self, image, label):
        assert image.size == label.size
        # resize
        new_height, new_width = (int(self.load_size / self.opt.aspect_ratio), self.load_size)
        image = TR.functional.resize(image, (new_height, new_width), Image.BICUBIC)
        label = TR.functional.resize(label, (new_height, new_width), Image.NEAREST)
        # crop
        crop_x = random.randint(0, np.maximum(0, new_width - self.opt.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - (self.opt.crop_size / self.opt.aspect_ratio)))
        image = image.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + (self.opt.crop_size / self.opt.aspect_ratio)))
        label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + (self.opt.crop_size / self.opt.aspect_ratio)))
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.set_name in ['valid', 'test']):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
            if random.random() < 0.5:
                image = TR.functional.vflip(image)
                label = TR.functional.vflip(label)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label
