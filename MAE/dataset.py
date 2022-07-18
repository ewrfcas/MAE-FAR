import os
import random
import sys
from glob import glob

import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append('..')
import cv2


def to_int(x):
    return tuple(map(int, x))


class ImgMaskDataset(Dataset):
    def __init__(self, pt_dataset, mask_path=None, test_mask_path=None, is_train=False, mask_rates=None, image_size=256):

        self.is_train = is_train
        self.pt_dataset = pt_dataset

        self.image_id_list = []
        with open(self.pt_dataset) as f:
            for line in f:
                self.image_id_list.append(line.strip())

        if is_train:
            if len(mask_path) > 1:
                self.irregular_mask_list = []
                with open(mask_path[0]) as f:
                    for line in f:
                        self.irregular_mask_list.append(line.strip())
                self.irregular_mask_list = sorted(self.irregular_mask_list, key=lambda x: x.split('/')[-1])
                self.segment_mask_list = []
                with open(mask_path[1]) as f:
                    for line in f:
                        self.segment_mask_list.append(line.strip())
                self.segment_mask_list = sorted(self.segment_mask_list, key=lambda x: x.split('/')[-1])
            else:
                total_masks = []
                with open(mask_path[0]) as f:
                    for line in f:
                        total_masks.append(line.strip())
                random.shuffle(total_masks)
                self.irregular_mask_list = total_masks[:len(total_masks) // 2]
                self.segment_mask_list = total_masks[len(total_masks) // 2:]
        else:
            self.mask_list = glob(test_mask_path + '/*')
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])

        self.image_size = image_size
        self.training = is_train
        self.mask_rates = mask_rates

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.image_id_list)

    def load_mask(self, index):
        imgh, imgw = self.image_size, self.image_size

        # test mode: load mask non random
        if self.training is False:
            mask = cv2.imread(self.mask_list[index % len(self.mask_list)], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode: 40% mask with random brush, 40% mask with coco mask, 20% with additions
            rdv = random.random()
            if rdv < self.mask_rates[0]:
                mask_index = random.randint(0, len(self.irregular_mask_list) - 1)
                mask = cv2.imread(self.irregular_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            elif rdv < self.mask_rates[1]:
                mask_index = random.randint(0, len(self.segment_mask_list) - 1)
                mask = cv2.imread(self.segment_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            else:
                mask_index1 = random.randint(0, len(self.segment_mask_list) - 1)
                mask_index2 = random.randint(0, len(self.irregular_mask_list) - 1)
                mask1 = cv2.imread(self.segment_mask_list[mask_index1],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask2 = cv2.imread(self.irregular_mask_list[mask_index2],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

            if mask.shape[0] != imgh or mask.shape[1] != imgw:
                mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

    def to_tensor(self, img, norm=False):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def __getitem__(self, idx):
        selected_img_name = self.image_id_list[idx]
        img = cv2.imread(selected_img_name)
        while img is None:
            print('Bad image {}...'.format(selected_img_name))
            idx = random.randint(0, len(self.image_id_list) - 1)
            img = cv2.imread(self.image_id_list[idx])
        img = img[:, :, ::-1]
        img = Image.fromarray(img)
        if self.training is True:
            img = self.transform_train(img)

        # load mask
        mask = self.load_mask(idx)
        # augment data
        if self.training is True:
            if random.random() < 0.5:
                mask = mask[:, ::-1, ...].copy()
            if random.random() < 0.5:
                mask = mask[::-1, :, ...].copy()

        mask = self.to_tensor(mask)
        meta = {'img': img, 'mask': mask,
                'name': os.path.basename(selected_img_name)}
        return meta
