import glob
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


class DynamicFARDataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, batch_size, mask_path=None, augment=True, training=True, test_mask_path=None, world_size=1):
        super(DynamicFARDataset, self).__init__()
        self.config = config
        self.augment = augment
        self.training = training
        self.batch_size = batch_size
        self.mask_rate = config['mask_rate']
        self.round = config['round']  # for places2 round is 64
        self.use_mpe = config['use_mpe']
        self.pos_num = config['rel_pos_num']
        self.default_size = config['default_size']
        if training:
            self.input_size = config['default_size']
        else:
            self.input_size = config['eval_size']
        self.pos_size = config['pos_size']
        self.world_size = world_size

        self.data = []
        if flist.endswith('txt'):
            f = open(flist, 'r')
            for i in f.readlines():
                i = i.strip()
                self.data.append(i)
            f.close()
        else:
            self.data = glob.glob(flist + '/*')
            self.data = sorted(self.data, key=lambda x: x.split('/')[-1])

        if training:
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
            self.mask_list = glob.glob(test_mask_path + '/*')
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])

        self.ones_filter = np.ones((3, 3), dtype=np.float32)
        self.d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
        self.d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
        self.d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
        self.d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def reset_dataset(self, shuffled_idx):
        self.idx_map = {}
        barrel_idx = 0
        count = 0
        for sid in shuffled_idx:
            self.idx_map[sid] = barrel_idx
            count += 1
            if count == self.batch_size:
                count = 0
                barrel_idx += 1
        # random img size:256~512
        if self.training:
            barrel_num = int(len(self.data) / (self.batch_size * self.world_size))
            barrel_num += 2
            if self.round == 1:
                self.input_size = np.clip(np.arange(16, 33, step=(33 - 16) / barrel_num * 2).astype(int) * 16, 256, 512).tolist()
                self.input_size = self.input_size[::-1] + self.input_size
            else:
                self.input_size = []
                input_size = np.clip(np.arange(15, 33, step=(33 - 16) / barrel_num * 2 * self.round).astype(int) * 16, 256, 512).tolist()
                for _ in range(self.round + 1):
                    self.input_size.extend(input_size[::-1])
                    self.input_size.extend(input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        if type(self.input_size) == list:
            maped_idx = self.idx_map[index]
            if maped_idx > len(self.input_size) - 1:
                size = 512
            else:
                size = self.input_size[maped_idx]
        else:
            size = self.input_size
        self.feat_size = size / 16 # upsampled MAE feature size

        # load image
        img = cv2.imread(self.data[index])
        while img is None:
            print('Bad image {}...'.format(self.data[index]))
            idx = random.randint(0, len(self.data) - 1)
            img = cv2.imread(self.data[idx])
        # origin_h, origin_w = img.shape[:2]
        img = img[:, :, ::-1]
        # resize/crop if needed
        img = self.resize(img, size, size)
        img_256 = self.resize(img, 256, 256)

        # load mask
        mask = self.load_mask(img, index)
        mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
        mask_256[mask_256 > 0] = 255

        # augment data
        if self.augment and random.random() > 0.5 and self.training:
            img = img[:, ::-1, ...].copy()
            img_256 = img_256[:, ::-1, ...].copy()
        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[:, ::-1, ...].copy()
            mask_256 = mask_256[:, ::-1, ...].copy()
        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[::-1, :, ...].copy()
            mask_256 = mask_256[::-1, :, ...].copy()

        img_256 = Image.fromarray(img_256)
        img_256 = self.transform(img_256)

        batch = dict()
        batch['image'] = self.to_tensor(img)
        batch['img_256'] = img_256
        batch['mask'] = self.to_tensor(mask)
        batch['mask_256'] = self.to_tensor(mask_256)
        batch['size_ratio'] = size / self.default_size
        batch['name'] = self.load_name(index)
        batch['feat_size'] = self.feat_size
        if self.use_mpe:
            # load pos encoding
            rel_pos, abs_pos, direct = self.load_masked_position_encoding(mask)
            batch['rel_pos'] = torch.LongTensor(rel_pos)
            batch['abs_pos'] = torch.LongTensor(abs_pos)
            batch['direct'] = torch.LongTensor(direct)

        return batch

    def make_coord(self, shape, ranges=None, flatten=True):
        """ Make coordinates at grid centers.
        """
        coord_seqs = []
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    def load_masked_position_encoding(self, mask):
        ori_mask = mask.copy()
        ori_h, ori_w = ori_mask.shape[0:2]
        ori_mask = ori_mask / 255
        mask = cv2.resize(mask, (self.pos_size, self.pos_size), interpolation=cv2.INTER_AREA)
        mask[mask > 0] = 255
        h, w = mask.shape[0:2]
        mask3 = mask.copy()
        mask3 = 1. - (mask3 / 255.0)
        pos = np.zeros((h, w), dtype=np.int32)
        direct = np.zeros((h, w, 4), dtype=np.int32)
        i = 0
        while np.sum(1 - mask3) > 0:
            i += 1
            mask3_ = cv2.filter2D(mask3, -1, self.ones_filter)
            mask3_[mask3_ > 0] = 1
            sub_mask = mask3_ - mask3
            pos[sub_mask == 1] = i

            m = cv2.filter2D(mask3, -1, self.d_filter1)
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 0] = 1

            m = cv2.filter2D(mask3, -1, self.d_filter2)
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 1] = 1

            m = cv2.filter2D(mask3, -1, self.d_filter3)
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 2] = 1

            m = cv2.filter2D(mask3, -1, self.d_filter4)
            m[m > 0] = 1
            m = m - mask3
            direct[m == 1, 3] = 1

            mask3 = mask3_

        abs_pos = pos.copy()
        rel_pos = pos / (self.pos_size / 2)  # to 0~1 maybe larger than 1
        rel_pos = (rel_pos * self.pos_num).astype(np.int32)
        rel_pos = np.clip(rel_pos, 0, self.pos_num - 1)

        if ori_w != w or ori_h != h:
            rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            rel_pos[ori_mask == 0] = 0
            direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            direct[ori_mask == 0, :] = 0

        return rel_pos, abs_pos, direct

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        if self.training is False:
            mask = cv2.imread(self.mask_list[index % len(self.mask_list)], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode: 40% mask with random brush, 40% mask with coco mask, 20% with additions
            rdv = random.random()
            if rdv < self.mask_rate[0]:
                mask_index = random.randint(0, len(self.irregular_mask_list) - 1)
                mask = cv2.imread(self.irregular_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            elif rdv < self.mask_rate[1]:
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

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)

        return img

    def crop(self, img, height, width):
        imgh, imgw = img.shape[0:2]
        w_start = random.randint(0, imgw - width)
        h_start = random.randint(0, imgh - height)
        cropped_img = img[h_start:h_start + height, w_start:w_start + width, :]
        return cropped_img, w_start, h_start

    def load_flist(self, flist):

        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                # flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist = self.getfilelist(flist)
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def getfilelist(self, path):
        all_file = []
        for dir, folder, file in os.walk(path):
            for i in file:
                t = "%s/%s" % (dir, i)
                if t.endswith('.png') or t.endswith('.jpg') or t.endswith('.JPG') or \
                        t.endswith('.PNG') or t.endswith('.JPEG'):
                    all_file.append(t)
        return all_file
