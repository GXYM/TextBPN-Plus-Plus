# -*- coding: utf-8 -*-
__author__ = "S.X.Zhang"
import warnings
warnings.filterwarnings("ignore")
import os
import re
import numpy as np
import scipy.io as io
from util import strs
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
import cv2
from util import io as libio


class ArtText(TextDataset):

    def __init__(self, data_root, ignore_list=None, is_training=True, load_memory=False, transform=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

        self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(data_root, 'gt', 'Train' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))
        self.annotation_list = ['{}'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]

        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img_gt(item))

    @staticmethod
    def parse_carve_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = libio.read_lines(gt_path + ".txt")
        polygons = []
        for line in lines:
            line = strs.remove_all(line, '\xef\xbb\xbf')
            gt = line.split(',')
            gt_corrdinate = gt[:-3]
            if len(gt_corrdinate) < 6:
                continue
            pts = np.stack([gt_corrdinate[0::2], gt_corrdinate[1::2]]).T.astype(np.int32)
            text = gt[-1].replace("\n","")
            polygons.append(TextInstance(pts, 'c', text))
        # print(polygon)
        return polygons

    def load_img_gt(self, item):
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)
        try:
            h, w, c = image.shape
            assert (c == 3)
        except:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        # polygons = self.parse_mat(annotation_path)
        polygons = self.parse_carve_txt(annotation_path)

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id
        data["image_path"] = image_path

        return data

    def __getitem__(self, item):

        if self.load_memory:
            data = self.datas[item]
        else:
            data = self.load_img_gt(item)

        if self.is_training:
            return self.get_training_data(data["image"], data["polygons"],
                                          image_id=data["image_id"], image_path=data["image_path"])
        else:
            return self.get_test_data(data["image"], data["polygons"],
                                      image_id=data["image_id"], image_path=data["image_path"])

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    import time
    from util.augmentation import Augmentation, BaseTransformNresize
    from util import canvas as cav

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = ArtText(
        data_root="/home/prir1005/pubdata/ArT",
        is_training=True,
        transform=transform,
    )

    t0 = time.time()
    image,  tr_mask, train_mask, label_mask, gt_points, ignore_tags = trainset[30]
    image,  tr_mask, train_mask, label_mask, gt_points, ignore_tags = \
        map(lambda x: x.cpu().numpy(), (image,  tr_mask, train_mask, label_mask, gt_points, ignore_tags))

    img = image.transpose(1, 2, 0)
    img = ((img * stds + means) * 255).astype(np.uint8)

    for i in range(tr_mask.shape[0]):
        heatmap = cav.heatmap(np.array(tr_mask[i, :, :] * 255 / np.max(tr_mask[i, :, :]), dtype=np.uint8))
        cv2.imshow("tr_mask_{}".format(i), heatmap)
        cv2.imshow('train_mask_{}'.format(i), cav.heatmap(np.array(train_mask[i] * 255 / np.max(train_mask[i]), dtype=np.uint8)))

    boundary_points = gt_points[np.where(ignore_tags != 0)[0]]
    ignore_points = gt_points[np.where(ignore_tags == -1)[0]]
    for i in range(tr_mask.shape[0]):
        im = img.copy()
        gt_point = boundary_points[:, i, :, :]
        ignore_point = ignore_points[:, i, :, :]
        cv2.drawContours(im, gt_point.astype(np.int32), -1, (0, 255, 0), 1)
        cv2.drawContours(im, ignore_point.astype(np.int32), -1, (0, 0, 255), 1)
        cv2.imshow('imgs_{}'.format(i), im)
        cv2.waitKey(0)






