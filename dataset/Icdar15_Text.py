#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'
import re
import os
import numpy as np
from util import strs
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
from util.io import read_lines
from util.misc import norm2


class Icdar15Text(TextDataset):

    def __init__(self, data_root, is_training=True, load_memory=False, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        self.image_root = os.path.join(data_root, 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(data_root,'Train' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)

        p = re.compile('.rar|.txt')
        self.image_list = [x for x in self.image_list if not p.findall(x)]
        p = re.compile('(.jpg|.JPG|.PNG|.JPEG)')
        self.annotation_list = ['{}'.format(p.sub("", img_name)) for img_name in self.image_list]

        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img_gt(item))

    @staticmethod
    def parse_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = read_lines(gt_path+".txt")
        polygons = []
        for line in lines:
            line = strs.remove_all(line.strip('\ufeff'), '\xef\xbb\xbf')
            gt = line.split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, gt[:8]))
            xx = [x1, x2, x3, x4]
            yy = [y1, y2, y3, y4]

            label = gt[-1].strip().replace("###", "#")

            pts = np.stack([xx, yy]).T.astype(np.int32)
            # d1 = norm2(pts[0] - pts[1])
            # d2 = norm2(pts[1] - pts[2])
            # d3 = norm2(pts[2] - pts[3])
            # d4 = norm2(pts[3] - pts[0])
            # if min([d1, d2, d3, d4]) < 2:
            #     continue
            polygons.append(TextInstance(pts, 'c', label))

        return polygons

    def load_img_gt(self, item):
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)
        try:
            # Read annotation
            annotation_id = self.annotation_list[item]
            annotation_path = os.path.join(self.annotation_root, annotation_id)
            polygons = self.parse_txt(annotation_path)
        except Exception as e:
            print(e)
            polygons = None

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id.split("/")[-1]
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
    import cv2
    from util.augmentation import Augmentation
    from util import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = Icdar15Text(
        data_root='../data/Icdar2015',
        is_training=True,
        transform=transform
    )

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]
    for idx in range(0, len(trainset)):
        t0 = time.time()
        img, train_mask, tr_mask = trainset[idx]
        img, train_mask, tr_mask = map(lambda x: x.cpu().numpy(), (img, train_mask, tr_mask))

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)
        print(idx, img.shape)

        for i in range(tr_mask.shape[2]):
            cv2.imshow("tr_mask_{}".format(i),
                       cav.heatmap(np.array(tr_mask[:, :, i] * 255 / np.max(tr_mask[:, :, i]), dtype=np.uint8)))

        cv2.imshow("train_mask", cav.heatmap(np.array(train_mask * 255 / np.max(train_mask), dtype=np.uint8)))

        cv2.imshow('imgs', img)
        cv2.waitKey(0)
