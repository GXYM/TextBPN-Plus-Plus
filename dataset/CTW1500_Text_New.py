#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
from util.io import read_lines
from lxml import etree as ET


class Ctw1500Text_New(TextDataset):
    def __init__(self, data_root, is_training=True, load_memory=False, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(data_root, 'gt', 'train_labels' if is_training else 'test_labels')
        self.image_list = os.listdir(self.image_root)
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
        lines = read_lines(gt_path + ".txt")
        polygons = []
        for line in lines:
            line = line.split(",")
            gt = list(map(int, line[:-1]))
            pts = np.stack([gt[0::2], gt[1::2]]).T.astype(np.int32)
            label = line[-1].split("###")[-1].replace("###", "#")
            polygons.append(TextInstance(pts, 'c', label))

        return polygons

    @staticmethod
    def parse_carve_xml(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        root = ET.parse(gt_path + ".xml").getroot()

        polygons = []
        for tag in root.findall('image/box'):
            label = tag.find("label").text.replace("###", "#")
            gt = list(map(int, tag.find("segs").text.split(",")))
            pts = np.stack([gt[0::2], gt[1::2]]).T.astype(np.int32)

            if label != "#":
                x = []; y = []
                for cps in tag.findall("pts"):
                    x.append(cps.get("x"))
                    y.append(cps.get("y"))
                cts = np.stack([list(map(int, x)), list(map(int, y))]).T.astype(np.int32)
            else:
                cts =None

            polygons.append(TextInstance(pts, 'c', label, key_pts=cts))

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
        if self.is_training:
            annotation_id = self.annotation_list[item]
            annotation_path = os.path.join(self.annotation_root, annotation_id)
            polygons = self.parse_carve_xml(annotation_path)
            pass
        else:
            annotation_id = self.annotation_list[item]
            annotation_path = os.path.join(self.annotation_root, "000" + annotation_id)
            polygons = self.parse_carve_txt(annotation_path)

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
    import os
    import cv2
    from util.augmentation import Augmentation
    from util import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = Ctw1500Text_New(
        data_root='../data/CTW-1500',
        is_training=True,
        transform=transform
    )

    for idx in range(0, len(trainset)):
        t0 = time.time()
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags = trainset[idx]
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags\
            = map(lambda x: x.cpu().numpy(),
                  (img, train_mask, tr_mask, distance_field,
                   direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags))

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)

        distance_map = cav.heatmap(np.array(distance_field * 255 / np.max(distance_field), dtype=np.uint8))
        cv2.imshow("distance_map", distance_map)
        cv2.waitKey(0)

        direction_map = cav.heatmap(np.array(direction_field[0] * 255 / np.max(direction_field[0]), dtype=np.uint8))
        cv2.imshow("direction_field", direction_map)
        cv2.waitKey(0)

        from util.vis_flux import vis_direction_field
        vis_direction_field(direction_field)

        weight_map = cav.heatmap(np.array(weight_matrix * 255 / np.max(weight_matrix), dtype=np.uint8))
        cv2.imshow("weight_matrix", weight_map)
        cv2.waitKey(0)
