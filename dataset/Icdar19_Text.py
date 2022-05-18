# -*- coding: utf-8 -*-
__author__ = "S.X.Zhang"
import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
from util.io import read_lines
from util.misc import norm2
from util import strs
import cv2


class Mlt2019Text(TextDataset):

    def __init__(self, data_root, is_training=True, transform=None, load_memory=False, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        self.image_root = os.path.join(data_root, 'TrainImages' if is_training else 'TestImages')
        self.annotation_root = os.path.join(data_root, 'Train_gt' if is_training else None)

        if is_training:
            with open(os.path.join(data_root, 'train_list.txt')) as f:
                self.img_list = [line.strip() for line in f.readlines()]

            if ignore_list:
                with open(ignore_list) as f:
                    ignore_list = f.readlines()
                    ignore_list = [line.strip() for line in ignore_list]
            else:
                ignore_list = []
        else:
            with open(os.path.join(data_root, 'test_list.txt')) as f:
                self.img_list = [line.strip() for line in f.readlines()]

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
        lines = read_lines(gt_path + ".txt")
        polygons = []
        for line in lines:
            line = strs.remove_all(line.strip('\ufeff'), '\xef\xbb\xbf')
            gt = line.split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, gt[:8]))
            xx = [x1, x2, x3, x4]
            yy = [y1, y2, y3, y4]
            if gt[-1].strip() == "###":
                label = gt[-1].strip().replace("###", "#")
            else:
                label = "GG"
            pts = np.stack([xx, yy]).T.astype(np.int32)

            polygons.append(TextInstance(pts, 'c', label))

        return polygons

    def load_img_gt(self, item):
        image_id = self.img_list[item]

        if self.is_training:
            # Read annotation
            annotation_path = os.path.join(self.annotation_root, image_id.split(".")[0])

            polygons = self.parse_txt(annotation_path)
        else:
            polygons = None

        # Read image data
        image_path = os.path.join(self.image_root, image_id)
        image = pil_load_img(image_path)
        try:
            h, w, c = image.shape
            assert (c == 3)
        except:
            # print(image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)

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
        return len(self.img_list)


if __name__ == '__main__':
    import os
    from util.augmentation import BaseTransform, Augmentation
    from util import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = Mlt2019Text(
        data_root='../data/MLT-2019',
        is_training=True,
        transform=transform
    )
    for idx in range(0, len(trainset)):
        t0 = time.time()
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags = trainset[idx]
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags \
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
        # cv2.waitKey(0)

        boundary_point = ctrl_points[np.where(ignore_tags != 0)[0]]
        for i, bpts in enumerate(boundary_point):
            cv2.drawContours(img, [bpts.astype(np.int32)], -1, (0, 255, 0), 1)
            for j, pp in enumerate(bpts):
                if j == 0:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (255, 0, 255), -1)
                elif j == 1:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 255), -1)
                else:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 0, 255), -1)

            ppts = proposal_points[i]
            cv2.drawContours(img, [ppts.astype(np.int32)], -1, (0, 0, 255), 1)
            for j, pp in enumerate(ppts):
                if j == 0:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (255, 0, 255), -1)
                elif j == 1:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 255), -1)
                else:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 0, 255), -1)
            cv2.imshow('imgs', img)
            cv2.waitKey(0)

        # from util.misc import split_edge_seqence
        # from cfglib.config import config as cfg
        #
        # ret, labels = cv2.connectedComponents(np.array(distance_field >0.35, dtype=np.uint8), connectivity=4)
        # for idx in range(1, ret):
        #     text_mask = labels == idx
        #     ist_id = int(np.sum(text_mask*tr_mask)/np.sum(text_mask))-1
        #     contours, _ = cv2.findContours(text_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     epsilon = 0.007 * cv2.arcLength(contours[0], True)
        #     approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))
        #
        #     control_points = split_edge_seqence(approx, cfg.num_points)
        #     control_points = np.array(control_points[:cfg.num_points, :]).astype(np.int32)
        #
        #     cv2.drawContours(img, [ctrl_points[ist_id].astype(np.int32)], -1, (0, 255, 0), 1)
        #     cv2.drawContours(img, [control_points.astype(np.int32)], -1, (0, 0, 255), 1)
        #     for j,  pp in enumerate(control_points):
        #         if j == 0:
        #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (255, 0, 255), -1)
        #         elif j == 1:
        #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 255), -1)
        #         else:
        #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 0), -1)
        #
        #     cv2.imshow('imgs', img)
        #     cv2.waitKey(0)

