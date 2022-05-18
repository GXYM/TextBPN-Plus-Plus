# -*- coding: utf-8 -*-
__author__ = "S.X.Zhang"
import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
from util.io import read_lines, load_json
import cv2


class ArtTextJson(TextDataset):
    def __init__(self, data_root, is_training=True, ignore_list=None, load_memory=False, transform=None):
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

        self.image_root = os.path.join(data_root, "Images", "Train" if is_training else "Test")
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace(".jpg", "") not in ignore_list, self.image_list))

        if self.is_training:
            annotation_file = os.path.join(data_root, "gt", "train_labels.json" if is_training else "None")
            self.annotation_data = load_json(annotation_file)
            self.image_list, self.annotationdata_list = self.preprocess(self.image_list, self.annotation_data)

        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img_gt(item))

    @staticmethod
    def preprocess(image_list: list, annotation_data: dict):
        """
        Decompose the all in one annotation_dict into seperate list element(annotation_list).
        The order of the annotation_list will be the same with image_list. To keep it simple,
        here both image_list and annotationdata_list will be sorted following the same criteria.
        """
        annotationdata_list = [
            v for _, v in sorted(annotation_data.items(), key=lambda item: item[0])
        ]
        image_list = sorted(image_list)
        return image_list, annotationdata_list

    def parse_curve_txt(self, gt_data):
        polygons = []
        for candidate in gt_data:
            text = candidate.get("transcription").strip().replace("###", "#")
            pts = candidate.get("points")
            pts = np.array(pts).astype(np.int32)
            if pts.shape[0] < 4:
                continue
            polygons.append(TextInstance(pts, "c", text))

        return polygons

    def load_img_gt(self, item):
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        image = pil_load_img(image_path)
        try:
            assert image.shape[-1] == 3
        except:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)

        if self.is_training:
            # Read annotation
            annotation_data = self.annotationdata_list[item]
            polygons = self.parse_curve_txt(annotation_data)
        else:
            polygons = None

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


if __name__ == "__main__":
    # execute in base dir `PYTHONPATH=.:$PYTHONPATH python dataset/Icdar19ArT_Text.py`
    from util.augmentation import Augmentation
    from util.misc import regularize_sin_cos
    from util.pbox import bbox_transfor_inv, minConnectPath
    from util import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(size=640, mean=means, std=stds)

    trainset = ArtTextJson(
        data_root="/home/prir1005/pubdata/ArT",
        is_training=True,
        transform=transform,
    )

    t0 = time.time()
    img, train_mask, tr_mask, distance_field, \
    direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags = trainset[1000]
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
