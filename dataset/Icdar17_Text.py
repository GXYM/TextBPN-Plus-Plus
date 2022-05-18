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


class Mlt2017Text(TextDataset):

    def __init__(self, data_root, is_training=True, load_memory=False, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        if is_training:
            with open(os.path.join(data_root, 'train_list.txt')) as f:
                self.img_train_list = [line.strip() for line in f.readlines()]

            with open(os.path.join(data_root, 'val_list.txt')) as f:
                self.img_val_list = [line.strip() for line in f.readlines()]

            # with open(os.path.join(data_root, 'ic15_list.txt')) as f:
            #     self.img_15_list = [line.strip() for line in f.readlines()]

            if ignore_list:
                with open(ignore_list) as f:
                    ignore_list = f.readlines()
                    ignore_list = [line.strip() for line in ignore_list]
            else:
                ignore_list = []

            self.img_list = self.img_val_list + self.img_train_list #+self.img_15_list
            # self.img_list = list(filter(lambda img: img.replace('', '') not in ignore_list, self.img_list))
        else:
            with open(os.path.join(data_root, 'test_list.txt')) as f:
                self.img_list = [line.strip() for line in f.readlines()]\

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
            annotation_id = "{}/gt_{}".format("/".join(image_id.split("/")[0:-1]),
                                              image_id.split("/")[-1].replace(".jpg", ''))
            annotation_path = os.path.join(self.data_root, annotation_id)

            polygons = self.parse_txt(annotation_path)
        else:
            polygons = None

        # Read image data
        image_path = os.path.join(self.data_root, image_id)
        image = pil_load_img(image_path)
        try:
            h, w, c = image.shape
            assert (c == 3)
        except:
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

    trainset = Mlt2017Text(
        data_root='../data/MLT2017',
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
        cv2.imshow("tr_mask",
                   cav.heatmap(np.array(train_mask * 255 / np.max(train_mask), dtype=np.uint8)))

        cv2.imshow('imgs', img)
        cv2.waitKey(0)

