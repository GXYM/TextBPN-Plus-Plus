# -*- coding: utf-8 -*-
__author__ = "S.X.Zhang"
import copy
import cv2
import torch
import numpy as np
from PIL import Image
from scipy import ndimage as ndimg
from cfglib.config import config as cfg
from util.misc import find_bottom, find_long_edges, split_edge_seqence, \
    vector_sin, get_sample_point


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance(object):
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None
        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        '''
        remove_points = []
        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area)/ori_area < 0.0017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)
        '''
        self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def get_sample_point(self, size=None):
        mask = np.zeros(size, np.uint8)
        cv2.fillPoly(mask, [self.points.astype(np.int32)], color=(1,))
        control_points = get_sample_point(mask, cfg.num_points, cfg.approx_factor)

        return control_points

    def get_control_points(self, size=None):
        n_disk = cfg.num_control_points // 2 - 1
        sideline1 = split_edge_seqence(self.points, self.e1, n_disk)
        sideline2 = split_edge_seqence(self.points, self.e2, n_disk)[::-1]
        if sideline1[0][0] > sideline1[-1][0]:
            sideline1 = sideline1[::-1]
            sideline2 = sideline2[::-1]
        p1 = np.mean(sideline1, axis=0)
        p2 = np.mean(sideline2, axis=0)
        vpp = vector_sin(p1 - p2)
        if vpp >= 0:
            top = sideline2
            bot = sideline1
        else:
            top = sideline1
            bot = sideline2

        control_points = np.concatenate([np.array(top), np.array(bot[::-1])], axis=0).astype(np.float32)

        return control_points

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(object):

    def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.min_text_size = 4
        self.jitter = 0.7
        self.th_b = 0.4

    @staticmethod
    def sigmoid_alpha(x, k):
        betak = (1 + np.exp(-k)) / (1 - np.exp(-k))
        dm = max(np.max(x), 0.0001)
        res = (2 / (1 + np.exp(-x * k / dm)) - 1) * betak
        return np.maximum(0, res)

    @staticmethod
    def fill_polygon(mask, pts, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param pts: polygon to draw
        :param value: fill value
        """
        # cv2.drawContours(mask, [polygon.astype(np.int32)], -1, value, -1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], color=(value,))
        # rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(mask.shape[0],mask.shape[1]))
        # mask[rr, cc] = value

    @staticmethod
    def generate_proposal_point(text_mask, num_points, approx_factor, jitter=0.0, distance=10.0):
        # get  proposal point in contours
        h, w = text_mask.shape[0:2]
        contours, _ = cv2.findContours(text_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        epsilon = approx_factor * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))
        ctrl_points = split_edge_seqence(approx, num_points)

        ctrl_points = np.array(ctrl_points[:num_points, :]).astype(np.int32)

        if jitter > 0:
            x_offset = (np.random.rand(ctrl_points.shape[0]) - 0.5) * distance*jitter
            y_offset = (np.random.rand(ctrl_points.shape[0]) - 0.5) * distance*jitter
            ctrl_points[:, 0] += x_offset.astype(np.int32)
            ctrl_points[:, 1] += y_offset.astype(np.int32)
        ctrl_points[:, 0] = np.clip(ctrl_points[:, 0], 1, w - 2)
        ctrl_points[:, 1] = np.clip(ctrl_points[:, 1], 1, h - 2)
        return ctrl_points

    @staticmethod
    def compute_direction_field(inst_mask, h, w):
        _, labels = cv2.distanceTransformWithLabels(inst_mask, cv2.DIST_L2,
                                                    cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
        # # compute the direction field
        index = np.copy(labels)
        index[inst_mask > 0] = 0
        place = np.argwhere(index > 0)
        nearCord = place[labels - 1, :]
        x = nearCord[:, :, 0]
        y = nearCord[:, :, 1]
        nearPixel = np.zeros((2, h, w))
        nearPixel[0, :, :] = x
        nearPixel[1, :, :] = y
        grid = np.indices(inst_mask.shape)
        grid = grid.astype(float)
        diff = nearPixel - grid

        return diff

    def make_text_region(self, img, polygons):
        h, w = img.shape[0], img.shape[1]
        mask_zeros = np.zeros(img.shape[:2], np.uint8)

        train_mask = np.ones((h, w), np.uint8)
        tr_mask = np.zeros((h, w), np.uint8)
        weight_matrix = np.zeros((h, w), dtype=np.float)
        direction_field = np.zeros((2, h, w), dtype=np.float)
        distance_field = np.zeros((h, w), np.float)

        gt_points = np.zeros((cfg.max_annotation, cfg.num_points, 2), dtype=np.float)
        proposal_points = np.zeros((cfg.max_annotation, cfg.num_points, 2), dtype=np.float)
        ignore_tags = np.zeros((cfg.max_annotation,), dtype=np.int)

        if polygons is None:
            return train_mask, tr_mask, \
                   distance_field, direction_field, \
                   weight_matrix, gt_points, proposal_points, ignore_tags

        for idx, polygon in enumerate(polygons):
            if idx >= cfg.max_annotation:
                break
            polygon.points[:, 0] = np.clip(polygon.points[:, 0], 1, w - 2)
            polygon.points[:, 1] = np.clip(polygon.points[:, 1], 1, h - 2)
            gt_points[idx, :, :] = polygon.get_sample_point(size=(h, w))
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int)], color=(idx + 1,))

            inst_mask = mask_zeros.copy()
            cv2.fillPoly(inst_mask, [polygon.points.astype(np.int32)], color=(1,))
            dmp = ndimg.distance_transform_edt(inst_mask)  # distance transform

            if polygon.text == '#' or np.max(dmp) < self.min_text_size:
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))
                ignore_tags[idx] = -1
            else:
                ignore_tags[idx] = 1

            proposal_points[idx, :, :] = \
                self.generate_proposal_point(dmp / (np.max(dmp)+1e-3) >= self.th_b, cfg.num_points,
                                             cfg.approx_factor, jitter=self.jitter, distance=self.th_b * np.max(dmp))

            distance_field[:, :] = np.maximum(distance_field[:, :], dmp / (np.max(dmp)+1e-3))

            weight_matrix[inst_mask > 0] = 1. / np.sqrt(inst_mask.sum())
            diff = self.compute_direction_field(inst_mask, h, w)
            direction_field[:, inst_mask > 0] = diff[:, inst_mask > 0]

        # ### background ######
        weight_matrix[tr_mask == 0] = 1. / np.sqrt(np.sum(tr_mask == 0))
        # diff = self.compute_direction_field((tr_mask == 0).astype(np.uint8), h, w)
        # direction_field[:, tr_mask == 0] = diff[:, tr_mask == 0]

        train_mask = np.clip(train_mask, 0, 1)

        return train_mask, tr_mask, \
               distance_field, direction_field, \
               weight_matrix, gt_points, proposal_points, ignore_tags

    def get_training_data(self, image, polygons, image_id=None, image_path=None):
        np.random.seed()
        if self.transform:
            #image, polygons = self.transform(image, polygons)
            image, polygons = self.transform(copy.deepcopy(image), copy.deepcopy(polygons))

        train_mask, tr_mask, \
        distance_field, direction_field, \
        weight_matrix, gt_points, proposal_points, ignore_tags = self.make_text_region(image, polygons)

        # # to pytorch channel sequence
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        train_mask = torch.from_numpy(train_mask).bool()
        tr_mask = torch.from_numpy(tr_mask).int()
        weight_matrix = torch.from_numpy(weight_matrix).float()
        direction_field = torch.from_numpy(direction_field).float()
        distance_field = torch.from_numpy(distance_field).float()
        gt_points = torch.from_numpy(gt_points).float()
        proposal_points = torch.from_numpy(proposal_points).float()
        ignore_tags = torch.from_numpy(ignore_tags).int()

        return image, train_mask, tr_mask, distance_field, \
               direction_field, weight_matrix, gt_points, proposal_points, ignore_tags

    def get_test_data(self, image, polygons=None, image_id=None, image_path=None):
        H, W, _ = image.shape
        if self.transform:
            image, polygons = self.transform(image, polygons)

        # max point per polygon for annotation
        points = np.zeros((cfg.max_annotation, 20, 2))
        length = np.zeros(cfg.max_annotation, dtype=int)
        label_tag = np.zeros(cfg.max_annotation, dtype=int)
        if polygons is not None:
            for i, polygon in enumerate(polygons):
                pts = polygon.points
                points[i, :pts.shape[0]] = polygon.points
                length[i] = pts.shape[0]
                if polygon.text != '#':
                    label_tag[i] = 1
                else:
                    label_tag[i] = -1

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'annotation': points,
            'n_annotation': length,
            'label_tag': label_tag,
            'Height': H,
            'Width': W
        }

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        return image, meta

    def __len__(self):
        raise NotImplementedError()
