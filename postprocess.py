#!/bin/env python
# -*- coding: utf-8 -*-

"""MonFlow Postprocessor.
This class is responsible for postprocessing the incoming image"""

# Basic Imports
import os
import logging


# MonFlow Imports


class PostProcessing:
    """Class that provides a suite of postprocessing functions for handling the incoming image."""

    def __init__(self, tam_total_width=1920, tam_total_height=1080,
                 tam_slice_width=960, tam_slice_height=540,
                 margin_h=10, margin_v=10, clz_names=["pessoa", "p1", "p2", "p3"],
                 clz_colors=[[221, 44, 0], [0, 118, 212], [152, 78, 163], [102, 166, 30]]):
        # Logger
        self.logger = logging.getLogger("monflow.postprocessor")

        # Save variables
        self.tam_total_width = tam_total_width
        self.tam_total_height = tam_total_height
        self.tam_slice_width = tam_slice_width
        self.tam_slice_height = tam_slice_height
        self.margin_h = margin_h
        self.margin_v = margin_v
        self.clz_names = clz_names
        self.clz_colors = clz_colors

    def merge_boxes(self, bb1, bb2, new_id):
        xmin = 0
        xmax = 0
        ymin = 0
        ymax = 0
        conf = 0

        if bb1['ymin'] < bb2['ymin']:
            ymin = bb1['ymin']
        else:
            ymin = bb2['ymin']

        if bb1['xmin'] < bb2['xmin']:
            xmin = bb1['xmin']
        else:
            xmin = bb2['xmin']

        if bb1['xmax'] > bb2['xmax']:
            xmax = bb1['xmax']
        else:
            xmax = bb2['xmax']

        if bb1['ymax'] > bb2['ymax']:
            ymax = bb1['ymax']
        else:
            ymax = bb2['ymax']

        if bb1['conf'] > bb2['conf']:
            conf = bb1['conf']
        else:
            conf = bb2['conf']

        return {'id': new_id, 'xmin': xmin, 'ymin': ymin,
                'xmax': xmax, 'ymax': ymax, 'conf': conf}

    def analyze_q0_q1(self, bb_slice0, bb_slice1):
        q0_tam_slice_width = 0
        q0_tam_slice_height = 0

        q1_tam_slice_width = self.tam_slice_width
        q1_tam_slice_height = 0

        return self.analyze_list_boxes(bb_slice0, bb_slice1, q0_tam_slice_width, q0_tam_slice_height,
                                       q1_tam_slice_width, q1_tam_slice_height, ori="hori")

    def analyze_q0_q2(self, bb_slice0, bb_slice2):
        q0_tam_slice_width = 0
        q0_tam_slice_height = 0

        q2_tam_slice_width = 0
        q2_tam_slice_height = self.tam_slice_height

        return self.analyze_list_boxes(bb_slice0, bb_slice2, q0_tam_slice_width, q0_tam_slice_height,
                                       q2_tam_slice_width, q2_tam_slice_height, ori="vert")

    def analyze_q2_q3(self, bb_slice2, bb_slice3):
        q2_tam_slice_width = 0
        q2_tam_slice_height = self.tam_slice_height

        q3_tam_slice_width = self.tam_slice_width
        q3_tam_slice_height = self.tam_slice_height

        return self.analyze_list_boxes(bb_slice2, bb_slice3, q2_tam_slice_width, q2_tam_slice_height,
                                       q3_tam_slice_width, q3_tam_slice_height, ori="hori")

    def analyze_q1_q3(self, bb_slice1, bb_slice3):
        q1_tam_slice_width = self.tam_slice_width
        q1_tam_slice_height = 0

        q3_tam_slice_width = self.tam_slice_width
        q3_tam_slice_height = self.tam_slice_height

        return self.analyze_list_boxes(bb_slice1, bb_slice3, q1_tam_slice_width, q1_tam_slice_height,
                                       q3_tam_slice_width, q3_tam_slice_height, ori="vert")

    def analyze_list_boxes(self, bb_slice1, bb_slice2, q1_tam_slice_width, q1_tam_slice_height,
                           q2_tam_slice_width, q2_tam_slice_height, ori="vert"):
        bb_img = []
        erase_list = []

        if ori == "vert":
            for bb1 in bb_slice1:
                if bb1['ymax'] > (self.tam_slice_height - self.margin_v):
                    for bb2 in bb_slice2:
                        if bb2['id'] in erase_list:
                            continue

                        if bb2['ymin'] < self.margin_h:
                            bb1limit = list(range(int(bb1['xmin']), int(bb1['xmax'] + 1)))
                            bb2limit = list(range(int(bb2['xmin']), int(bb2['xmax'] + 1)))
                            bothintersec = set(bb1limit).intersection(bb2limit)

                            bb1intersecc = len(bothintersec) / len(bb1limit)
                            bb2intersecc = len(bothintersec) / len(bb2limit)

                            print("bb1 intersec", bb1intersecc)
                            print("bb2 intersec", bb2intersecc)

                            if (
                                    (abs(bb1['xmin'] - bb2['xmin']) < self.margin_h and
                                     abs(bb1['xmax'] - bb2['xmax']) < self.margin_h)
                                    or
                                    (bb1['xmin'] > bb2['xmin'] and bb1['xmax'] < bb2['xmax'])
                                    or
                                    (bb2['xmin'] > bb1['xmin'] and bb2['xmax'] < bb1['xmax'])
                                    or
                                    (bb1intersecc >= 0.9 or bb2intersecc >= 0.9)
                            ):
                                erase_list.append(bb1['id'])
                                erase_list.append(bb2['id'])

                                bb1_aux = {'id': 0,
                                           'xmin': bb1['xmin'] + q1_tam_slice_width,
                                           'ymin': bb1['ymin'] + q1_tam_slice_height,
                                           'xmax': bb1['xmax'] + q1_tam_slice_width,
                                           'ymax': bb1['ymax'] + q1_tam_slice_height,
                                           'conf': bb1['conf']}
                                bb2_aux = {'id': 0,
                                           'xmin': bb2['xmin'] + q2_tam_slice_width,
                                           'ymin': bb2['ymin'] + q2_tam_slice_height,
                                           'xmax': bb2['xmax'] + q2_tam_slice_width,
                                           'ymax': bb2['ymax'] + q2_tam_slice_height,
                                           'conf': bb2['conf']}

                                bb_img.append(self.merge_boxes(bb1_aux, bb2_aux, bb1['id']))

        else:
            for bb1 in bb_slice1:
                if bb1['xmax'] > (self.tam_slice_width - self.margin_h):
                    for bb2 in bb_slice2:
                        if bb2['id'] in erase_list:
                            continue

                        if bb2['xmin'] < self.margin_h:
                            bb1limit = list(range(int(bb1['ymin']), int(bb1['ymax'] + 1)))
                            bb2limit = list(range(int(bb2['ymin']), int(bb2['ymax'] + 1)))
                            bothintersec = set(bb1limit).intersection(bb2limit)

                            bb1intersecc = len(bothintersec) / len(bb1limit)
                            bb2intersecc = len(bothintersec) / len(bb2limit)

                            print("bb1 intersec", bb1intersecc)
                            print("bb2 intersec", bb2intersecc)

                            if (
                                    (abs(bb1['ymin'] - bb2['ymin']) < self.margin_v and
                                     abs(bb1['ymax'] - bb2['ymax']) < self.margin_v)
                                    or
                                    (bb1['ymin'] > bb2['ymin'] and bb1['ymax'] < bb2['ymax'])
                                    or
                                    (bb2['ymin'] > bb1['ymin'] and bb2['ymax'] < bb1['ymax'])
                                    or
                                    (bb1intersecc >= 0.9 or bb2intersecc >= 0.9)
                            ):
                                erase_list.append(bb1['id'])
                                erase_list.append(bb2['id'])

                                bb1_aux = {'id': 0,
                                           'xmin': bb1['xmin'] + q1_tam_slice_width,
                                           'ymin': bb1['ymin'] + q1_tam_slice_height,
                                           'xmax': bb1['xmax'] + q1_tam_slice_width,
                                           'ymax': bb1['ymax'] + q1_tam_slice_height,
                                           'conf': bb1['conf']}
                                bb2_aux = {'id': 0,
                                           'xmin': bb2['xmin'] + q2_tam_slice_width,
                                           'ymin': bb2['ymin'] + q2_tam_slice_height,
                                           'xmax': bb2['xmax'] + q2_tam_slice_width,
                                           'ymax': bb2['ymax'] + q2_tam_slice_height,
                                           'conf': bb2['conf']}

                                bb_img.append(self.merge_boxes(bb1_aux, bb2_aux, bb1['id']))

        return bb_img, erase_list

    def expand_bounding_box(self, bb_slice0, bb_slice1, bb_slice2, bb_slice3):
        bb_img = []

        for bb0 in bb_slice0:
            bb0["clz"] = 0
            bb_img.append(bb0)

        for bb1 in bb_slice1:
            bb1 = {'id': bb1['id'], 'xmin': bb1['xmin'] + self.tam_slice_width, 'ymin': bb1['ymin'],
                   'xmax': bb1['xmax'] + self.tam_slice_width, 'ymax': bb1['ymax'], 'conf': bb1['conf'], 'clz': 1}
            bb_img.append(bb1)

        for bb2 in bb_slice2:
            bb2 = {'id': bb2['id'], 'xmin': bb2['xmin'], 'ymin': bb2['ymin'] + self.tam_slice_height,
                   'xmax': bb2['xmax'], 'ymax': bb2['ymax'] + self.tam_slice_height, 'conf': bb2['conf'], 'clz': 2}
            bb_img.append(bb2)

        for bb3 in bb_slice3:
            bb3 = {'id': bb3['id'], 'xmin': bb3['xmin'] + self.tam_slice_width,
                   'ymin': bb3['ymin'] + self.tam_slice_height,
                   'xmax': bb3['xmax'] + self.tam_slice_width, 'ymax': bb3['ymax'] + self.tam_slice_height,
                   'conf': bb3['conf'], 'clz': 3}
            bb_img.append(bb3)

        return bb_img

    def plot_bounding_box(self, img, bb_img, clz=0):
        for bb in bb_img:
            xy = [bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]]
            id = bb["id"]
            conf = bb["conf"]

            label = '%d %s %.2f' % (id, self.clz_names[int(clz)], conf)
            color = self.clz_colors[int(clz)]
            plot_one_box(xy, img, label=label, color=color)

        return img

    def fix_bounding_box(self, bb_slice0, bb_slice1, bb_slice2, bb_slice3):
        bb_img = []
        erase_list = []

        # Merge entre Q0 e Q1
        bb_img_aux, erase_list_aux = self.analyze_q0_q1(bb_slice0, bb_slice1)
        bb_img += bb_img_aux
        erase_list += erase_list_aux

        # Merge entre Q0 e Q2
        bb_img_aux, erase_list_aux = self.analyze_q0_q2(bb_slice0, bb_slice2)
        bb_img += bb_img_aux
        erase_list += erase_list_aux

        # Merge entre Q1 e Q3
        bb_img_aux, erase_list_aux = self.analyze_q1_q3(bb_slice1, bb_slice3)
        bb_img += bb_img_aux
        erase_list += erase_list_aux

        # Merge entre Q2 e Q3
        bb_img_aux, erase_list_aux = self.analyze_q2_q3(bb_slice2, bb_slice3)
        bb_img += bb_img_aux
        erase_list += erase_list_aux

        for bb0 in bb_slice0:
            if bb0['id'] in erase_list:
                continue
            bb_img.append(bb0)

        for bb1 in bb_slice1:
            if bb1['id'] in erase_list:
                continue
            bb1 = {'id': bb1['id'], 'xmin': bb1['xmin'] + self.tam_slice_width, 'ymin': bb1['ymin'],
                   'xmax': bb1['xmax'] + self.tam_slice_width, 'ymax': bb1['ymax'], 'conf': bb1['conf']}
            bb_img.append(bb1)

        for bb2 in bb_slice2:
            if bb2['id'] in erase_list:
                continue
            bb2 = {'id': bb2['id'], 'xmin': bb2['xmin'], 'ymin': bb2['ymin'] + self.tam_slice_height,
                   'xmax': bb2['xmax'], 'ymax': bb2['ymax'] + self.tam_slice_height, 'conf': bb2['conf']}
            bb_img.append(bb2)

        for bb3 in bb_slice3:
            if bb3['id'] in erase_list:
                continue
            bb3 = {'id': bb3['id'], 'xmin': bb3['xmin'] + self.tam_slice_width,
                   'ymin': bb3['ymin'] + self.tam_slice_height,
                   'xmax': bb3['xmax'] + self.tam_slice_width, 'ymax': bb3['ymax'] + self.tam_slice_height,
                   'conf': bb3['conf']}
            bb_img.append(bb3)

        return bb_img
