# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torch.nn import functional as F

from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import json

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.boxes import BoxList
from maskrcnn_benchmark.structures.masks import SegmentationMask
from maskrcnn_benchmark.utils.visualization.create_palette import create_palette
from maskrcnn_benchmark.utils.visualization.cv2_util import findContours
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.visualization.colormap import random_color
from maskrcnn_benchmark.data.datasets.coco import COCODataset

_LARGE_MASK_AREA_THRESH = 120000

CATEGORIES = [
    'real',
    'fake',
]


class Visualizer(object):

    def __init__(
        self,
        cfg=None,
        confidence_threshold=-1,
        max_confidence=1,
        show_mask_heatmaps=False,
        show_mask_montage=True,
        masks_per_dim=2,
        categories=None,
        show_text=True,
        show_bbox=True,
        show_mask=True,
        show_contour=True,
        display_text_inside_object=False,
        show_only_label=False,
        shift_index=None,
    ):
        self.categories = categories
        if cfg is not None:
            self.cfg = cfg.clone()
        else:
            self.cfg = None

        mask_threshold = -1 if show_mask_montage else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        if confidence_threshold >= 0 or self.cfg is None:
            self.confidence_threshold = confidence_threshold
        else:
            self.confidence_threshold = self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_VISUALIZATION
        self.max_confidence = max_confidence

        self.show_heatmap = show_mask_heatmaps
        self.show_text = show_text
        self.show_bbox = show_bbox
        self.show_mask = show_mask
        self.show_contour = show_contour
        self.display_text_inside_object = display_text_inside_object
        self.show_only_label = show_only_label
        self.show_mask_montage = show_mask_montage
        self.masks_per_dim = masks_per_dim
        self.shift_index = shift_index

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.my_palette = []
        colors = create_palette()
        self.my_palette = []
        for color in colors:
            self.my_palette.append(color['rgb'])
        if self.categories is not None:
            if len(self.categories) > len(self.my_palette):
                for i in range(len(self.categories)):
                    color = random_color()
                    color = (int(color[0]), int(color[1]), int(color[2]))
                    self.my_palette.append(color)

    def visualize_predictions(self, predictions, dataset, output_folder, mask_alpha=0.5, text_alpha=0.5,
                              use_random_color_mask=False, use_white_color_contour=True, text_color=(255, 255, 255)):
        mkdir(output_folder)

        root = dataset.root
        for image_id, prediction in enumerate(tqdm(predictions)):
            original_id = dataset.id_to_img_map[image_id]
            image_width = dataset.coco.imgs[original_id]["width"]
            image_height = dataset.coco.imgs[original_id]["height"]
            file_name = dataset.coco.imgs[original_id]["file_name"]
            if not os.path.exists(os.path.join(root, file_name)):
                print(os.path.join(root, file_name), 'not exist')
                continue
            pil_image = Image.open(os.path.join(root, file_name)).convert("RGB")
            image = np.array(pil_image)[:, :, [2, 1, 0]] # convert to BGR format

            if prediction is None or len(prediction) == 0:
                dir_img_output = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.jpg')
                dir_folder_output = os.path.dirname(dir_img_output)
                mkdir(dir_folder_output)
                cv2.imwrite(dir_img_output, image)
                print(os.path.join(root, file_name), 'no GT')
                continue

            prediction = prediction.resize((image_width, image_height))

            # paste the masks in the right position in the image, as defined by the bounding boxes
            if prediction.has_field("mask"):
                masks = prediction.get_field("mask")
                masks = self.masker([masks], [prediction])[0]
                prediction.add_field("mask", masks)

            main_name_property = "labels"
            main_score_property = "scores"

            if main_score_property:
                try:
                    prediction = self._select_top_predictions(prediction, main_score_property)
                except:
                    pass

            result = image.copy()
            if prediction.has_field("mask"):
                if self.show_mask:
                    result = self.overlay_masks(result, prediction, name_property=main_name_property,
                                                mask_property="mask", alpha=mask_alpha, use_random_color=use_random_color_mask)
                if self.show_contour:
                    result = self.overlay_contours(result, prediction, name_property=main_name_property,
                                                   mask_property="mask", use_white_color=use_white_color_contour, use_random_color=use_random_color_mask)
            if self.show_bbox:
                result = self.overlay_boxes(result, prediction, name_property=main_name_property)
            if self.show_text:
                result = self.overlay_class_names(result, prediction, name_property=main_name_property,
                                                  score_property=main_score_property, alpha=text_alpha, text_color=text_color)

            dir_img_output = os.path.join(output_folder, os.path.dirname(file_name),
                                          os.path.splitext(os.path.basename(file_name))[0] + '.jpg')
            dir_folder_output = os.path.dirname(dir_img_output)
            mkdir(dir_folder_output)
            # result = cv2.resize(result, (512, 340))
            cv2.imwrite(dir_img_output, result)

    def _select_top_predictions(self, predictions, score_property="scores"):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field(score_property)
        keep1 = scores >= self.confidence_threshold
        keep2 = scores <= self.max_confidence
        keep = torch.nonzero(keep1 * keep2).squeeze(1)
        # print(scores, keep)
        predictions = predictions[keep]
        scores = predictions.get_field(score_property)
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def _compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        if self.my_palette is not None:
            colors = []
            for l in labels:
                if self.shift_index is not None and int(l) in self.shift_index:
                    l = self.shift_index[int(l)]
                colors.append(self.my_palette[l])
            colors = np.asarray(colors).astype("uint8")
        else:
            colors = labels[:, None] * self.palette
            colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions, name_property="labels"):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field(name_property)
        boxes = predictions.bbox

        colors = self._compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 5)

        return image

    def overlay_class_names(self, image, predictions, name_property="labels", score_property="scores", alpha=0.5, text_color=(255, 255, 255)):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        if predictions.has_field(score_property):
            scores = predictions.get_field(score_property).tolist()
        else:
            scores = None
        labels = predictions.get_field(name_property).tolist()
        labels = [self.categories[i] for i in labels]
        colors = self._compute_colors_for_labels(predictions.get_field(name_property)).tolist()
        boxes = predictions.bbox

        if self.display_text_inside_object and predictions.has_field("mask"):
            masks = predictions.get_field("mask").numpy()
            for i, (box, mask, label, color) in enumerate(zip(boxes, masks, labels, colors)):
                mask = mask[0, :, :]
                _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
                try:
                    largest_component_id = np.argmax(stats[1:, -1]) + 1
                except:
                    continue
                # draw text on the largest component, as well as other very large components.
                for cid in range(1, _num_cc):
                    if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
                        # median is more stable than centroid
                        # center = centroids[largest_component_id]
                        center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                        x = center[0] - 30
                        y = center[1]
                        if scores is not None and not self.show_only_label:
                            template = "{}: {:.2f}"
                            s = template.format(label, scores[i])
                        else:
                            template = "{}"
                            s = template.format(label)
                        cv2.putText(image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        else:
            for i, (box, label, color) in enumerate(zip(boxes, labels, colors)):
                box = box.to(torch.int64)
                top_left = box[:2].tolist()
                bottom_right = box[2:].tolist()
                bottom_right[1] = top_left[1] - 20

                image2 = cv2.rectangle(image.copy(), tuple(top_left), tuple(bottom_right), tuple(color), -1)
                image = cv2.addWeighted(image2, alpha, image, 1-alpha, 0)

                x = (top_left[0] + bottom_right[0]) / 2 - 50
                y = (top_left[1]) - 4
                if scores is not None and not self.show_only_label:
                    template = "{}: {:.2f}"
                    s = template.format(label, scores[i])
                else:
                    template = "{}"
                    s = template.format(label)
                cv2.putText(image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        return image

    def overlay_contours(self, image, predictions, name_property="labels", mask_property="mask", use_white_color=False,
                         contour_bold=5, use_random_color=False):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field(mask_property).numpy()
        labels = predictions.get_field(name_property)
        colors = self._compute_colors_for_labels(labels).tolist()
        for mask, color, label in zip(masks, colors, labels):
            thresh = mask[0, :, :, None]
            contours, hierarchy = findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if use_white_color:
                color = (255, 255, 255)
            elif use_random_color:
                color = random_color()
                color = (int(color[0]), int(color[1]), int(color[2]))
            image = cv2.drawContours(image, contours, -1, color, contour_bold)
        composite = image
        return composite

    def overlay_masks(self, image, predictions, name_property="labels", mask_property="mask", alpha=0.5, use_random_color=False):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field(mask_property).numpy()
        if not use_random_color:
            labels = predictions.get_field(name_property)
            colors = self._compute_colors_for_labels(labels).tolist()

        for i, (mask, label) in enumerate(zip(masks, labels)):
            if use_random_color:
                color = random_color()
                color = (int(color[0]), int(color[1]), int(color[2]))
            else:
                color = colors[i]

            thresh = mask[0, :, :, None]
            contours, hierarchy = findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image2 = cv2.drawContours(image.copy(), contours, -1, color, -1)
            image = cv2.addWeighted(image2, alpha, image, 1 - alpha, 0)
        return image


def prediction_to_boxes(prediction, dataset, mode='poly'):
    def get_key(my_dict, val):
        for key, value in my_dict.items():
            if val == value:
                return key
        return None

    id_to_img_map = dataset.id_to_img_map
    pred_dict = {}
    for pred in prediction:
        if pred['image_id'] not in pred_dict:
            pred_dict[pred['image_id']] = []
        pred_dict[pred['image_id']].append(pred)

    boxes = [None] * len(id_to_img_map)
    for image_id in sorted(pred_dict.keys()):
        pred = pred_dict[image_id]
        id = get_key(id_to_img_map, image_id)
        if id is not None:
            info = {}
            for i, x in enumerate(pred):
                for k, v in x.items():
                    if k not in info:
                        info[k] = []
                    info[k].append(v)
            img = dataset.coco.imgs[image_id]
            if 'bbox' not in info:
                info['bbox'] = [[0, 0, img['width'], img['height']]] * len(info['segmentation'])
            info['bbox'] = np.array(info['bbox'])
            boxes[id] = BoxList(info['bbox'], (img['width'], img['height']), mode="xywh").convert('xyxy')

            if "segmentation" in info and len(info['segmentation']):
                mask = []
                for i, data in enumerate(info['segmentation']):
                    if type(data) != list or type(data) != tuple:
                        data = [data]
                    data = SegmentationMask(data, (img['width'], img['height']), mode=mode)
                    data = data.get_mask_tensor()
                    box = info['bbox'][i, :]
                    box = [max(int(x), 0) for x in box]
                    data = data[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]
                    data = data.unsqueeze(0).unsqueeze(0).type(torch.float)
                    data = F.interpolate(data, size=500)
                    mask.append(data)
                mask = torch.cat(mask, dim=0)
                boxes[id].add_field("mask", mask)
            if "category_id" in info:
                data = torch.tensor(np.array(info['category_id']))
                boxes[id].add_field("labels", data)
            if "score" in info:
                data = torch.tensor(np.array(info['score']))
                boxes[id].add_field("scores", data)
    return boxes


def load_json(file=''):
    info = json.load(open(file))
    return info


def visualize_gt(ann_file, output_dir, categories, root=''):
    dataset = COCODataset(ann_file=ann_file,
                          root=root,
                          remove_images_without_annotations=False, transforms=None)
    visualizer = Visualizer(categories=categories, cfg=None,
                            show_text=False,
                            show_bbox=True,
                            show_mask=True,
                            show_contour=True)
    os.makedirs(output_dir, exist_ok=True)
    predictions = load_json(ann_file)['annotations']
    predictions = prediction_to_boxes(predictions, dataset, 'poly')
    visualizer.visualize_predictions(predictions, dataset, output_dir,
                                     mask_alpha=0.5, text_alpha=0.5,
                                     use_random_color_mask=False, use_white_color_contour=False,
                                     text_color=(255, 255, 255))


if __name__ == "__main__":
    dataset_dir = '/OpenForensics/Release/1.0.0'
    visualize_gt(
        ann_file='/OpenForensics/Release/1.0.0/Val_poly.json',
        output_dir='/OpenForensics/Release/1.0.0/Visualization/',
        categories=CATEGORIES,
        root=dataset_dir)

