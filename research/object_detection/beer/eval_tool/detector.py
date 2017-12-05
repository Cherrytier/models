import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

from utils import label_map_util
from utils import visualization_utils as vis_util

from beer.eval_tool.tools import is_overlap
from beer.eval_tool.tools import get_overlap_area


class BeerDetector(object):
    """
    a detection tool class
    """

    def __init__(self, pd_file, use_gpu=True):
        assert isinstance(pd_file, str)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(pd_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = use_gpu
        self.src_img_info = None
        self.images = None
        self.objects = []

    def _process(self, boxes, scores, classes, index):
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        src_h, src_w = self.src_img_info['shape']
        h, w = self.src_img_info['crop_shape']
        idx_x, idx_y = int(index.split('_')[0]), int(index.split('_')[1])
        for box, score, _class in zip(boxes, scores, classes):
            xmin = (round(box[0] * w) + idx_x) / src_w
            ymin = (round(box[1] * h) + idx_y) / src_h
            xmax = (round(box[2] * w) + idx_x) / src_w
            ymax = (round(box[3] * h) + idx_y) / src_h
            if (xmax >= 1.0) or (ymax >= 1.0):
                continue
            self.objects.append([_class, xmin, ymin, xmax, ymax, score])

    def _merge(self):
        objects = np.array(self.objects)
        self.objects = list(objects[np.lexsort(objects.T)])[:]
        filtered = []
        for ob in self.objects:
            is_add = True
            for _ob in filtered:
                if is_overlap(ob[1:], _ob[1:]):
                    src_area = min((ob[3] - ob[1]) * (ob[4] - ob[2]), (_ob[3] - _ob[1]) * (_ob[4] - _ob[2]))
                    area = get_overlap_area(ob[1:], _ob[1:])
                    if area / src_area > 0.75:
                        is_add = False
                        break
                    elif area / src_area < 0.25:
                        is_add = True
                    else:
                        is_add = False
                        break
            if is_add:
                filtered.append(ob)
        self.objects.clear()
        self.objects = filtered[:]

    def _eval(self):
        objects = self.src_img_info['objects']
        for ob in self.objects:
            for _ob in objects:
                if is_overlap(ob[1:], _ob[1:]):
                    src_area = min((ob[3] - ob[1]) * (ob[4] - ob[2]), (_ob[3] - _ob[1]) * (_ob[4] - _ob[2]))
                    area = get_overlap_area(ob[1:], _ob[1:])
                    ob.append((area / src_area > 0.8) and (ob[0] == _ob[0]))
                    break
            if len(ob) == 6:
                ob.append(False)

    def visualize(self, labels, output_img='', is_show=False):
        if isinstance(labels, str):
            label_map = label_map_util.load_labelmap(labels)
            categories = label_map_util.convert_label_map_to_categories(
                label_map, max_num_classes=9, use_display_name=True)
            category_index = label_map_util.create_category_index(categories)
        else:
            category_index = labels
        src_objects = self.src_img_info['objects']
        src_image = self.src_img_info['image']
        image = src_image[:]
        classes = []
        scores = []
        boxes = []
        for ob in src_objects:
            classes.append(ob[0])
            boxes.append(ob[1:])
            scores.append(1)
        vis_util.visualize_boxes_and_labels_on_image_array(
            src_image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.subplots(121)
        plt.imshow(src_image)
        plt.title('ground true')
        classes.clear()
        boxes.clear()
        scores.clear()
        for ob in self.objects:
            classes.append(ob[0])
            boxes.append(ob[1:-2])
            scores.append(ob[-2])
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.subplots(122)
        plt.imshow(image)
        plt.title('prediction')
        if output_img != '':
            plt.savefig(output_img)
        if is_show:
            plt.show()
        plt.close()

    def _detect(self, image_list):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph, config=self.config) as sess:
                parameters = [self.detection_graph.get_tensor_by_name('detection_boxes:0'),
                              self.detection_graph.get_tensor_by_name('detection_scores:0'),
                              self.detection_graph.get_tensor_by_name('detection_classes:0'),
                              self.detection_graph.get_tensor_by_name('num_detections:0')]
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                for idx, image_ in enumerate(image_list):
                    print('detecting {} of {}'.format(idx, len(image_list)))
                    image = Image.open(image_)
                    image_np = np.array(image).astype(np.uint8)
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    boxes, scores, classes, _ = sess.run(parameters,
                                                         feed_dict={image_tensor: image_np_expanded})
                    self._process(boxes, scores, classes, os.path.basename(image_).split('.')[0])

    def detect_images(self, images, src_img_info):
        self.src_img_info = src_img_info
        self.images = images
        self.objects.clear()
        if isinstance(self.images, np.ndarray):
            self._detect([self.images])
        elif isinstance(self.images, list):
            self._detect(self.images)
        else:
            raise TypeError('please input correct image !')
        self._merge()
        self._eval()

    def get_result(self):
        return self.objects
