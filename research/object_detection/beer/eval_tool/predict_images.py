import time
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

from utils import label_map_util
from utils import visualization_utils as vis_util

from beer.crop_tools.create_lists import create_file_list
from beer.eval_tool.tools import read_img_xml_as_eval_info
from beer.eval_tool.tools import is_overlap
from beer.eval_tool.tools import get_overlap_area

IMAGE_ROOT = '/home/admins/data/beer_data'
image_lists, _ = create_file_list(os.path.join(IMAGE_ROOT, 'crop', 'val'))
OUTPUT_ROOT = os.path.join(IMAGE_ROOT, 'pre')
PATH_TO_CKPT = '/home/admins/cmake/ssd_mobilenet_v1_coco_11_06_2017/test/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/admins/cmake/models/research/object_detection/data/beer_label_map.pbtxt'
NUM_CLASSES = 9
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def eval_single(_classes, _boxes, _scores, info_):
    objects_ = info_['objects']
    _pre_objects = []
    _gt_num = len(objects_)
    _true_pre = 0
    for cls, box, score in zip(_classes, _boxes, _scores):
        if score > 0.5:
            for _ob in objects_:
                if is_overlap(box, _ob[1:]):
                    src_area = min((box[2] - box[0]) * (box[3] - box[1]), (_ob[3] - _ob[1]) * (_ob[4] - _ob[2]))
                    area = get_overlap_area(box, _ob[1:])
                    if (area / src_area > 0.8) and (cls == _ob[0]):
                        _true_pre += 1
                        _pre_objects.append([cls, *box, score])
                    break
    return _gt_num, _true_pre, _pre_objects


def write_pre_xml(_info, _pre_objects, _file_name):
    def _add_element(_root, _name, _value):
        sub_element = ET.SubElement(_root, _name)
        sub_element.text = _value

    root = ET.Element('annotation')
    size = ET.SubElement(root, 'size')
    shape = _info['shape']
    _add_element(size, 'height', str(shape[0]))
    _add_element(size, 'width', str(shape[1]))
    _add_element(size, 'depth', '3')
    origin = ET.SubElement(root, 'origin')
    objects_ = _info['objects']
    for ob in objects_:
        ob_xml = ET.SubElement(origin, 'object')
        _add_element(ob_xml, 'name', ob[0])
        _add_element(ob_xml, 'difficult', '0')
        bndbox = ET.SubElement(ob_xml, 'bndbox')
        _add_element(bndbox, 'xmin', str(int(ob[1]*shape[1])))
        _add_element(bndbox, 'ymin', str(int(ob[2]*shape[0])))
        _add_element(bndbox, 'xmax', str(int(ob[3]*shape[1])))
        _add_element(bndbox, 'ymax', str(int(ob[4]*shape[0])))

    prediction = ET.SubElement(root, 'prediction')
    for ob in _pre_objects:
        ob_xml = ET.SubElement(prediction, 'object')
        _add_element(ob_xml, 'name', ob[0])
        _add_element(ob_xml, 'difficult', '0')
        bndbox = ET.SubElement(ob_xml, 'bndbox')
        _add_element(bndbox, 'xmin', str(int(ob[1]*shape[1])))
        _add_element(bndbox, 'ymin', str(int(ob[2]*shape[0])))
        _add_element(bndbox, 'xmax', str(int(ob[3]*shape[1])))
        _add_element(bndbox, 'ymax', str(int(ob[4]*shape[0])))
    tree = ET.ElementTree(root)
    tree.write(os.path.join(OUTPUT_ROOT, _file_name))


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
        start_time = time.time()
        print(time.ctime())
        for idx, paths in enumerate(image_lists):
            print('predicting {} of {} images'.format(idx, len(image_lists)))
            img_path, xml_path = paths.split('&!&')
            image = Image.open(img_path)
            info = read_img_xml_as_eval_info(img_path, xml_path)
            image_np = np.array(image).astype(np.uint8)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={
                    image_tensor: image_np_expanded
                })
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                boxes,
                classes,
                scores,
                category_index,
                use_normalized_coordinates=True,
                line_thickness=5)
            pic = Image.fromarray(image_np)
            pic.save(os.path.join(OUTPUT_ROOT, '{}.jpg'.format(idx)))
            gt_num, true_pre, pre_objects = eval_single(boxes, classes, scores, info)
            with open(os.path.join(IMAGE_ROOT, 'gt_pre.txt', 'a')) as txt_file:
                print('{} {} {}'.format(idx, gt_num, true_pre), file=txt_file)
            write_pre_xml(info, pre_objects, '{}.xml'.format(idx))
            print('{} elapsed time: {:.3f}s'.format(time.ctime(),
                                                    time.time() - start_time))

