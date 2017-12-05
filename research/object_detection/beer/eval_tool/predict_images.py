import time
import os
import tensorflow as tf
import numpy as np
from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util

from beer.crop_tools.create_lists import create_file_list

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
            image = Image.open(paths.split('&!&')[0])
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
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=5)
            pic = Image.fromarray(image_np)
            pic.save(os.path.join(OUTPUT_ROOT, '{}.jpg'.format(idx)))
            print('{} elapsed time: {:.3f}s'.format(time.ctime(),
                                                    time.time() - start_time))
