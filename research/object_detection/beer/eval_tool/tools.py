import cv2
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import xml.etree.ElementTree as ET


def read_img_xml_as_eval_info(img_path, xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    class_names = [
        'Budweiser 600ML Bottle', 'Harbin Wheat 330ML Can',
        'budweiser15', 'Budweiser Beer 500ML Can', 'harbin26',
        'budweiser26', 'Budweiser Beer 330ML Can', 'budweiser31',
        'budweiser30'
    ]
    info = {}
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    info['shape'] = (height, width)
    objects = []
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        if cls_name not in class_names:
            continue
        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text) / width
        ymin = int(xml_box.find('ymin').text) / height
        xmax = int(xml_box.find('xmax').text) / width
        ymax = int(xml_box.find('ymax').text) / height
        objects.append([class_names.index(cls_name), xmin, ymin, xmax, ymax])
    info['objects'] = objects
    info['image'] = cv2.imread(img_path)
    return info


def is_overlap(rect1, rect2):
    return (rect1[0] >= rect2[2]) or (rect1[1] >= rect2[3]) or (rect1[2] <= rect2[0]) or (rect1[3] <= rect2[1])


def get_overlap_area(rect1, rect2):
    xmin = min(rect1[0], rect2[0])
    ymin = min(rect1[1], rect2[1])
    xmax = min(rect1[2], rect2[2])
    ymax = min(rect1[3], rect2[3])
    return (xmax - xmin) * (ymax - ymin)


def get_recalls_and_precisions(predictions):
    ground_true = np.ones((len(predictions),))
    recalls = []
    precisions = []
    for __i in range(1, len(predictions) + 1):
        recalls.append(recall_score(ground_true[:__i], predictions[:__i]))
        precisions.append(accuracy_score(ground_true[:__i], predictions[:__i]))
    return recalls, precisions


def compute_average_precision(recalls, precisions, k):
    ap = 0.0
    for __i in range(1, k + 1):
        ap += precisions[__i] * (recalls[__i] - recalls[__i - 1])
    return ap


def compute_mean_average_precision(predictions, top_k=0):
    predictions_ = np.array(predictions)
    predictions_ = predictions_[np.lexsort(predictions_[:, ::-1].T)][:]
    end = predictions_.shape[0]
    if top_k > 0:
        assert top_k < predictions_.shape[0], 'top_k is larger than predictions !'
        end = top_k
    precision, recall = get_recalls_and_precisions(predictions_)
    aps = []
    for __i in range(1, end + 1):
        aps.append(compute_average_precision(precision, recall, __i))
    return np.mean(aps)
