import xml.etree.ElementTree as ET


def read_xml_as_eval_info(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        if cls_name not in [
            'Budweiser 600ML Bottle', 'Harbin Wheat 330ML Can',
            'budweiser15', 'Budweiser Beer 500ML Can', 'harbin26',
            'budweiser26', 'Budweiser Beer 330ML Can', 'budweiser31',
            'budweiser30'
        ]:
            continue
        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text)
        ymin = int(xml_box.find('ymin').text)
        xmax = int(xml_box.find('xmax').text)
        ymax = int(xml_box.find('ymax').text)
        objects.append([cls_name, xmin, ymin, xmax, ymax, width, height])

    return objects


def is_overlap(rect1, rect2):
    return (rect1[0] >= rect2[2]) or (rect1[1] >= rect2[3]) or (rect1[2] <= rect2[0]) or (rect1[3] <= rect2[1])


def get_overlap_area(rect1, rect2):
    xmin = min(rect1[0], rect2[0])
    ymin = min(rect1[1], rect2[1])
    xmax = min(rect1[2], rect2[2])
    ymax = min(rect1[3], rect2[3])
    return (xmax - xmin) * (ymax - ymin)
