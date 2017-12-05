import os

from beer.eval_tool.tools import read_img_xml_as_eval_info
from beer.eval_tool.detector import BeerDetector
from beer.crop_tools.create_lists import create_file_list
from beer.crop_tools.tools import ImageCropper


def process_all(lists, output_root, pd_file):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    detector = BeerDetector(pd_file)
    objects = []
    for count, paths in enumerate(lists):
        print(paths)
        img_path, xml_path = paths.split('&!&')
        info = read_img_xml_as_eval_info(img_path, xml_path)
        info['crop_shape'] = (416, 416)
        out_root = os.path.join(output_root, '{:04}'.format(count // 1000),
                                '{:08}'.format(count))
        out_file_root = os.path.join(output_root, '{:04}'.format(count // 1000))
        cropper = ImageCropper(img_path, xml_path, out_root)
        cropper.update(output_root + '/break.txt')
        images, _ = create_file_list(out_root, out_file_root + '/img.txt')
        img_list = list(map(lambda x: x.split('&!&')[0], images))
        detector.detect_images(img_list, info)
        objects += detector.get_result()
        detector.visualize()
    return objects


