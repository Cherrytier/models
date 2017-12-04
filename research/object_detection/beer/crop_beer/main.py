from create_beer_lists import create_train_val_list
from create_beer_lists import create_file_list
from tools import ImageCropper
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare lists txt file for dataset')
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='dataset to use',
        default='data',
        type=str)
    parser.add_argument(
        '--target',
        dest='target',
        help='output list file',
        default='crop',
        type=str)
    parser.add_argument(
        '--root',
        dest='root_path',
        help='dataset root path',
        default=os.path.join(os.getcwd(), 'data', 'beer'),
        type=str)
    args = parser.parse_args()
    return args


def read_file(root):
    info = []
    file = open(root, 'rt')
    while True:
        string = file.readline()
        if not string:
            break
        info.append(string[:-1])
    return info


def _process_all(lists, output_root):
    count = 0
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    for paths in lists:
        print(paths)
        img_path, xml_path = paths.split('&!&')
        out_root = os.path.join(output_root, '{:04}'.format(count // 1000),
                                '{:08}'.format(count))
        cropper = ImageCropper(img_path, xml_path, out_root)
        cropper.update(output_root + '/break.txt')
        count += 1


def make_data(args):
    origin_data = os.path.join(args.root_path, args.dataset)
    output_data = os.path.join(args.root_path, args.target)
    create_train_val_list(origin_data, args.root_path)
    train_list = read_file(os.path.join(args.root_path, 'train_list.txt'))
    train_path = os.path.join(output_data, 'train')
    _process_all(train_list, train_path)
    create_file_list(train_path, os.path.join(args.root_path, 'train.txt'))
    val_list = read_file(os.path.join(args.root_path, 'val_list.txt'))
    val_path = os.path.join(output_data, 'val')
    _process_all(val_list, val_path)
    create_file_list(val_path, os.path.join(args.root_path, 'val.txt'))


if __name__ == '__main__':
    args = parse_args()
    make_data(args)
