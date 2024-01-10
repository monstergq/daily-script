import cv2 as cv
import os, shutil
import numpy as np
from tqdm import tqdm
from utils.utils import *


def split_dataset(train_path, label, percent=0.1):

    val_path = train_path.replace('train', 'val')
    check_path(os.path.join(val_path, 'img'))
    check_path(os.path.join(val_path, 'mask', label))

    imgs_list = os.listdir(os.path.join(train_path, 'img'))
    val_imgs_list = random.sample(imgs_list, int(len(imgs_list)*percent)+1)

    for val_img in val_imgs_list:

        shutil.move(os.path.join(train_path, 'img', val_img), os.path.join(val_path, 'img'))
        shutil.move(os.path.join(train_path, 'mask', label, val_img), os.path.join(val_path, 'mask', label))


def split_IMG(label, svs_path, json_path, save_path, crop_size, level, down_sample, overlap):

    check_path(os.path.join(save_path, 'img'))
    check_path(os.path.join(save_path, 'mask', label))

    path_list = os.listdir(svs_path)
    scale = 0.5 if down_sample else 1

    for name in path_list:

        svs_name = os.path.join(svs_path, name)
        image = get_img_from_wsi(svs_name, level, np.array([None]))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        json_name = os.path.join(json_path, name.replace('.svs', '.json'))
        shapes = read_geojson(json_name)

        for shape in shapes:

            label_injson = shape['label']

            if label_injson == label:

                contours = [np.abs(np.array([shape['points'][-1]], dtype=np.uint8))]
                mask = cv.drawContours(mask, contours, -1, 255, -1)

        if scale != 1:
            mask = cv.resize(mask, dsize=None, fx=scale, fy=scale)
            image = cv.resize(image, dsize=None, fx=scale, fy=scale)

        crop_coords = []
        h, w = image.shape[:2]

        for y in range(0, h, crop_size-overlap):

            if y + crop_size > h:
                y = h - crop_size

            for x in range(0, w, crop_size-overlap):

                if x + crop_size > w:
                    x = w - crop_size

                crop_coords.append((x, y))

        for n, (x, y) in tqdm(enumerate(crop_coords)):

            if np.max(mask[y:y+crop_size, x:x+crop_size]) != 0:

                base_name = name.split('.')[0]

                save_img_path = os.path.join(save_path,  'img', f'{base_name}_{n}.png')
                cv.imwrite(save_img_path, image[y:y + crop_size, x:x + crop_size])

                save_mask_path = os.path.join(save_path, 'mask', label, f'{base_name}_{n}.png')
                cv.imwrite(save_mask_path, mask[y:y + crop_size, x:x + crop_size])

    split_dataset(save_path, label)


if __name__ == '__main__':

    svs_path = r'F:\datasets\datasets_HG\colouring matter\svs'
    json_path = r'F:\datasets\datasets_HG\colouring matter\json'
    save_path = r'F:\datasets\datasets_HG\colouring matter\patch\train'

    level = 0
    overlap = 0
    crop_size = 512
    down_sample = False
    label = '大鼠哈氏腺色素'

    split_IMG(label, svs_path, json_path, save_path, crop_size, level, down_sample, overlap)