import cv2 as cv
import os, shutil
import numpy as np
from tqdm import tqdm
from utils.utils import *


class generate_masks:

    def __init__(self, svs_path, json_path, save_path, labels, draw, down_sample, level=0):

        self.draw = draw
        self.level = level
        self.labels = labels
        self.svs_path = svs_path
        self.json_path = json_path
        self.down_sample = down_sample
        self.base_name = os.path.basename(svs_path)[:-4]

        self.img_save_path = os.path.join(save_path, 'img_roi')
        self.mask_save_path = os.path.join(save_path, 'mask')

        if not os.path.exists(self.img_save_path):

            os.makedirs(self.img_save_path)
            os.makedirs(self.mask_save_path)

            for label in labels.keys():
                os.makedirs(os.path.join(self.mask_save_path, label))

    def get_roi_mask(self, roiLabels, colors):

        shapes = read_geojson(self.json_path)
        roi_areas = get_roi_conts(shapes, roi_labels=roiLabels)

        for i, roi_area in enumerate(roi_areas):

            img, min_x, min_y = get_img_from_wsi(self.svs_path, self.level, roi_area)
            masks = generate_masks_list(self.labels, img.shape[:2])

            for shape in shapes:

                label = shape['label']
                contours = [np.abs(np.array([shape['points'][-1]], dtype=np.int32)) - (min_x, min_y)]

                if label in masks.keys():
                    masks[label] = cv.drawContours(masks[label], contours, -1, 255, -1)

                for label_name in self.labels.keys():

                    if label == label_name:
                        masks[label_name] = cv.drawContours(masks[label_name], contours, -1, 255, -1)
            
            for label_name in self.labels.keys():
                
                if self.labels[label_name]:
                    masks[label_name] = erode(masks[label_name])
                    
            if self.down_sample:
                img = cv.resize(img, dsize=None, fx=1/2, fy=1/2)
                masks = {label_name: cv.resize(masks[label_name], dsize=None, fx=1/2, fy=1/2) for label_name in masks.keys()}

            if self.draw:

                if not i:

                    for j, label_name in enumerate(masks.keys()):
                        
                        Contours = cv.findContours(masks[label_name], cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[-2]
                        cv.drawContours(img, Contours, -1, colors[j], 2)
                        cv.imwrite(os.path.join(self.img_save_path, f'{self.base_name}.png'), img)

                else:

                    for j, label_name in enumerate(masks.keys()):
                        
                        Contours = cv.findContours(masks[label_name], cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[-2]
                        cv.drawContours(img, Contours, -1, colors[j], 2)
                        cv.imwrite(os.path.join(self.img_save_path, f'{self.base_name}_{i}.png'), img)

            if not i:
                
                cv.imwrite(os.path.join(self.img_save_path, f'{self.base_name}.png'), img)

                for label_name in masks.keys():
                    cv.imwrite(os.path.join(self.mask_save_path, 'A', f'{self.base_name}.png'), masks[label_name])
                    cv.imwrite(os.path.join(self.mask_save_path, label_name, f'{self.base_name}.png'), masks[label_name])
            
            else:

                cv.imwrite(os.path.join(self.img_save_path, f'{self.base_name}_{i}.png'), img)

                for label_name in masks.keys():
                    cv.imwrite(os.path.join(self.mask_save_path, 'A', f'{self.base_name}_{i}.png'), masks[label_name])
                    cv.imwrite(os.path.join(self.mask_save_path, label_name, f'{self.base_name}_{i}.png'), masks[label_name])

        print(f'{self.base_name} >>>>>>>>>> pass')


def split_dataset(train_path, labels, percent=0.1):

    val_path = train_path.replace('train', 'val')
    check_path(os.path.join(val_path, 'img'))

    for label_name in labels.keys():
        check_path(os.path.join(val_path, 'mask', label_name))

    imgs_list = os.listdir(os.path.join(train_path, 'img'))
    val_imgs_list = random.sample(imgs_list, int(len(imgs_list)*percent)+1)

    for val_img in val_imgs_list:

        shutil.move(os.path.join(train_path, 'img', val_img), os.path.join(val_path, 'img'))

        for label_name in labels.keys():
            shutil.move(os.path.join(train_path, 'mask', label_name, val_img), os.path.join(val_path, 'mask', label_name))


def split_IMG(labels, img_path, mask_path, save_path, crop_size, down_sample, overlap):

    check_path(os.path.join(save_path, 'img'))

    for label_name in labels.keys():
        check_path(os.path.join(save_path, 'mask', 'A'))
        check_path(os.path.join(save_path, 'mask', label_name))

    scale = 0.5 if down_sample else 1

    img_list = []
    path_list = os.listdir(img_path)
    masks_list = generate_masks_list(labels, shape=[crop_size, crop_size])
    mask_path_list = generate_masks_list(labels, shape=[crop_size, crop_size])

    for i, name in enumerate(path_list):

        new_img_path = os.path.join(img_path, name)
        img_list.append(new_img_path)

        for label_name in mask_path_list.keys():

            if not i:
                mask_path_list[label_name] = [os.path.join(mask_path, 'A', name)]
                # mask_path_list[label_name] = [os.path.join(mask_path, label_name, name)]

            else:
                mask_path_list[label_name].append(os.path.join(mask_path, 'A', name))
                # mask_path_list[label_name].append(os.path.join(mask_path, label_name, name))

    for i in tqdm(range(len(img_list))):

        for label_name in masks_list.keys():
            masks_list[label_name] = cv.resize(cv.imread(mask_path_list[label_name][i], 0), dsize=None, fx=scale, fy=scale)

        image = cv.imread(img_list[i])
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

        for n, (x, y) in enumerate(crop_coords):

            base_name = path_list[i].split('.')[0]

            save_img_path = os.path.join(save_path,  'img', f'{base_name}_{n}.png')
            cv.imwrite(save_img_path, image[y:y + crop_size, x:x + crop_size])

            for label_name in masks_list.keys():
                    
                save_mask_path = os.path.join(save_path, 'mask', 'A', f'{base_name}_{n}.png')
                # save_mask_path = os.path.join(save_path, 'mask', label_name, f'{base_name}_{n}.png')
                cv.imwrite(save_mask_path, masks_list[label_name][y:y + crop_size, x:x + crop_size])

    split_dataset(save_path, labels)


def generate_datasets(json_paths, svs_paths, save_path, labels, roiLabels, draw, level, crop_size, down_sample, overlap, flag):

    json_path_list = os.listdir(json_paths)
    colors = generate_color(len(labels.keys()))

    for path in json_path_list:

        json_path = os.path.join(json_paths, path)
        svs_path = os.path.join(svs_paths, path.replace('json', 'svs'))

        gm = generate_masks(svs_path, json_path, save_path, labels, draw, down_sample, level)
        gm.get_roi_mask(roiLabels, colors)

    if not draw:

        temp_name = generate_dataset_name(down_sample, crop_size, flag, level)

        root_path = json_paths.strip('\json')
        masks_path = os.path.join(root_path, 'label_roi_mask', 'mask')
        img_path = os.path.join(root_path, 'label_roi_mask', 'img_roi')
        save_path = os.path.join(root_path, 'patch', f'{temp_name}', 'train')

        split_IMG(labels, img_path, masks_path, save_path, crop_size, down_sample, overlap)