import os, json
import cv2 as cv
import numpy as np


def example(jsons_path, images_path, save_masks_path):

    for json_path in os.scandir(jsons_path):

        name = json_path.name.split('.')[0]

        image_path = os.path.join(images_path, name+'.png')
        image = cv.imread(image_path)

        with open(json_path, 'r') as f:

            contours = []
            json_data = json.load(f)
            mask = np.zeros(image.shape[:2], np.uint8)

            for region in json_data['shapes']:

                contour = []

                if region['label'] == 'A':

                    for i in region['points']:
                        contour.append([int(i[0]), int(i[1])])

                    contours.append(np.array(contour))

            if contours != []:
                cv.drawContours(mask, contours, -1, (255), -1) 

            cv.imwrite(f'{save_masks_path}/{name}.png', mask)


if __name__ == '__main__':

    json_path = r'D:\datasets\datasets_TG\patch\level0_1024\json'
    image_path =r'D:\datasets\datasets_TG\patch\level0_1024\img'
    save_mask_path = r'D:\datasets\datasets_TG\patch\level0_1024\mask'

    example(json_path, image_path, save_mask_path)