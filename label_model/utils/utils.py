import cv2 as cv
import numpy as np
from addict import Dict
from geojson import load
from utils.PostProcess import *
import os, csv, json, math, yaml, torch, random, zipfile, openslide, tifffile


def zip_jsonfile(filename):

    _, name = os.path.split(filename)

    zipf = zipfile.ZipFile(name.replace('.json', '.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipf.write(name)

    zipf.close()


def read_csv(filename):

    data = []

    # 打开CSV文件并读取
    with open(filename, newline='', encoding='utf-8') as file:

        csv_dict_reader = csv.DictReader(file)

        for row in csv_dict_reader:
            data.append(row)

    return data


def read_yaml(fpath=None):

    with open(fpath, mode="r", encoding='utf-8') as file:

        yml = yaml.load(file, Loader=yaml.Loader)
        
        return Dict(yml)
    

def get_contours(pred, use_dilate, use_watershed):

    if use_dilate:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        contours = cv.findContours(cv.dilate(pred, kernel), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[-2]

    else:
        contours = cv.findContours(pred, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[-2]

    res = []

    for contour in contours:

        if cv.contourArea(contour) >= 10:
            res.append(contour)
    
    img = cv.drawContours(np.zeros(pred.shape[:2], dtype=np.uint8), res, -1, 255, -1)

    if use_watershed:
        img = watershed_algorithm(img)

    return img


def get_Targets(image, model, device, totensor, thresh=0.6, use_dilate=False, use_sigmoid=False, use_watershed=False, target=None):

    with torch.no_grad():

        img = totensor(image.copy()).unsqueeze(0).to(device)

        if use_sigmoid:
            preds = torch.sigmoid(model(img)[0]).cpu().detach().numpy().squeeze()
        else:
            preds = model(img)[0].cpu().detach().numpy().squeeze()

        if len(preds.shape) == 2:

            preds = np.array(np.where(preds>thresh, 255., 0.), dtype=np.uint8)
            Contours = [get_contours(preds, use_dilate, use_watershed)]

        elif target==2:

            for i, pred in enumerate(preds):

                pred = np.array(np.where(pred>thresh, 255., 0.), dtype=np.uint8)

                if not i:
                    Contours = [get_contours(pred, use_dilate, use_watershed)]
                
                else:
                    Contours.append(get_contours(pred, use_dilate, use_watershed))
        
        elif target==1:

            for i, pred in enumerate(preds):

                pred = np.array(np.where(pred>thresh, 255., 0.), dtype=np.uint8)

                if i:
                    Contours = [get_contours(pred, use_dilate, use_watershed)]

        else:

            for i, pred in enumerate(preds):

                pred = np.array(np.where(pred>thresh, 255., 0.), dtype=np.uint8)

                if not i:
                    Contours = [get_contours(pred, use_dilate, use_watershed)]

    return Contours


def get_labels_dict(labels):

    with open('./configs/label_code.json', 'r', encoding='utf8') as f:
        laebl_dict_ = json.load(f)

    for i, label in enumerate(labels):

        orgn, struct = label.split('_')

        label_code = laebl_dict_[orgn][struct]

        if not i:
            labels_dict = {label: {'label_name': label, 'label_color': "rgba(125, 128, 128, 1)", 'label_code': label_code}}
        else:
            labels_dict[label] = {'label_name': label, 'label_color': "rgba(125, 128, 128, 1)", 'label_code': label_code}

    return labels_dict


def load_model(Model, model_path, num_classes, device):

    model = Model(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def check_path(path):

    if not os.path.exists(path):
        os.makedirs(path)


def generate_dataset_name(down_sample, crop_size, flag, level):

    if down_sample:

        temp_name = f'20X_{crop_size*2}_{flag}' if flag else f'20X_{crop_size*2}'

    else:

        level_name = {0: '40X', 1: '10X', 2: '2.5X'}
        temp_name = f'{level_name[level]}_{crop_size}_{flag}' if flag else f'{level_name[level]}_{crop_size}'

    return temp_name


def get_img_from_wsi(svs_path, level, roi_area):

    slide = openslide.OpenSlide(svs_path)

    if roi_area.any():

        min_x, min_y, max_x, max_y = find_rect_point(roi_area)
        img_roi = np.array(slide.read_region((min_x, min_y), level, (max_x-min_x, max_y-min_y)))[:, :, -2::-1]

        return img_roi.copy(), min_x, min_y

    else:

        w, h = slide.level_dimensions[level]
        img_roi = np.array(slide.read_region((0, 0), level, (w, h)))[:, :, :3][:, :, ::-1]

        return img_roi.copy()
    

def read_svs(wsi_path, level, trans_channel):

    image = tifffile.TiffFile(wsi_path).pages[level].asarray()

    if trans_channel:
        return image[:, :, ::-1]
    
    else:
        return image
    
    
def find_rect_point(contours):

    min_x, min_y, max_x, max_y = math.inf, math.inf, 0, 0

    x, y, w, h = cv.boundingRect(contours)
    min_x = x if x < min_x else min_x
    min_y = y if y < min_y else min_y
    max_x = x + w if x + w > max_x else max_x
    max_y = y + h if y + h > max_y else max_y

    return min_x, min_y, max_x, max_y
    

def get_roi_conts(shapes, roi_labels):

    new_contours = []  # 用于存储新轮廓的列表

    for shape in shapes:

        if shape['label'] in roi_labels:

            new_contours.append(np.abs(np.array([shape['points'][-1]], dtype=np.int32)).squeeze())

    return new_contours  # 返回包含轮廓的列表
    

def read_geojson(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        data = load(f)

    try:
        shapes = [
                    dict(
                            label=feature["properties"]["label_name"],
                            points=feature["geometry"]["coordinates"],
                        )
                    for feature in data["features"]
                ]
    except:

        shapes = []

        for feature in data["features"]:
            
            try:

                shape = dict(
                                label=feature["properties"]["label_name"],
                                points=feature["geometry"]["coordinates"],
                            )
                shapes.append(shape)
            
            except:

                print(f'error in {json_path}')

    return shapes


def generate_list(labels, label=None):

    res = []

    for _ in range(len(labels)):

        if label:
            res.append([label])
        else:
            res.append([])

    return res


def generate_masks_list(labels, shape):

    masks = {}

    for label_name in labels:
        masks[label_name] = np.zeros(shape, dtype=np.uint8)

    return masks


def erode(image, kernel_size=3):

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv.erode(image, kernel, iterations=1)

    return erosion


def generate_color(num_classes):

    colors = []

    hue_values = random.sample(range(0, 360), num_classes)

    for hue_value in hue_values:

        hsv_color = np.array([[[hue_value, 255, 255]]]).astype(np.uint8)
        bgr_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR).squeeze()
        colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))

    return colors