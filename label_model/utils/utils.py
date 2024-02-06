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
    

def get_contours(preds, use_dilate, use_watershed):

    for i, pred in enumerate(preds):

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

        if not i:
            IMG = [img]
        
        else:
            IMG.append(img)

    return IMG


def get_Targets(images, model, device, totensor, overlap_size, batch_size, num_class, thresh=0.6, use_dilate=False, use_sigmoid=False, use_watershed=False, target=None):

    with torch.no_grad():

        assert len(images) == batch_size

        for i, image in enumerate(images):

            # cv.imshow('image', image)
            # cv.waitKey(0)

            if not i:
                img = totensor(image.copy()).unsqueeze(0).to(device)
            
            else:
                img = torch.cat((img, totensor(image.copy()).unsqueeze(0).to(device)), dim=0)

        if use_sigmoid:
            preds = torch.sigmoid(model(img)[0]).cpu().detach().numpy()
        else:
            preds = model(img)[0].cpu().detach().numpy()

        if len(preds[1:].shape) == 2:

            input_size = preds.shape[1] - (2*overlap_size)
            preds = np.array(np.where(preds>thresh, 255., 0.), dtype=np.uint8)
            Contours = [[contour[overlap_size:overlap_size+input_size, overlap_size:overlap_size+input_size] for contour in get_contours(preds, use_dilate, use_watershed)]]

        elif target==2:

            for i in range(num_class):

                pred = preds[:, i, :, :]
                input_size = pred.shape[1] - (2*overlap_size)
                pred = np.array(np.where(pred>thresh, 255., 0.), dtype=np.uint8)

                if not i:
                    Contours = [[contour[overlap_size:overlap_size+input_size, overlap_size:overlap_size+input_size] for contour in get_contours(pred, use_dilate, use_watershed)]]
                    # Contours = [get_contours(pred, use_dilate, use_watershed)[overlap_size:overlap_size+input_size, overlap_size:overlap_size+input_size]]

                else:
                    Contours.append([contour[overlap_size:overlap_size+input_size, overlap_size:overlap_size+input_size] for contour in get_contours(pred, use_dilate, use_watershed)])
                    # Contours.append(get_contours(pred, use_dilate, use_watershed)[overlap_size:overlap_size+input_size, overlap_size:overlap_size+input_size])
        
        elif target==1:

            # for i in range(4):
            #     res = preds[i, 0, :, :]
            #     cv.imshow('res', res)
            #     cv.waitKey(0)
            pred = preds[:, 0, :, :]
            input_size = pred.shape[1] - (2*overlap_size)
            pred = np.array(np.where(pred>thresh, 255., 0.), dtype=np.uint8)
            Contours = [[contour[overlap_size:overlap_size+input_size, overlap_size:overlap_size+input_size] for contour in get_contours(pred, use_dilate, use_watershed)]]
            # Contours = [get_contours(pred, use_dilate, use_watershed)[overlap_size:overlap_size+input_size, overlap_size:overlap_size+input_size]]

        else:

            pred = preds[:, 1, :, :]
            input_size = pred.shape[1] - (2*overlap_size)
            pred = np.array(np.where(pred>thresh, 255., 0.), dtype=np.uint8)
            Contours = [[contour[overlap_size:overlap_size+input_size, overlap_size:overlap_size+input_size] for contour in get_contours(pred, use_dilate, use_watershed)]]
            # Contours = [get_contours(pred, use_dilate, use_watershed)[overlap_size:overlap_size+input_size, overlap_size:overlap_size+input_size]]

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


def get_pretrain(model, path):

    """ 加载预训练模型，并打印出未加载部分"""
    if path != '':

        print('Load weights {}.'.format(path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        load_key, no_load_key, temp_dict = [], [], {}

        for k, v in pretrained_dict.items():

            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)

        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        print("\nSuccessful Load Key:", str(load_key)[:500], "\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "\nFail To Load Key num:", len(no_load_key))

        return model


def load_model(Model, model_path, num_classes, device):

    model = Model(num_classes).to(device)
    model = get_pretrain(model, model_path)
    # model.load_state_dict(torch.load(model_path))
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
    

def read_svs(wsi_path, level, trans_channel, overlap_size):

    image = tifffile.TiffFile(wsi_path).pages[level].asarray()

    if overlap_size != 0:
        image = cv.copyMakeBorder(image, overlap_size, overlap_size, overlap_size, overlap_size, cv.BORDER_REFLECT)

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