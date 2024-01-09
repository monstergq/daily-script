import cv2 as cv
import numpy as np
from geojson import load
import torch.nn.functional as F 
import os, json, torch, tifffile
from utils.NewJson import WriteJson
# from model.net.sam import Model as Model
from model.net.U2Net import U2net as Model
# from model.net.model import U2net as Model
# from model.net.UNext import UNext as Model
import torchvision.transforms as transforms
from utils.CropMergeSvs import classCropMerge


totensor = transforms.ToTensor()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def morphology(binary, surface):

    kernel = np.ones((3, 3), np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_DILATE, kernel, iterations=2)
    unknown = binary - surface

    return unknown


def watershed_algorithm(image):

    img = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    dist = cv.distanceTransform(binary, cv.DIST_L2, 5)
    dist_out = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    surface = cv.threshold(dist_out, 0.2*dist_out.max(), 255, cv.THRESH_BINARY)[1].astype(np.uint8)
    
    unknown = morphology(binary, surface)
    markers = cv.connectedComponents(surface)[1] + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(img, markers=markers)

    colors = [(0, 0, 0)] * 12

    for i in range(2, int(cv.minMaxLoc(markers)[1]+1)):
        
        thres1 = cv.threshold(markers.astype(np.uint8), i-1, 255, cv.THRESH_BINARY)[1]
        thres2 = cv.threshold(markers.astype(np.uint8), i, 255, cv.THRESH_BINARY)[1]
        
        # 生成轮廓掩膜
        mask = thres1 - thres2
        
        # 查找轮廓
        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2]
        
        # 在原图上绘制轮廓
        cv.drawContours(img, contours, -1, colors[(i - 2) % 12], 3)
        cv.drawContours(image, contours, -1, colors[(i - 2) % 12], 3)

    img_bgr = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    img_bgr[4:-4, 4:-4, :] = img[4:-4, 4:-4, :]

    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)


def get_roi_conts(json_path, key_label):

    with open(json_path, 'rb') as f:
        all_feature = load(f)

    features = all_feature['features']

    for feature in features:

        properties = feature['properties']

        if properties['label_name'] == key_label:

            geometry = feature['geometry']
            coord = cv.boundingRect(np.array(geometry['coordinates'][0], dtype=np.int32).squeeze())

    return coord


def get_contours(pred, use_dilate, use_watershed):

    if use_dilate:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        contours = cv.findContours(cv.dilate(pred, kernel), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[-2]

    else:
        contours = cv.findContours(pred, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[-2]

    res = []

    for contour in contours:

        if cv.contourArea(contour) >= 50:
            res.append(contour)
    
    img = cv.drawContours(np.zeros(pred.shape[:2], dtype=np.uint8), res, -1, 255, -1)

    if use_watershed:
        img = watershed_algorithm(img)

    return img


def read_svs(wsi_path, level):

    image = tifffile.TiffFile(wsi_path).pages[level].asarray()[:, :, ::-1]

    return image


def rect_filter(contours, thresh=600):

    res = []

    for contour in contours:

        area = cv.contourArea(contour)

        if area >= thresh and contour.shape[0] > 4:
            res.append(contour)

    return res


def line_filter(contours, thresh=400):

    error = []

    for id, contour in enumerate(contours):

        length, threshold = 0, thresh
        point_start = contour[0]

        for i in range(1, len(contour)):

            if length > threshold:
                error.append(id)
                break

            if (contour[i][0][0] == point_start[0][0]) or (contour[i][0][1] == point_start[0][1]):
                length += 1

            else:
                length = 0
                point_start = contour[i]

    return [contour for id, contour in enumerate(contours) if id not in error]


def generate_list(labels, label=None):

    res = []

    for _ in range(len(labels)):

        if label:
            res.append([label])
        else:
            res.append([])

    return res


def get_Targets(image, model, thresh=0.6, use_dilate=False, use_sigmoid=False, use_watershed=False, target=None):

    with torch.no_grad():

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        img = totensor(image.copy()).unsqueeze(0).to(device)
        # img = F.pad(img, pad=(0, input_size-img.shape[2], 0, input_size-img.shape[3]), mode='constant', value=0)

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


def example(root_path, model_path, labels, out_path, target, crop_size, overlap_size, num_classes, version, rect_thresh=1, line_thresh=10000, roi_label=None, downscale=1, use_dilate=False, use_sigmoid=False, use_watershed=False):
    
    model = Model(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    myCropMerge = classCropMerge(crop_size_height=crop_size, crop_size_width=crop_size, crop_stride=crop_size-overlap_size)

    for e in os.scandir(root_path):
        
        print(f'read svs:{e.name}')
        image = read_svs(e.path, 0)

        print(f'start crop')

        if roi_label:

            json_path = os.path.join(os.path.split(e.path)[0].replace('svs', 'gt_json'), os.path.split(e.path)[1].replace('.svs', '.json'))
            x, y, w, h = get_roi_conts(json_path, roi_label)
            x -= 11000
            y -= 2500
            w += 20000
            h += 20000
            patch_imgs = myCropMerge.crop_Image(image[y:y+h, x:x+w, :])

        else:

            patch_imgs = myCropMerge.crop_Image(image)

        del image
        
        masks_pre_list = generate_list(labels)

        for img in patch_imgs:

            mean_img = np.mean(img)

            if ((mean_img > 250) or (mean_img == 0)):

                roi_mask = np.zeros((img.shape[:2]), dtype=np.uint8)
                [masks_pre_list[i].append(roi_mask) for i in range(len(labels))]

            else:

                roi_masks = get_Targets(img, model, thresh=0.6, use_dilate=use_dilate, use_sigmoid=use_sigmoid, use_watershed=use_watershed, target=target)
                [masks_pre_list[i].append(roi_masks[i]) for i in range(len(labels))]
                
        print(f'start merge')
        masks_pre = myCropMerge.merge_Image(masks_pre_list)
        masks_pre = [cv.resize(mask_pre,(mask_pre.shape[1]//downscale, mask_pre.shape[0]//downscale)) for mask_pre in masks_pre]

        masks_pre = [get_contours(mask_pre, use_dilate, use_watershed) for mask_pre in masks_pre]
        masks_pre = [[cv.findContours(masks_pre[i], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2]] for i in range(len(labels))]

        Contours = generate_list(labels)

        for i, mask_pre in enumerate(masks_pre): 
            
            for mask in mask_pre[0]:

                if len(mask) >= 3:

                    mask *= downscale

                    if roi_label:
                        Contours[i].append(mask+[x, y])
                    else:
                        Contours[i].append(mask)

            Contours[i] = rect_filter(Contours[i], thresh=rect_thresh)
            Contours[i] = line_filter(Contours[i], thresh=line_thresh)

        labels_dict = get_labels_dict(labels)

        for i, label in enumerate(labels):

            labels[i] = generate_list(Contours[i], label)
            labels[i] = [label] * len(Contours[i])

        print(f'start save')
        write_json = WriteJson()
        write_json.write_json(Contours, labels, labels_dict, out_path, e.path, version)

        print(f'svs:{e.name} is done')
        print('-----------------------------------------')


if __name__ == '__main__':

    root_path, out_path = r'./configs/svs', r'./result/json'
    model_path = r'model_path/HG_cell_model_lastet.pth'

    crop_size = 512
    overlap_size = 128

    target = 0
    downscale = 1
    num_classes = 1
    rect_thresh = 50
    line_thresh = 1000
    labels = ["哈氏腺_腺泡细胞核"]

    roi_label = '大鼠哈氏腺标注区域'
    use_dilate = False
    use_sigmoid = False
    use_watershed = False

    version = '腺泡细胞核_v4.0.1'

    example(root_path, model_path, labels, out_path, target, crop_size, overlap_size, num_classes, version, rect_thresh, line_thresh, roi_label, downscale, use_dilate, use_sigmoid, use_watershed)