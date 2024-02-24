import os
import cv2 as cv
import numpy as np
from utils.utils import *
from utils.NewJson import WriteJson
from utils.eval_metric import eval_res
# from model.net.sam import Model as Model
from model.net.U2Net import U2net as Model
# from model.net.model import U2net as Model
# from model.net.UNext import UNext as Model
import torchvision.transforms as transforms
from utils.CropMergeSvs import classCropMerge


def predict_cotours(config_path):

    para = read_yaml(config_path)
    model = load_model(Model, para.Path.model_path, para.Model.num_classes, para.Model.device)
    myCropMerge = classCropMerge(para.Parameter.crop_size, para.Parameter.crop_size, para.Parameter.crop_size)

    totensor = transforms.ToTensor()

    _, name = os.path.split(para.Path.root_path)
    print(f'svs is :{para.Path.root_path}')

    print(f'start crop')
    patch_imgs = myCropMerge.crop_Image(para)
    masks_pre_list = generate_list(para.Model.labels)

    for img in patch_imgs:

        roi_masks = get_Targets(img, model, para.Model.device, totensor, para.Parameter.overlap_size, para.Parameter.batch_size, para.Model.num_classes, thresh=0.5, use_dilate=para.Post.use_dilate, use_sigmoid=para.Post.use_sigmoid, use_watershed=para.Post.use_watershed, target=para.Model.target)
        [masks_pre_list[i].append(roi_masks[i]) for i in range(len(para.Model.labels))]
            
    print(f'start merge')
    masks_pre = myCropMerge.merge_Image(para, masks_pre_list)
    downscale = 2.0 if para.Parameter.downscale else 1
    masks_pre = [cv.resize(mask_pre,(mask_pre.shape[1]//downscale, mask_pre.shape[0]//downscale)) for mask_pre in masks_pre]

    # masks_pre = [get_contours(mask_pre, para.Post.use_dilate, para.Post.use_watershed) for mask_pre in masks_pre]
    masks_pre = [[cv.findContours(masks_pre[i], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2]] for i in range(len(para.Model.labels))]

    Contours = generate_list(para.Model.labels)

    for i, mask_pre in enumerate(masks_pre): 
        
        for mask in mask_pre[0]:

            if len(mask) >= para.Post.polygon_filter+1:

                mask *= downscale
                Contours[i].append(mask)

        Contours[i] = rect_filter(Contours[i], thresh=para.Post.rect_thresh)
        Contours[i] = line_filter(Contours[i], thresh=para.Post.line_thresh)

    labels = para.Model.labels.copy()
    labels_dict = get_labels_dict(para.Model.labels)

    for i, label in enumerate(para.Model.labels):

        labels[i] = generate_list(Contours[i], label)
        labels[i] = [label] * len(Contours[i])

    print(f'start save')
    write_json = WriteJson()
    write_json.write_json(Contours, labels, labels_dict, para.Path.out_path, para.Path.root_path, para.Other.version)

    print(f'svs:{name} is done')
    print('-----------------------------------------')

    check_path(os.path.join(para.Path.out_path.replace('json', 'excel'), name.strip('.svs')))
    json_path = os.path.join(para.Path.out_path, f"{name.strip('.svs')}_{para.Other.version}.json")
    out_csv_path = os.path.join(para.Path.out_path.replace('json', 'excel'), name.strip('.svs'), f"{name.strip('.svs')}_{para.Other.version}.csv")
    eval_res(para.Path.gt_json_path, json_path, para.Model.labels[0], para.Model.labels[0], para.Model.roi_label[0], out_csv_path)