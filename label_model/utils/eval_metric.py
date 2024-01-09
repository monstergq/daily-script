import cv2 as cv
import numpy as np
import pandas as pd
from geojson import load
import os, csv, sys, time


class calculate_metric():

    def __init__(self, pre_con, mask_con, level=1):

        # 调整轮廓坐标
        self.pre_con, self.mask_con = np.array(pre_con, dtype=object) // (4**level), np.array(mask_con, dtype=object) // (4**level)

        # 获取掩膜图像
        self.pre, self.mask, (x, y, h, w) = self.get_mask(self.pre_con, self.mask_con)
        self.x, self.y, self.h, self.w = x, y, h, w

        self.pre_con, self.mask_con = self.get_con(self.pre_con, self.mask_con)
        self.pre_con_, self.mask_con_ = [con for con in self.pre_con], [con for con in self.mask_con]

        self.union_con_ = cv.findContours(cv.bitwise_or(self.pre, self.mask), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    
    def get_con(self, pre_cons, mask_cons):

        # 计算图像高度和宽度
        x11, y11 = np.squeeze(np.min([np.min(con, axis=0) for con in pre_cons], axis=0))
        x21, y21 = np.squeeze(np.min([np.min(con, axis=0) for con in mask_cons], axis=0))

        x, y = min(x11, x21), min(y11, y21)

        pre_cons = [np.array(con-(x, y)+(100, 100), dtype=np.int32) for con in pre_cons]
        mask_cons = [np.array(con-(x, y)+(100, 100), dtype=np.int32) for con in mask_cons]

        return pre_cons, mask_cons

    def get_mask(self, pre_con, mask_con, d=-1):

        # 计算图像高度和宽度
        x11, y11 = np.squeeze(np.min([np.min(con, axis=0) for con in pre_con], axis=0))
        x12, y12 = np.squeeze(np.max([np.max(con, axis=0) for con in pre_con], axis=0))
        x21, y21 = np.squeeze(np.min([np.min(con, axis=0) for con in mask_con], axis=0))
        x22, y22 = np.squeeze(np.max([np.max(con, axis=0) for con in mask_con], axis=0))

        x, y, x_max, y_max = min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22)

        h, w = y_max-y+200, x_max-x+200

        # 创建掩膜图像
        pre = cv.drawContours(np.zeros((h, w), dtype=np.uint8), [np.array(cnt-(x, y)+(100, 100), dtype=np.int32) for cnt in pre_con], -1, 255, d)
        mask = cv.drawContours(np.zeros((h, w), dtype=np.uint8), [np.array(cnt-(x, y)+(100, 100), dtype=np.int32) for cnt in mask_con], -1, 255, d) 

        return pre, mask, (x, y, h, w)
    
    def get_area(self, pre_con, mask_con, mode=0):

        if mode == 2:

            P1, G1, _ = self.get_mask(pre_con, mask_con, d=1)
            Pd, Gd, _ = self.get_mask(pre_con, mask_con, d=10)
            recall = np.sum(cv.bitwise_and(Pd, G1)==255.0) / (np.sum(G1==255.0) + 1e-10)
            precision = np.sum(cv.bitwise_and(P1, Gd)==255.0) / (np.sum(P1==255.0) + 1e-10)

            return recall, precision
        
        elif mode == 3:

            P, _, _ = self.get_mask(pre_con, mask_con, d=-1)
            _, Gd, _ = self.get_mask(pre_con, mask_con, d=10)

            res = np.sum(cv.bitwise_and(Gd, P)==255.0) / (np.sum(Gd==255.0) + 1e-10)

            return res

        else:

            in_area, ou_area = self.get_in_area(pre_con, mask_con, mode), self.get_ou_area(pre_con, mask_con, mode)

            return in_area, ou_area

    def get_ou_area(self, pre_con, mask_con, mode=0):

        if mode == 0:

            pre, mask, _ = self.get_mask(pre_con, mask_con)

        elif mode == 1:
                
            pre, mask, _ = self.get_mask(pre_con, mask_con, d=10)

        union = cv.bitwise_or(pre, mask)

        area = np.sum(union==255.0)

        return area
    
    
    def get_in_area(self, pre_con, mask_con, mode=0):

        if mode == 0:

            pre, mask, _ = self.get_mask(pre_con, mask_con)

        elif mode == 1:
                
            pre, mask, _ = self.get_mask(pre_con, mask_con, d=10)

        inter = cv.bitwise_and(pre, mask)
        area = np.sum(inter==255.0)

        return area

    def get_true_metrics(self, union_idx):

        # 初始化存储Metrics的字典
        num_miss, num_wrong = 0, 0
        IoUs, FIoUs, BIoUs, TIoUs = {}, {}, {}, {}

        # 计算Metrics
        for key, values in union_idx.items():

            if len(values['pre']) == 0:

                num_miss += 1
                
            elif len(values['mask']) == 0:

                num_wrong += 1

            else:

                # 计算IoU
                inter_area, union_area = self.get_area([self.pre_con_[i] for i in values['pre']], [self.mask_con_[i] for i in values['mask']], mode=0)
                IoUs[key] = inter_area / (union_area + 1e-10)

                # 计算BIoU
                inter_area, union_area = self.get_area([self.pre_con_[i] for i in values['pre']], [self.mask_con_[i] for i in values['mask']], mode=1)
                BIoUs[key] = inter_area / (union_area + 1e-10)

                # 计算FIoU
                recall, precision = self.get_area([self.pre_con_[i] for i in values['pre']], [self.mask_con_[i] for i in values['mask']], mode=2)
                FIoUs[key] = 2 * recall * precision / (recall + precision + 1e-10)

                # 计算TIoU
                res = self.get_area([self.pre_con_[i] for i in values['pre']], [self.mask_con_[i] for i in values['mask']], mode=3)
                TIoUs[key] = res

        return num_miss, num_wrong, IoUs, FIoUs, BIoUs, TIoUs
        

    def calculate(self):

        # 初始化存储交并比的字典
        union_idx = {}

        for id_union, i in enumerate(self.union_con_):

            # 计算并集面积，若面积为0则跳过
            if cv.contourArea(i) == 0:
                continue

            union_idx[id_union] = {'mask': [], 'pre': []}

            for id_mask, j in enumerate(self.mask_con_):

                x, y = j[0][0], j[0][1]
                ret = cv.pointPolygonTest(i, (int(x), int(y)), False)

                if ret >= 0:
                    
                    # 保存真实轮廓的索引
                    union_idx[id_union]['mask'].append(id_mask)

            for id_pre, k in enumerate(self.pre_con_):
                
                x, y = k[0][0], k[0][1]

                ret = cv.pointPolygonTest(i, (int(x), int(y)), False)

                if ret >= 0:

                    # 保存预测轮廓的索引
                    union_idx[id_union]['pre'].append(id_pre)

        # 计算真实交并比
        num_miss, num_wrong, IoUs, FIoUs, BIoUs, TIoUs = self.get_true_metrics(union_idx)

        # 计算漏误率
        size_mask, size_pred = len(self.mask_con_), len(self.pre_con_)
        sys.stdout.write(f' 漏检率: {num_miss/size_mask:.3f}({num_miss}/{size_mask}),\t 误检率: {num_wrong/size_pred:.3f}({num_wrong}/{size_pred}),\t')

        return IoUs, FIoUs, BIoUs, TIoUs, [num_miss, size_mask], [num_wrong, size_pred]


def is_in_poly(point, contours):

    is_in = False
    x, y = point

    for i, corner in enumerate(contours):

        next_i = i + 1 if i + 1 < len(contours) else 0

        x1, y1 = corner
        x2, y2 = contours[next_i]

        if (x1==x and y1==y) or (x2==x and y2==y):

            is_in = True
            break

        if min(y1, y2) < y <= max(y1, y2):
                
                X = x1 + (y-y1) * (x2-x1) / (y2-y1)
        
                if X == x:
                    is_in = True
                    break
        
                if X > x:
                    is_in = not is_in

    return is_in


def get_contours_within_region(contours, region_points):

    if len(region_points)==0:
        return contours
    
    filtered_contours = []

    for contour in contours:

        swapped_contour = []

        for point in contour:
            
            try:
                x, y = point[:2]
            except:
                pass

            for region_polygon in region_points:
        
                if is_in_poly([x, y], region_polygon):
                    swapped_contour.append([x, y])

        if swapped_contour:
            filtered_contours.append(np.array(swapped_contour, dtype=np.int32))

    return filtered_contours


def get_contours_from_json(json_path, key_label, roi_label=None):
    
    new_contours, roi_contours = {}, []  # 用于存储新轮廓的列表

    with open(json_path, 'rb') as f:
        all_feature = load(f)

    features = all_feature['features']

    for feature in features:

            properties = feature['properties']

            if properties['label_name'] == key_label:

                geometry = feature['geometry']

                if np.array(geometry['coordinates'][0]).shape[0] > 2:

                    if properties['annotation_owner'] not in new_contours:
                        new_contours[properties['annotation_owner']] = [np.array(geometry['coordinates'][0], dtype=np.int32).squeeze()]

                    else:
                        new_contours[properties['annotation_owner']].append(np.array(geometry['coordinates'][0], dtype=np.int32).squeeze())
            
            elif properties['label_name'] == roi_label:
                    
                    geometry = feature['geometry']
                    roi_contours.append(np.array(geometry['coordinates'][0], dtype=np.int32).squeeze())

    if len(roi_contours) > 0:
        return new_contours, roi_contours
    
    else:
        return  new_contours


def print_metrics(metric, metrics_data, label, user_value, mode='miou'):

    mean_metric_values = list(metric.values())
    average_mean_metric = np.mean(mean_metric_values)
    metrics_data[user_value][label][mode] = average_mean_metric

    if mode != 'tiou':
        print(f" {mode}: {np.mean(average_mean_metric):.3f}", end='\t')

    else:
        print(f" {mode}: {np.mean(average_mean_metric):.3f}")

    return metrics_data


def process_json_files(pred_contours, gt_contours, roi_area, label, user_ai):

    if roi_area: # 过滤预测区域
        pred_contours = get_contours_within_region(pred_contours[user_ai], roi_area)

    user_metrics_data = {}

    for user_value in gt_contours.keys():

        user_metrics_data[user_value] = {}
        sys.stdout.write(f'{user_value}:\t')

        gt_contours_ = gt_contours[user_value]
        user_metrics_data[user_value][label] = {}

        if gt_contours_ and pred_contours:

            cal_metric = calculate_metric(pred_contours, gt_contours_)

            sys.stdout.write(f' {len(pred_contours)}({label},{user_ai}) vs {len(gt_contours_)}({label},{user_value})\t')

            MIoUs, FIoUs, BIoUs, TIoUs, Miss, Wrong = cal_metric.calculate()

            user_metrics_data = print_metrics(MIoUs, user_metrics_data, label, user_value, mode='miou')
            user_metrics_data = print_metrics(FIoUs, user_metrics_data, label, user_value, mode='fiou')
            user_metrics_data = print_metrics(BIoUs, user_metrics_data, label, user_value, mode='biou')
            user_metrics_data = print_metrics(TIoUs, user_metrics_data, label, user_value, mode='tiou')

            user_metrics_data[user_value][label]['num_miss'] = Miss[0]
            user_metrics_data[user_value][label]['size_mask'] = Miss[1]
            user_metrics_data[user_value][label]['loss_rate'] = Miss[0] / Miss[1]

            user_metrics_data[user_value][label]['num_wrong'] = Wrong[0]
            user_metrics_data[user_value][label]['size_pred'] = Wrong[1]
            user_metrics_data[user_value][label]['wrong_rate'] = Wrong[0] / Wrong[1]

        elif not pred_contours:
            print('there no contours in pred_contours')

        elif not gt_contours_:
            print('there no contours in gt_contours')

    return user_metrics_data


def write_results(res, label, csv_file_path):

    with open(csv_file_path, mode='w', newline='') as csv_file:

        writer = csv.writer(csv_file)

        header_row = ['user name', 'miou', 'fiou', 'biou', 'tiou', 'loss_rate', 'wrong_rate', 'num_miss', 'size_mask', 'num_wrong', 'size_pred']
        writer.writerow(header_row)

        for user_ai in res.keys():

            for user in res[user_ai].keys():

                data_row = [f'{user_ai} vs {user}']

                for head in header_row[1:]:

                    data_row.append(res[user_ai][user][label][head])
            
                writer.writerow(data_row)


def eval_res(gt_dir, pred_dir, label_gt, label_pred, roi_label, out_csv_path):

    Res = {}
    pre_contours = get_contours_from_json(pred_dir, label_pred)
    gt_contours, roi_area = get_contours_from_json(gt_dir, label_gt, roi_label)

    Res['0'] = process_json_files(pre_contours, gt_contours, roi_area, label_pred, '0')
    print('-'*20)

    for user in gt_contours.keys():

        pre_contours = gt_contours[user]
        gt_contours_ = gt_contours.copy()
        del gt_contours_[user]

        Res[user] = process_json_files(pre_contours, gt_contours_, None, label_pred, user)
        print('-'*20)

    write_results(Res, label_pred, out_csv_path)