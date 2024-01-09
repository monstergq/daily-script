from utils.generate_masks import generate_datasets


if __name__ == '__main__':

    draw = False  # 是否把标注数据画出来，如果画则不切图
    trans_channel = True

    level = 0
    overlap = 0
    down_sample = False  # 下采样倍率，这个参数是为了完成可以取20x
    crop_size = 1024  # 裁剪大小=crop_size//down_sample, down_sample=2 if down_sample else 1

    flag = 'erode'  # 数据集加的后缀名

    labels = {'A': False, 'B': True}
    roiLabels = ['ROI']

    json_paths = r'D:\datasets\datasets_TG\A+B\json'
    svs_paths = r'D:\datasets\datasets_TG\svs'
    save_path = r'D:\datasets\datasets_TG\A+B\label_roi_mask'

    generate_datasets(json_paths, svs_paths, save_path, labels, roiLabels, draw, level, crop_size, down_sample, overlap, flag)