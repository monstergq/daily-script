import os, torch
import cv2 as cv
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from model.net.ResNet import ResNet


def check_path(path):

    if not os.path.exists(path):
        os.makedirs(path)


def to_pas(Generator, img):

    trans = transforms.Compose([transforms.ToTensor()])

    img = trans(img.copy()).to(torch.float32).to('cuda').unsqueeze(0)
    fake_img = Generator(img).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)*255.0

    return fake_img.astype(np.uint8)
    

def read_data(img_path, mask_path, save_path, Generator, down_sample=1):

    scale = 1 / down_sample
    img_list, mask_list = [], [[], []]
    path_list = os.listdir(img_path)

    for name in path_list:

        new_img_path = os.path.join(img_path, name)

        new_mask_path_A = os.path.join(mask_path, 'A', name)
        new_mask_path_B = os.path.join(mask_path, 'B', name)

        img_list.append(new_img_path)
        
        mask_list[0].append(new_mask_path_A)
        mask_list[1].append(new_mask_path_B)

    crop_size = 1024

    for i in tqdm(range(len(img_list))):

        mask_A = cv.imread(mask_list[0][i], 0)
        mask_A = cv.resize(mask_A, dsize=None, fx=scale, fy=scale)

        mask_B = cv.imread(mask_list[1][i], 0)
        mask_B = cv.resize(mask_B, dsize=None, fx=scale, fy=scale)

        image = cv.imread(img_list[i])
        image = cv.resize(image, dsize=None, fx=scale, fy=scale)

        crop_coords = []
        h, w = image.shape[:2]

        for y in range(0, h, crop_size):

            if y + crop_size > h:
                y = h - crop_size

            for x in range(0, w, crop_size):

                if x + crop_size > w:
                    x = w - crop_size

                crop_coords.append((x, y))

        for n, (x, y) in enumerate(crop_coords):

            if np.sum(mask_A[y:y + crop_size, x:x + crop_size] != 0) < 100 \
                and np.sum(mask_B[y:y + crop_size, x:x + crop_size] != 0) < 100:
                continue

            base_name = path_list[i].split('.')[0]
            save_img_path = os.path.join(save_path,  'img', f'{base_name}_{n}.png')
            save_mask_A_path = os.path.join(save_path, 'mask', 'A', f'{base_name}_{n}.png')
            save_mask_B_path = os.path.join(save_path, 'mask', 'B', f'{base_name}_{n}.png')
            cv.imwrite(save_img_path, image[y:y + crop_size, x:x + crop_size])
            cv.imwrite(save_mask_A_path, mask_A[y:y + crop_size, x:x + crop_size])
            cv.imwrite(save_mask_B_path, mask_B[y:y + crop_size, x:x + crop_size])

            # img = cv.cvtColor(image[y:y + crop_size, x:x + crop_size], cv.COLOR_RGB2BGR)
            img = image[y:y + crop_size, x:x + crop_size]
            fake_img = to_pas(Generator, img)
            save_img_path = os.path.join(save_path,  'img', f'{base_name}_{n}_pas.png')
            save_mask_A_path = os.path.join(save_path, 'mask', 'A', f'{base_name}_{n}_pas.png')
            save_mask_B_path = os.path.join(save_path, 'mask', 'B', f'{base_name}_{n}_pas.png')
            cv.imwrite(save_img_path, fake_img)
            cv.imwrite(save_mask_A_path, mask_A[y:y + crop_size, x:x + crop_size])
            cv.imwrite(save_mask_B_path, mask_B[y:y + crop_size, x:x + crop_size])


if __name__ == '__main__':

    img_path = r'D:\datasets\datasets_HG\label_roi_mask\img_roi'
    masks_path = r'D:\datasets\datasets_HG\label_roi_mask\mask'
    save_path = r'D:\datasets\datasets_HG\patch\20x_1024_pas\train'
    model_path = r'./model_path/2_20X_1024_Ga2b.pth'

    Generator = ResNet(in_channel=3).to('cuda')
    Generator.load_state_dict(torch.load(model_path, map_location='cpu'))
    # Generator.eval()

    check_path(os.path.join(save_path, 'img'))
    check_path(os.path.join(save_path, 'mask', 'A'))
    check_path(os.path.join(save_path, 'mask', 'B'))

    read_data(img_path, masks_path, save_path, Generator, down_sample=2.0)