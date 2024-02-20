import os, sys
import numpy as np
from utils.utils import *
from openslide import OpenSlide


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)


class classCropMerge:

    def __init__(self, crop_size_width=1024, crop_size_height=1024, crop_stride=768):

        """
        Process Image.
        :param crop_size_width: 512 default.
        :param crop_size_height: 512 default.
        :param crop_stride:
        """

        self.crop_stride = crop_stride
        self.crop_size_width = crop_size_width
        self.crop_size_height = crop_size_height
        
        self.width = None
        self.height = None
        self.points = None

    def get_image(self, para, i, j, level):

        i = i - self.overlap_size
        j = j - self.overlap_size

        w_min_border = 0-i if i < 0 else 0
        h_min_border = 0-j if j < 0 else 0

        w_max_border = (j+self.crop_size_width+self.overlap_size)-self.x-self.width if (j+self.crop_size_width+self.overlap_size) > self.x+self.width else 0
        h_max_border = (i+self.crop_size_height+self.overlap_size)-self.y-self.height if (i+self.crop_size_height+self.overlap_size) > self.y+self.height else 0

        if i < 0 and j < 0:
            image = np.array(self.svs.read_region((0, 0), level, (self.crop_size_height+self.overlap_size, self.crop_size_width+self.overlap_size)))[:, :, :3]
            image = cv.copyMakeBorder(image, w_min_border, w_max_border, h_min_border, h_max_border, cv.BORDER_REFLECT)
        
        elif i < 0 and j >= 0:
            image = np.array(self.svs.read_region((j, 0), level, (self.crop_size_height+2*self.overlap_size, self.crop_size_width+self.overlap_size)))[:, :, :3]
            image = cv.copyMakeBorder(image, w_min_border, w_max_border, h_min_border, h_max_border, cv.BORDER_REFLECT)
        
        elif i >= 0 and j < 0:
            image = np.array(self.svs.read_region((0, i), level, (self.crop_size_height+self.overlap_size, self.crop_size_width+2*self.overlap_size)))[:, :, :3]
            image = cv.copyMakeBorder(image, w_min_border, w_max_border, h_min_border, h_max_border, cv.BORDER_REFLECT)
        
        else:
            image = np.array(self.svs.read_region((j, i), level, (self.crop_size_height+2*self.overlap_size, self.crop_size_width+2*self.overlap_size)))[:, :, :3]

        if para.Parameter.trans_channel:
            return image[:, :, ::-1]

        else:
            return image

    def crop_Image(self, para):

        """
        Crop Image.
        :param image_shape: Image shape.
        :return: Patchs coords.
        """

        self.points = []
        self.x, self.y = 0, 0
        
        self.svs = OpenSlide(para.Path.root_path)
        self.overlap_size = para.Parameter.overlap_size

        if para.Model.roi_label[-1]:

            shapes = read_geojson(para.Path.gt_json_path)
            roi_contours = get_roi_conts(shapes, para.Model.roi_label[0])[0]
            self.x, self.y, w, h = cv.boundingRect(roi_contours)
            self.height, self.width = h, w

        else:

            self.width, self.height = self.svs.level_dimensions[para.Parameter.level]

        print(f'self.height is: {self.height}, self.width is: {self.width}')

        counter = 0
        patch_imgs = []

        for i in range(self.y, self.y+self.height, self.crop_stride):

            if i+self.crop_size_height > self.y+self.height:
                i = self.y + self.height - self.crop_size_height

            for j in range(self.x, self.x+self.width, self.crop_stride):

                if j+self.crop_size_width > self.x+self.width:
                    j = self.x + self.width - self.crop_size_width

                image = self.get_image(para, i, j, para.Parameter.level)

                if para.Parameter.batch_size == 1:
                        
                    self.points.append([[i, j]])
                    patch_imgs.append([image])

                else:

                    if counter == 0:

                        counter += 1
                        batch_coord = [[i, j]]
                        batch_imgs = [image]
                    
                    elif counter == para.Parameter.batch_size-1:

                        counter = 0
                        batch_coord.append([i, j])
                        self.points.append(batch_coord)
                        batch_imgs.append(image)
                        patch_imgs.append(batch_imgs)
                    
                    else:

                        counter += 1
                        batch_coord.append([i, j])
                        batch_imgs.append(image)

        return patch_imgs

    def merge_Image(self, para, imagelist):

        """
        Merageg Image.
        :param imagelist: Patch list.
        :return: Total Mask.
        """

        width, height = self.svs.level_dimensions[para.Parameter.level]
        total_masks = [np.zeros((height, width), dtype=np.uint8)] * len(imagelist)

        for i, batch_coords in enumerate(self.points):
            
            for j, coord in enumerate(batch_coords):

                for n, total_mask in enumerate(total_masks):

                    total_mask[coord[0]:coord[0]+self.crop_size_height, coord[1]:coord[1]+self.crop_size_width] += imagelist[n][i][j]

        return total_masks