import numpy as np
import os, sys, tifffile
from utils.utils import *


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

    def get_contours(self, para):

        img = tifffile.TiffFile(para.Path.root_path).pages[3].asarray()
        h, w, _ = img.shape

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        dst = 255.0 - cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        dst = cv.dilate(dst.astype(np.uint8), kernel, iterations=1)

        mask_coords = np.zeros((h, w), dtype=np.uint8)
        contours, _ = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            area = cv.contourArea(contour, False)

            if area >= 1e3:
                x, y, w, h = cv.boundingRect(contour)
                mask_coords = cv.rectangle(mask_coords, (x, y), (x+w, y+h), 255, -1)

        contours_coords = cv.findContours(mask_coords, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        coords = []

        for contour in contours_coords:

            x, y, w, h = cv.boundingRect(np.dot(contour, 4**2))
            coords.append([x, y, w, h])

        return coords

    def get_image(self, para, i, j, level):

        i = i - self.overlap_size
        j = j - self.overlap_size

        w_min_border = 0-i if i < 0 else 0
        h_min_border = 0-j if j < 0 else 0

        w_max_border = (j+self.crop_size_width+self.overlap_size)-self.x-self.width if (j+self.crop_size_width+self.overlap_size) > self.x+self.width else 0
        h_max_border = (i+self.crop_size_height+self.overlap_size)-self.y-self.height if (i+self.crop_size_height+self.overlap_size) > self.y+self.height else 0

        if i < 0 and j < 0:
            image = np.array(self.svs.read_region((0, 0), level, (self.crop_size_height+self.overlap_size, self.crop_size_width+self.overlap_size)))[:, :, :3]
            flag = False if ((np.mean(image) < 245) and (np.mean(image) != 0)) else True
            image = cv.copyMakeBorder(image, w_min_border, w_max_border, h_min_border, h_max_border, cv.BORDER_REFLECT)
        
        elif i < 0 and j >= 0:
            image = np.array(self.svs.read_region((j*4**level, 0), level, (self.crop_size_height+2*self.overlap_size, self.crop_size_width+self.overlap_size)))[:, :, :3]
            flag = False if ((np.mean(image) < 245) and (np.mean(image) != 0)) else True
            image = cv.copyMakeBorder(image, w_min_border, w_max_border, h_min_border, h_max_border, cv.BORDER_REFLECT)
        
        elif i >= 0 and j < 0:
            image = np.array(self.svs.read_region((0, i*4**level), level, (self.crop_size_height+self.overlap_size, self.crop_size_width+2*self.overlap_size)))[:, :, :3]
            flag = False if ((np.mean(image) < 245) and (np.mean(image) != 0)) else True
            image = cv.copyMakeBorder(image, w_min_border, w_max_border, h_min_border, h_max_border, cv.BORDER_REFLECT)
        
        else:
            image = np.array(self.svs.read_region((j*4**level, i*4**level), level, (self.crop_size_height+2*self.overlap_size, self.crop_size_width+2*self.overlap_size)))[:, :, :3]
            flag = False if ((np.mean(image) < 245) and (np.mean(image) != 0)) else True

        if para.Parameter.trans_channel:
            return image[:, :, ::-1], flag

        else:
            return image, flag

    def crop_Image(self, para):

        """
        Crop Image.
        :param image_shape: Image shape.
        :return: Patchs coords.
        """

        self.points = []

        if para.Model.roi_label[-1]:

            shapes = read_geojson(para.Path.gt_json_path)
            roi_contours = get_roi_conts(shapes, para.Model.roi_label[0])[0]
            self.x, self.y, w, h = cv.boundingRect(roi_contours)
            self.height, self.width = h, w

            for i in range(self.y, self.y+self.height, self.crop_stride):

                if i+self.crop_size_height > self.y+self.height:
                    i = self.y + self.height - self.crop_size_height

                for j in range(self.x, self.x+self.width, self.crop_stride):

                    if j+self.crop_size_width > self.x+self.width:
                        j = self.x + self.width - self.crop_size_width
                            
                    self.points.append([i, j])

            return self.points

        else:

            roi_coord = self.get_contours(para)

            for (self.x, self.y, self.width, self.height) in roi_coord:

                for i in range(self.y, self.y+self.height, self.crop_stride):

                    if i+self.crop_size_height > self.y+self.height:
                        i = self.y + self.height - self.crop_size_height

                    for j in range(self.x, self.x+self.width, self.crop_stride):

                        if j+self.crop_size_width > self.x+self.width:
                            j = self.x + self.width - self.crop_size_width
                                
                        self.points.append([i, j])

            return self.points
        
    def merge_Image(self, para, imagelist):

        """
        Merageg Image.
        :param imagelist: Patch list.
        :return: Total Mask.
        """

        w, h = self.svs.level_dimensions[0]
        width, height = self.svs.level_dimensions[para.Parameter.level]
        total_masks = [np.zeros((height, width), dtype=np.uint8)] * len(imagelist)

        for i, batch_coords in enumerate(self.points):
            
            for j, coord in enumerate(batch_coords):

                for n, total_mask in enumerate(total_masks):

                    total_mask[coord[0]:coord[0]+self.crop_size_height, coord[1]:coord[1]+self.crop_size_width] += imagelist[n][i][j]
        
        if para.Parameter.level > 0:

            for total_mask in total_masks:
                total_mask = cv.resize(total_mask, (h, w))

        cv.imwrite('mask.png', cv.resize(total_masks[0], dsize=None, fx=1/8, fy=1/8))

        return total_masks