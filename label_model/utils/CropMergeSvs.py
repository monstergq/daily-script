import os, sys
import numpy as np


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

    def crop_Image(self, image):

        """
        Crop Image.
        :param image_shape: Image shape.
        :return: Patchs coords.
        """

        self.points = []
        self.image = image

        self.height, self.width = image.shape[:2]

        print(f'self.height is: {self.height}, self.width is: {self.width}')

        patch_imgs = []

        for i in range(0, self.height, self.crop_stride):

            if i+self.crop_size_height > self.height:
                i = self.height - self.crop_size_height

            for j in range(0, self.width, self.crop_stride):

                if j+self.crop_size_width > self.width:
                    j = self.width - self.crop_size_width

                self.points.append([i, j])

                # I = i-128 if i-128>=0 else 0
                # J = j-128 if j-128>=0 else 0
                # I_ = I+self.crop_size_height+256 if I+self.crop_size_height < self.height else self.height - self.crop_size_height
                # J_ = J+self.crop_size_width+256 if J+self.crop_size_width < self.width else self.width - self.crop_size_width

                # patch_imgs.append(image[I:I_, J:J_])
                patch_imgs.append(image[i:i+self.crop_size_height, j:j+self.crop_size_width])

        return patch_imgs

    def merge_Image(self, imagelist):

        """
        Merageg Image.
        :param imagelist: Patch list.
        :return: Total Mask.
        """

        total_masks = [np.zeros(self.image.shape[:2], dtype=np.uint8)] * len(imagelist)

        for i, coord in enumerate(self.points):
            
            for j, total_mask in enumerate(total_masks):
                
                total_mask[coord[0]:coord[0]+self.crop_size_height, coord[1]:coord[1]+self.crop_size_width] += imagelist[j][i]

        return total_masks