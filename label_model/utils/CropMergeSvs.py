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

    def crop_Image(self, image, overlap_size, batch_size):

        """
        Crop Image.
        :param image_shape: Image shape.
        :return: Patchs coords.
        """

        self.points = []
        self.image = image
        self.overlap_size = overlap_size

        self.height, self.width = image.shape[:2]

        print(f'self.height is: {self.height}, self.width is: {self.width}')

        counter = 0
        patch_imgs = []

        for i in range(0, self.height, self.crop_stride):

            if i+self.crop_size_height+(2*overlap_size) > self.height:
                i = self.height - self.crop_size_height - (2*overlap_size)

            for j in range(0, self.width, self.crop_stride):

                if j+self.crop_size_width+(2*overlap_size) > self.width:
                    j = self.width - self.crop_size_width - (2*overlap_size)

                if counter == batch_size-1:
                    counter = 0
                    batch_coord.append([i, j])
                    batch_imgs.append(image[i:i+self.crop_size_height+(2*overlap_size), j:j+self.crop_size_width+(2*overlap_size)])
                    self.points.append(batch_coord)
                    patch_imgs.append(batch_imgs)
                
                elif counter == 0:
                    counter += 1
                    batch_coord = [[i, j]]
                    batch_imgs = [image[i:i+self.crop_size_height+(2*overlap_size), j:j+self.crop_size_width+(2*overlap_size)]]
                
                else:
                    counter += 1
                    batch_coord.append([i, j])
                    batch_imgs.append(image[i:i+self.crop_size_height+(2*overlap_size), j:j+self.crop_size_width+(2*overlap_size)])

        return patch_imgs

    def merge_Image(self, imagelist):

        """
        Merageg Image.
        :param imagelist: Patch list.
        :return: Total Mask.
        """

        total_masks = [np.zeros((self.image.shape[0]-(2*self.overlap_size), self.image.shape[1]-(2*self.overlap_size)), dtype=np.uint8)] * len(imagelist)

        for i, batch_coords in enumerate(self.points):
            
            for j, coord in enumerate(batch_coords):

                for n, total_mask in enumerate(total_masks):

                    total_mask[coord[0]:coord[0]+self.crop_size_height, coord[1]:coord[1]+self.crop_size_width] += imagelist[n][i][j]

        return total_masks