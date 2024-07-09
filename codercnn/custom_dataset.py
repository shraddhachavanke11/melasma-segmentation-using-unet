import os
import mrcnn

import numpy as np
import cv2
from mrcnn import utils

class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        # Add class and object names for melasma patches
        self.add_class("melasma_patches", 1, "melasma")

        # Add images
        image_dir = os.path.join(dataset_dir, "images")
        image_ids = os.listdir(image_dir)
        for image_id in image_ids:
            self.add_image("melasma_patches", image_id=image_id, path=os.path.join(image_dir, image_id))

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        mask_path = os.path.join(mask_dir, info['id'].replace(".jpg", "_mask.jpg"))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return (mask > 0).astype(np.uint8), np.array([1])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
