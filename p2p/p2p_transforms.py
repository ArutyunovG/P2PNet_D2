from fvcore.transforms.transform import Transform

import cv2

import random

class ShapeTransform(Transform):

    def __init__(
        self,
        scale_range,
        patch_size,
        flip
    ):
        super().__init__()

        self.scale_range = scale_range
        self.patch_size = patch_size
        self.flip = flip

        self._remembered_scale = None
        self._remembered_patch_coords = None
        self._did_flip = None


    def apply_image(self, img):

        scale = random.uniform(*self.scale_range)

        do_flip = (random.random() < self.flip)

        self._remembered_scale = scale
        self._did_flip = do_flip

        # scale the image and points
        if scale * min(img.shape[:2]) > self.patch_size:
            img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) 

        start_h = random.randint(0, img.shape[0] - self.patch_size)
        start_w = random.randint(0, img.shape[1] - self.patch_size)
        end_h = start_h + self.patch_size
        end_w = start_w + self.patch_size
        self._remembered_patch_coords = (start_h, start_w, end_h, end_w)
        
        img = img[start_h:end_h, start_w:end_w, :]

        if do_flip:
            img = img[:, ::-1, :]

        return img


    def apply_coords(self, coords):
        
        coords = coords * self._remembered_scale
        self._remembered_scale = None

        start_h, start_w, end_h, end_w = self._remembered_patch_coords
        idx = (coords[:, 0] >= start_w) & (coords[:, 0] <= end_w) & (coords[:, 1] >= start_h) & (coords[:, 1] <= end_h)
        coords = coords[idx]
        coords[:, 0] -= start_w
        coords[:, 1] -= start_h
        self._remembered_patch_coords = None

        if self._did_flip:
            coords[:, 0] = self.patch_size - coords[:, 0]
        self._did_flip = None

        return coords
