from p2p.p2p_transforms import ShapeTransform

import copy
import logging

from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.structures import Instances

import torch
import numpy as np

class P2PShapeTransform(T.TransformGen):
    def __init__(
        self,
        scale_range,
        patch_size,
        flip
    ):
        super().__init__()
        self._init(locals())


    def get_transform(self, img):
        return ShapeTransform(
            scale_range=self.scale_range,
            patch_size=self.patch_size,
            flip=self.flip
        )


class DatasetMapper:


    def __init__(self, cfg, is_train=True):

        assert not cfg.MODEL.MASK_ON
        assert not cfg.MODEL.KEYPOINT_ON
        assert not cfg.MODEL.LOAD_PROPOSALS

        self.is_train = is_train

        self.img_format = cfg.INPUT.FORMAT
        self.mask_format = cfg.INPUT.MASK_FORMAT

        if self.is_train:
            self.tfm_gens = [
                P2PShapeTransform(
                    scale_range=cfg.INPUT.P2PNET.SCALE_RANGE,
                    patch_size=cfg.INPUT.P2PNET.PATCH_SIZE,
                    flip=cfg.INPUT.P2PNET.FLIP
                )
            ]
        else:
            self.tfm_gens = []

        logging.getLogger(__name__).warning(str(self.tfm_gens))


    def __call__(self, dataset_dict):

        dataset_dict = copy.deepcopy(dataset_dict) 
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        assert "annotations"  in dataset_dict

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w


        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        annos = dataset_dict.pop("annotations")
        coords = np.vstack([anno['point'] for anno in annos])
        annos = [
            {
                'category_id': 1,
                'point': (coord[0], coord[1])
            }
            for coord in transforms.apply_coords(coords)
        ]

        target = Instances(image_shape)
        points = [obj["point"] for obj in annos]
        target.gt_points = torch.tensor(points, dtype=torch.float32)

        classes = [int(obj["category_id"]) for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes

        dataset_dict["instances"] = target

        return dataset_dict
