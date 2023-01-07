import argparse
import logging

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import torch

import cv2
import numpy as np

from p2p.config import add_p2p_config

def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_p2p_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        type=str
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    model = build_model(cfg)
    model.eval()

    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    input_format = cfg.INPUT.FORMAT

    original_image = cv2.imread(args.input)

    with torch.no_grad():
        if input_format == "RGB":
            original_image = original_image[:, :, ::-1]
        
        height, width = original_image.shape[:2]
        height = (height // 128 + 1) * 128
        width = (width // 128 + 1) * 128
        original_image = cv2.resize(original_image, (width, height))

        image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        predictions = model([inputs])[0]
        points = predictions['instances'].points.detach().cpu().numpy()

    dis_img = np.copy(original_image)
    for point in points:
        point_size = 2
        dis_img = cv2.circle(dis_img, (int(point[0]), int(point[1])), point_size, (0, 0, 255), -1)

    cv2.imshow('Points', dis_img)
    cv2.waitKey()
