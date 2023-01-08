import argparse
import logging

import torch
from torch import nn

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from p2p.config import add_p2p_config

import onnx
from onnxsim import simplify

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
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--output",
        default="model.onnx",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

class ExportModelWrapper(nn.Module):

    def __init__(self, p2p_model):
        super().__init__()
        self._p2p_model = p2p_model

    def forward(self, x):

        x = self._p2p_model.normalizer(x)
        features = self._p2p_model.backbone(x)

        features_fpn = self._p2p_model.fpn([features[idx] for idx in self._p2p_model.backbone_features_to_decoder])

        regression = self._p2p_model.regression(
                           features_fpn[self._p2p_model.decoder_feature_to_branches]) * self._p2p_model.regression_scale_coeff
        classification = self._p2p_model.classification(features_fpn[self._p2p_model.decoder_feature_to_branches])
        classification = torch.nn.functional.softmax(classification, -1)

        return regression, classification


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    model = build_model(cfg)
    model = model.to(torch.device('cpu'))
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    model = ExportModelWrapper(model)

    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))

    torch.onnx.export(model,
                      torch.randn(1, 3, 256, 256, requires_grad=True),
                      args.output,
                      input_names = ['input'], 
                      output_names = ['regression', 'classification'])

    model = onnx.load(args.output)
    model_simp, check = simplify(model)

    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, args.output)
