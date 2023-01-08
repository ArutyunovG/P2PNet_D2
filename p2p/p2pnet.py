from p2p.anchor_points import AnchorPoints
from p2p.classification_model import ClassificationModel
from p2p.decoder import Decoder
from p2p.p2p_loss import P2PCriterion
from p2p.p2p_matcher import P2PHungarianMatcher
from p2p.regression_model import RegressionModel

from detectron2.modeling.backbone import build_backbone
from detectron2.structures import ImageList, Instances
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

import torch
from torch import nn

# the defenition of the P2PNet model

@META_ARCH_REGISTRY.register()
class P2PNet(nn.Module):

    def __init__(self,
                 cfg):

        super().__init__()

        self.backbone = build_backbone(cfg)
        self.num_classes = cfg.MODEL.P2PNET.NUM_CLASSES

        rows, lines = cfg.MODEL.P2PNET.ROWS, cfg.MODEL.P2PNET.LINES

        num_decoder_features = cfg.MODEL.P2PNET.DECODER_FEATURE_SIZE

        # the number of all anchor points
        num_anchor_points = rows * lines
        self.regression = self.build_regresion_model(num_decoder_features, num_anchor_points)
        self.classification = self.build_classification_model(num_decoder_features, num_anchor_points)

        self.anchor_points = self.build_anchor_points_model(cfg.MODEL.P2PNET.ANCHOR_PYRAMID_LEVELS, rows, lines)

        self.fpn = self.build_decoder(cfg.MODEL.P2PNET.FEATURE_SIZES,
                                      num_decoder_features)

        self.backbone_features_to_decoder = cfg.MODEL.P2PNET.BACKBONE_FEATURES_TO_DECODER
        self.decoder_feature_to_branches = cfg.MODEL.P2PNET.DECODER_FEATURE_TO_BRANCHES

        self.class_loss_weight = cfg.MODEL.P2PNET.CLASS_LOSS_WEIGHT
        self.point_loss_weight = cfg.MODEL.P2PNET.POINT_LOSS_WEIGHT
        self.p2p_loss = P2PCriterion(
            P2PHungarianMatcher(
                cfg.MODEL.P2PNET.COST_CLASS,
                cfg.MODEL.P2PNET.COST_POINT
            ),
            cfg.MODEL.P2PNET.EMPTY_CLASS_LOSS_WEIGHT,
            self.num_classes
        )

        self.regression_scale_coeff = cfg.MODEL.P2PNET.REGRESSION_SCALE_COEFF
        self.threshold = cfg.MODEL.P2PNET.THRESHOLD

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std
        self.device = cfg.MODEL.DEVICE


    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def forward(self, batched_inputs):

        images = self.preprocess_image(batched_inputs)

        # get the backbone features
        features = self.backbone(images.tensor)
        # forward the feature pyramid
        features_fpn = self.fpn([features[idx] for idx in self.backbone_features_to_decoder])

        batch_size = features[0].shape[0]
        # run the regression and classification branch
        regression = self.regression(features_fpn[self.decoder_feature_to_branches]) * self.regression_scale_coeff
        classification = self.classification(features_fpn[self.decoder_feature_to_branches])
        anchor_points = self.anchor_points(images.tensor).repeat(batch_size, 1, 1)
        # decode the points as prediction
        output_coord = regression + anchor_points
        output_class = classification
       
        if self.training:

            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"

            outputs = {'pred_logits': output_class, 'pred_points': output_coord}
            targets = [
                {
                    'point': bi['instances'].gt_points.to(self.device),
                    'labels': bi['instances'].gt_classes.to(self.device)
                }
                for bi in batched_inputs
            ]

            losses = self.p2p_loss(outputs, targets)
            losses['loss_ce'] *= self.class_loss_weight
            losses['loss_point'] *= self.point_loss_weight
            return losses

        else:

            processed_results = []
            for img_idx, image_size in enumerate(images.image_sizes):

                output_scores = torch.nn.functional.softmax(output_class, -1)[:, :, 1][img_idx]
                score_mask = output_scores > self.threshold
                output_scores = output_scores[score_mask]
                output_points = output_coord[img_idx][score_mask]

                result = Instances(image_size)
                result.scores = output_scores.view(-1)
                result.points = output_points

                processed_results.append({'instances': result})

                return processed_results


    def build_anchor_points_model(self, pyramid_levels, rows, lines):
        return AnchorPoints(pyramid_levels=pyramid_levels, row=rows, line=lines)


    def build_classification_model(self, num_decoder_features, num_anchor_points):
        return ClassificationModel(num_features_in=num_decoder_features,
                                   num_classes=self.num_classes,
                                   num_anchor_points=num_anchor_points)


    def build_decoder(self, input_feature_sizes, feature_size):
        return Decoder(input_feature_sizes, feature_size)


    def build_regresion_model(self, num_decoder_features, num_anchor_points):
        return RegressionModel(num_features_in=num_decoder_features, num_anchor_points=num_anchor_points)
