"""Faster R-CNN baseline (torchvision) with anchors resized for small defects."""
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .common import CLASSES


def build_model(pretrained=True, num_classes=len(CLASSES)):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights, weights_backbone=None, min_size=800, max_size=800,  # 640px tiles are upscaled 1.25x
        box_score_thresh=0.001, box_detections_per_img=100,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 background
    # Default anchors start at 32px; PCB defects are often 15-40px. Same anchor count per
    # location (3 ratios), so the pretrained RPN head shapes still load.
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    return model
