import os
import pickle

import torch
import torchvision
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

from mmcv import Config


def load_model(checkpoint: str, config: str = '../configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_norway.py'):
    cfg = Config.fromfile(config)
    if 'roi_head' in cfg.model:
        cfg.model.roi_head.bbox_head.num_classes = 4
    else:
        cfg.model.bbox_head.num_classes = 4
    # download the checkpoint from model zoo and put it in `checkpoints/`
    # url: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    # checkpoint_file = './checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, checkpoint, device='cuda:0')
    return model


def load_images(path):
    return [path+filename for filename in os.listdir(path) if filename.endswith(".jpg")]


def generate_predictions(model, images):
    return tqdm((inference_detector(model, img) for img in images), total=len(images), desc="Generating predictions")


def _format_bounding_box(box, category):
    x_min, y_min, x_max, y_max = box
    # Round floats to the nearest integer
    x_min, y_min, x_max, y_max = round(x_min), round(y_min), round(x_max), round(y_max)
    return f"{category+1} {x_min} {y_min} {x_max} {y_max}"


def _scale_bounding_box(box):
    pass


def _swap_categories(category):
    return {
        0: 0,  # D00: Longitudinal Crack
        3: 1,  # D10: Transverse Crack
        1: 2,  # D20: Alligator Crack
        2: 3   # D40: Potholes
    }[category]


def format_predictions(image_name, prediction, confidence_threshold=0.5, nms_threshold=0.1):
    # Format predictions for one image
    desired_predictions = []
    for category, cls in enumerate(prediction):
        cls_nms = torchvision.ops.nms(torch.from_numpy(cls[:, :4]), torch.from_numpy(cls[:, 4]), iou_threshold=nms_threshold)
        cls = cls[cls_nms]  # Update cls with non-maximum suppression
        if len(cls.shape) == 1:
            cls = cls.reshape(1, -1)

        for row in range(cls.shape[0]):
            box = cls[row]
            if box[4] >= confidence_threshold:  # Check if confidence is above threshold
                desired_predictions.append((_swap_categories(category), box))

    # Sort desired_predictions by confidence
    desired_predictions.sort(key=lambda x: x[1][4], reverse=True)
    # Only take the top 5 predictions
    return f"{image_name.split('/')[-1]},{' '.join([_format_bounding_box(box[:4], category) for category, box in desired_predictions[:5]])}"


if __name__ == '__main__':
    _run = "Run3/"
    _epoch = "epoch_13.pth"
    _load_results = True
    _confidence_threshold = 0.30
    _nms_threshold = 0.35
    _config = '../configs/focalnet/atss_focalnet_tiny_patch4_fpn_3x_norway_lrf.py'  # Run3
    # _config = '../configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_norway.py'

    """
    R003_13: ct=0.1500, nms=1.00, F1-0.28190820275241335
    R003_13: ct=0.2000, nms=1.00, F1-0.3115529630527306
    R003_13: ct=0.2500, nms=1.00, F1-0.3233222498795245
    R003_13: ct=0.2750, nms=1.00, F1-0.32482788114836997
    R003_13: ct=0.2875, nms=1.00, F1-0.3245521522451119
    R003_13: ct=0.3000, nms=1.00, F1-0.32689806337139465
    R003_13: ct=0.3250, nms=1.00, F1-0.3107806607005156
    R003_13: ct=0.3500, nms=1.00, F1-0.2906885362664345
    R003_13: ct=0.3000, nms=0.05, F1-0.3255643363526942
    R003_13: ct=0.3000, nms=0.50, F1-0.32698125293836233
    R003_13: ct=0.3000, nms=0.45, F1-0.32703329903094513
    R003_13: ct=0.3000, nms=0.40, F1-0.32707719801696217
    R003_13: ct=0.3000, nms=0.35, F1-
    R003_13: ct=0.3000, nms=0.30, F1-0.326...
    
    R002_11: ct=0.50, nms=0.10, F1-0.312547345774213
    R002_12: ct=0.50, nms=0.10, F1-0.313777979520287
    R002_12: ct=0.55, nms=0.10, F1-0.3151471549551368
    R002_12: ct=0.55, nms=0.15, F1-0.31552874317006
    
    R004_34: ct=0.55, nms=0.15, F1-0.3209058025826765
    R004_34: ct=0.60, nms=0.15, F1-0.32139392960280655
    R004_34: ct=0.60, nms=0.20, F1-0.3223511159635064
    R004_34: ct=0.55, nms=1.00, F1-0.32583510564207874
    R004_34: ct=0.59, nms=1.00, F1-0.32543713860674545
    R004_34: ct=0.60, nms=1.00, F1-0.3261951245324745
    R004_34: ct=0.61, nms=1.00, F1-0.32482352086307004
    R004_34: ct=0.65, nms=1.00, F1-0.32473977704165047
    
    R004_33: ct=0.60, nms=1.00, F1-
    """

    _images = load_images('../data/norway/test/')
    # _images = _images[:10]
    print("Images loaded")

    if not _load_results:
        _model = load_model(checkpoint=f'../checkpoints/runs/{_run}{_epoch}', config=_config)
        print("Model loaded")

        _results = generate_predictions(_model, _images)

        # Save predictions list to file using pickle
        with open(f'../checkpoints/runs/{_run}predictions_{_epoch}.pkl', 'wb') as f:
            # Use pickle
            pickle.dump(list(_results), f)
        print("Predictions generated")
    else:
        # Load predictions list from file using pickle
        with open(f'../checkpoints/runs/{_run}predictions_{_epoch}.pkl', 'rb') as f:
            _results = pickle.load(f)
        print("Predictions loaded")

    _output = tqdm((format_predictions(image, result, confidence_threshold=_confidence_threshold, nms_threshold=_nms_threshold)
                    for image, result in zip(_images, _results)), total=len(_images), desc="Formatting predictions")
    print("Predictions formatted")

    # Write to file
    with open(f'../checkpoints/runs/{_run}submission.txt', 'w') as f:
        f.write("\n".join(_output))
