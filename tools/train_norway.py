from mmcv import Config
from mmdet.apis import set_random_seed
import mmdet.datasets.norway
import mmdet.datasets.kitti

if __name__ == '__main__':
    # Run with: python tools/train_norway.py

    # cfg = Config.fromfile('../configs/focalnet/mask_rcnn_focalnet_base_patch4_mstrain_480-800_adamw_3x_coco_lrf.py')
    # TODO : Not enough memory for this focalnet model :(
    # cfg = Config.fromfile('./configs/focalnet/mask_rcnn_focalnet_base_patch4_mstrain_480-800_adamw_3x_norway_lrf.py')
    cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_norway.py')
    # cfg = Config.fromfile('./configs/focalnet/atss_focalnet_tiny_patch4_fpn_3x_norway_lrf.py')

    # For custom small coco
    # cfg.data_root = './data/small_coco/'
    # cfg.data.train.data_root = './data/small_coco/'
    # cfg.data.test.data_root = './data/small_coco/'
    # cfg.data.val.data_root = './data/small_coco/'
    # cfg.data.train.ann_file = 'annotations/instances_train2017_small.json'
    # cfg.data.test.ann_file = 'annotations/instances_val2017_small.json'
    # cfg.data.val.ann_file = 'annotations/instances_val2017_small.json'
    # cfg.data.train.img_prefix = 'train2017/'
    # cfg.data.test.img_prefix = 'val2017/'
    # cfg.data.val.img_prefix = 'val2017/'

    # For custom small norway
    # cfg.dataset_type = 'NorwayDataset'
    cfg.data_root = './data/norway/'
    cfg.data.train.data_root = './data/norway/'
    cfg.data.test.data_root = './data/norway/'
    cfg.data.val.data_root = './data/norway/'
    cfg.data.train.ann_file = 'annotations/train_norway_normal.json'
    cfg.data.test.ann_file = 'annotations/val_norway.json'
    cfg.data.val.ann_file = 'annotations/val_norway.json'
    cfg.data.train.img_prefix = 'train/'
    cfg.data.test.img_prefix = 'val/'
    cfg.data.val.img_prefix = 'val/'

    # modify num classes of the model in box head and mask head
    cfg.model.roi_head.bbox_head.num_classes = 4  # TODO : For the faster rcnn model
    # cfg.model.roi_head.mask_head.num_classes = 4
    # cfg.model.bbox_head.num_classes = 4  # TODO : For the ATSS model
    print(cfg.model)
    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    # cfg.load_from = './checkpoints/runs/Run2/epoch_12.pth'
    cfg.runner.max_epochs = 36
    cfg.load_from = 'demo/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
    # cfg.load_from = './checkpoints/focalnet_tiny_lrf_atss_3x.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = './checkpoints/runs/'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = 0.02 / 8  # TODO : This is default
    # cfg.optimizer.lr = 0.0001 / 8  # 0.0001
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 100

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'bbox'  # TODO : Think things need to be changed in the Dataset class to use mAP
    # TODO : Think this should be 'bbox' and some items should be mAP instead? Look in coco.py
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 1
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 1

    # Set this to 1 to make training easier?
    # cfg.data.workers_per_gpu = 1
    #
    # # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # TODO : Only available in MMDet v2.25.0 and above
    # cfg.log_config.hooks.append(
    #     dict(type='MMDetWandbHook',
    #          init_kwargs={'project': 'MMDetection-tutorial'},
    #          interval=10,
    #          log_checkpoint=False,
    #          log_checkpoint_metadata=True,
    #          num_eval_images=100)
    # )

    # # We can initialize the logger for training and have a look
    # # at the final config used for training
    # print(f'Config:\n{cfg.pretty_text}')

    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.apis import train_detector
    import mmcv
    import os.path as osp

    # TODO : List of things I need to do:
    #  1. Make the dataset smaller to make it easier to work on. Filter out most of the samples.
    #  2. Figure out why the other dataset is not working by changing the working dataset (i.e. is it because of segmentation?).
    #    I would preferrably make the dataset using the exact code from CocoDataset.
    #  3. Make it run on the other model (i.e. focalnet).

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'))  # , test_cfg=cfg.get('test_cfg')
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=False)
