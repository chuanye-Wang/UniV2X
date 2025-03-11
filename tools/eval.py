# import argparse
# import os.path as osp
# import mmcv
# from mmcv import Config, DictAction
# from projects.mmdet3d_plugin.datasets.bui import build_dataset
# import time
import argparse
import cv2
import torch
import sklearn
import mmcv
import os
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.univ2x.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(description='3D Detection Evaluation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--result', required=True, help='result file path')
    parser.add_argument('--eval', type=str, nargs='+', required=True, 
                       help='evaluation metrics')
    parser.add_argument('--eval-options', nargs='+', action=DictAction)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()

def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    dataset = build_dataset(cfg.data.test)
    
    print(f'Loading results from {args.result}')
    outputs = mmcv.load(args.result)
    
    kwargs = args.eval_options or {}
    kwargs['jsonfile_prefix'] = osp.join('eval_results', 
                                       osp.basename(args.config).split('.')[0],
                                       time.strftime('%Y%m%d_%H%M%S', time.localtime()))
    
    eval_kwargs = cfg.get('evaluation', {}).copy()
    eval_kwargs.update({
        'metric': args.eval,
        **kwargs
    })
    
    print(dataset.evaluate(outputs, **eval_kwargs))

if __name__ == '__main__':
    main()