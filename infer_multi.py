# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder. Allows for using a combination of multiple models.
For example, one model may be used for RPN, another model for Fast R-CNN style
box detection, yet another model to predict masks, and yet another model to
predict keypoints.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import glob
import os
import sys
import yaml
import numpy as np
import pickle

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import load_cfg
from detectron.core.config import merge_cfg_from_cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
import detectron.core.rpn_generator as rpn_engine
import detectron.core.test_engine as model_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# infer.py
#   --im [path/to/image.jpg]
#   --rpn-model [path/to/rpn/model.pkl]
#   --rpn-config [path/to/rpn/config.yaml]
#   [model1] [config1] [model2] [config2] ...


def parse_args():
    parser = argparse.ArgumentParser(description='Inference on an image')
    parser.add_argument(
        '--im', dest='im_file', help='input image', default=None, type=str
    )
    parser.add_argument(
        '--rpn-pkl',
        dest='rpn_pkl',
        help='rpn model file (pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--rpn-cfg',
        dest='rpn_cfg',
        help='cfg model file (yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer)',
        default='/tmp/infer',
        type=str
    )
    parser.add_argument(
        'models_to_run',
        help='list of pkl, yaml pairs',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_rpn_box_proposals(im, args):
    cfg.immutable(False)
    merge_cfg_from_file(args.rpn_cfg)
    cfg.NUM_GPUS = 1
    cfg.MODEL.RPN_ONLY = True
    cfg.TEST.RPN_PRE_NMS_TOP_N = 10000
    cfg.TEST.RPN_POST_NMS_TOP_N = 2000
    assert_and_infer_cfg(cache_urls=False)

    model = model_engine.initialize_model_from_cfg(args.rpn_pkl)
    with c2_utils.NamedCudaScope(0):
        boxes, scores = rpn_engine.im_proposals(model, im)
    return boxes, scores


def main(args):
    logger = logging.getLogger(__name__)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    cfg_orig = load_cfg(yaml.dump(cfg))

    if os.path.isdir(args.im_file):
        im_list = glob.iglob(args.im_file + '/*.jpg')
    else:
        im_list = [args.im_file]
    
    for i, im_name in enumerate(im_list):

        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))

        im = cv2.imread(im_name)
        if args.rpn_pkl is not None:
            proposal_boxes, _proposal_scores = get_rpn_box_proposals(im, args)
            workspace.ResetWorkspace()
        else:
            proposal_boxes = None

        cls_boxes, cls_segms, cls_keyps, cls_bodys = None, None, None, None
        for i in range(0, len(args.models_to_run), 2):
            pkl = args.models_to_run[i]
            yml = args.models_to_run[i + 1]
            cfg.immutable(False)
            merge_cfg_from_cfg(cfg_orig)
            merge_cfg_from_file(yml)
            if len(pkl) > 0:
                weights_file = pkl
            else:
                weights_file = cfg.TEST.WEIGHTS
            cfg.NUM_GPUS = 1
            assert_and_infer_cfg(cache_urls=False)
            model = model_engine.initialize_model_from_cfg(weights_file)
            with c2_utils.NamedCudaScope(0):
                cls_boxes_, cls_segms_, cls_keyps_ , cls_bodys_= \
                    model_engine.im_detect_all(model, im, proposal_boxes)
            cls_boxes = cls_boxes_ if cls_boxes_ is not None else cls_boxes
            cls_segms = cls_segms_ if cls_segms_ is not None else cls_segms
            cls_keyps = cls_keyps_ if cls_keyps_ is not None else cls_keyps
            cls_bodys = cls_bodys_ if cls_bodys_ is not None else cls_bodys

            workspace.ResetWorkspace()

        f = open(os.path.join(args.output_dir,'test_vis_{}'.format(os.path.basename(im_name).split('.')[0])) + '.pkl' ,'w')
        pickle.dump({'im':im , 'cls_boxes':np.array(cls_boxes) , 'cls_bodys':np.array(cls_bodys), 'kp':np.array(cls_keyps) },f)
        f.close()

        vis_utils.vis_one_image(
            im[:, :, ::-1],
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            cls_bodys,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )


def check_args(args):
    assert (
        (args.rpn_pkl is not None and args.rpn_cfg is not None) or
        (args.rpn_pkl is None and args.rpn_cfg is None)
    )
    if args.rpn_pkl is not None:
        args.rpn_pkl = cache_url(args.rpn_pkl, cfg.DOWNLOAD_CACHE)
        assert os.path.exists(args.rpn_pkl)
        assert os.path.exists(args.rpn_cfg)
    if args.models_to_run is not None:
        assert len(args.models_to_run) % 2 == 0
        for i, model_file in enumerate(args.models_to_run):
            if len(model_file) > 0:
                if i % 2 == 0:
                    model_file = cache_url(model_file, cfg.DOWNLOAD_CACHE)
                    args.models_to_run[i] = model_file
                assert os.path.exists(model_file), \
                    '\'{}\' does not exist'.format(model_file)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    check_args(args)
    main(args)
