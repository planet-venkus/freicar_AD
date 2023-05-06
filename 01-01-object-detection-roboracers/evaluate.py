"""
COCO-Style Evaluations
"""

import argparse
import torch
import yaml
from tqdm import tqdm
from model.efficientdet.backbone import EfficientDetBackbone
from model.efficientdet.utils import BBoxTransform, ClipBoxes
from utils import postprocess, boolean_string
from dataloader.freicar_dataloader import FreiCarDataset
from model.efficientdet.dataset import collater
from torch.utils.data import DataLoader
import numpy as np
from utils import display
from utils import visualize_imgs_with_gt_bbox
from model.efficientdet.loss import calc_iou
from torchvision.ops.boxes import box_iou

########################################################################
# Object Detection model evaluation script
# Modified by: Jannik Zuern (zuern@informatik.uni-freiburg.de)
########################################################################


ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='freicar-detection', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5,
                help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device

project_name = args.project
weights_path = args.weights

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

threshold = 0.2
iou_threshold = 0.2

if __name__ == '__main__':

    '''
    Note: 
    When calling the model forward function on an image, the model returns
    features, regression, classification and anchors.
    
    In order to obtain the final bounding boxes from these predictions, they need to be postprocessed
    (this performs score-filtering and non-maximum suppression)
    
    Thus, you should call
    

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    preds = postprocess(imgs, anchors, regression, classification, regressBoxes, clipBoxes, threshold, nms_threshold)                  
    preds = preds[0]

    Now, the scores, class_indices and bounding boxes are saved as fields in the preds dict and can be used for subsequent evaluation.
    '''

    set_name = 'validation'

    freicar_dataset = FreiCarDataset(data_dir="./dataloader/data/",
                                     padding=(0, 0, 12, 12),
                                     split=set_name,
                                     load_real=False)
    val_params = {'batch_size': 1,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': 1}

    freicar_generator = DataLoader(freicar_dataset, **val_params)

    # instantiate model
    model = EfficientDetBackbone(compound_coef=compound_coef,
                                 num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']),
                                 scales=eval(params['anchors_scales']))
    # load model weights file from disk
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    ##########################################
    # TODO: implement me!
    ##########################################

    iou_list = []
    ap_list = []

    # metric_fn = MeanAveragePrecision(num_classes=1)

    for iter, data in enumerate(tqdm(freicar_generator)):
        with torch.no_grad():
            imgs = data['img']
            annot = data['annot']

            _, regression, classification, anchors = model(imgs)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            preds = postprocess(imgs, anchors, regression, classification, regressBoxes, clipBoxes, threshold,
                                nms_threshold)

            # preds_with_gt = preds
            # gt_rois = preds[0]['rois']
            # if gt_rois.size == 0:
            #     gt_rois = gt_rois.reshape(0, 4)
            # # print(f"Annots length= {len(annot[0][:, :4].numpy())}")
            # annot_len = len(annot[0][:, :4].numpy())
            # preds_with_gt[0]['rois'] = np.vstack((gt_rois, annot[0][:, :4].numpy()))
            # preds_with_gt[0]['class_ids'] = np.hstack((preds[0]['class_ids'], np.ones(annot_len))).astype(int)
            # preds_with_gt[0]['scores'] = np.hstack((preds[0]['scores'], np.ones(annot_len)))
            # obj_list_with_gt = ['freicar', 'Ground Truth']

            # print(f"Preds = {preds_with_gt}")
            # print(f"Annots = {annot[0][:, :4].numpy()}")

            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            # display(preds_with_gt, imgs, imshow=True, imwrite=False, obj_list=obj_list_with_gt)

            preds = preds[0]

            mean_IoU_perImage = -1
            map_perImage = -1
            gt_np = annot[0][:, :4].numpy()
            gt_no_box = np.empty_like(gt_np)
            gt_no_box.fill(-1)
            if torch.from_numpy(preds['rois']).shape != torch.Size([0]):
                iou = box_iou(torch.from_numpy(preds['rois']), annot[0][:, :4])
                iou_threshold = 0.5
                iou_np = iou.numpy()
                iou_filtered = iou_np[iou_np > iou_threshold]  # to produce prediction list
                iou_thresh = iou_np
                iou_thresh[
                    iou_thresh < iou_threshold] = 0  # to keep the matrix form of IoUs intact for calculation of mAP over rows of boxes

                if iou_filtered.size != 0:  # to check if there is any predicted box
                    max_ious_alongRows = np.amax(iou_thresh, axis=1)  # use the best prediction as the true positive
                    mAP_num = max_ious_alongRows[max_ious_alongRows > 0].size   # only keep the non-zero terms
                    mAP_denom = iou_thresh[iou_thresh > 0].size     # all the predictions(includes true positives + false positives)
                    mean_IoU_perImage = sum(iou_thresh[iou_thresh > 0]) / mAP_denom    # mean of filtered non-zero IoU's
                    map_perImage = mAP_num / mAP_denom

            else:  # if our prediction says no bounding box but ground truth has a bounding box, then append both mIoU and mAP as 0.
                if not np.array_equal(gt_np, gt_no_box):
                    mean_IoU_perImage = 0
                    map_perImage = 0
                elif np.array_equal(gt_np, gt_no_box):
                    map_perImage = 1

            iou_list.append(mean_IoU_perImage)
            ap_list.append(map_perImage)

    iou_list = [i for i in iou_list if i != -1]
    ap_list = [i for i in ap_list if i != -1]
    mIoU = sum(iou_list) / len(iou_list)
    map = sum(ap_list) / len(ap_list)
    print(f"mIoU = {mIoU}")
    print(f"mAP: {map}")

    # for future reference:
    # c_id = np.expand_dims(preds['class_ids'], axis=0).T
    # s_id = np.expand_dims(preds['scores'], axis=0).T
    # preds_list = np.hstack((preds['rois'], c_id, s_id))
    # preds_list = [preds['rois'], preds['class_ids'], preds['scores']]
    # gt_list = annot[0].numpy()
    # shp = (gt_list.shape[0], 2)
    # gt_list = np.hstack((gt_list, np.zeros(shp)))
    # gt_list = gt_list.extend([0,0])
    # print(f"Our gt list {gt_list}")
    # print(f"Our preds list class { preds}")

    # create metric_fn
    # metric_fn.add(preds2, gt2)
