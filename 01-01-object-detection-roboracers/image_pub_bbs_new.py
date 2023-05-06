#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import cv2
import numpy as np
import argparse
import yaml
import torch
import os
import sys
from rospy_tutorials.msg import Floats
sys.path.append('/home/freicar/freicar_ws/src/freicar-2020-exercises/01-01-object-detection-exercise/')
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float32MultiArray
from model.efficientdet.backbone import EfficientDetBackbone
from model.efficientdet.utils import BBoxTransform, ClipBoxes

from utils import postprocess, boolean_string
# , STANDARD_COLORS, standard_to_bgr
from utils import display
from dataloader.freicar_dataloader import FreiCarDataset
from model.efficientdet.dataset import collater
from torch.utils.data import DataLoader

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

obj_list = ['freicar']
params = yaml.safe_load(open(f'projects/{project_name}.yml'))

def img_callback(msg):
    np_img = np.fromstring(msg.data, dtype=np.uint8).reshape((720, 1280, 3))
    bgr = np.zeros((np_img.shape[0], np_img.shape[1], 3), dtype=np.uint8)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR, bgr, 3)
    rgb = np.zeros((np_img.shape[0], np_img.shape[1], 3), dtype=np.uint8)
    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, rgb, 3)
    rgb = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
    rgb = cv2.copyMakeBorder(rgb, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    rgb = (torch.from_numpy(rgb)).permute(2, 0, 1)
    rgb_tensor = torch.unsqueeze(rgb, 0)
    rgb_tensor = rgb_tensor.cuda().float()
    # Feeding the tensor into the model
    bbs_preds = generate_predictionBox(rgb_tensor)
    # print(f"Output of generate_predictionBox = {bbs_preds}")
    #for further use
    # print(f"rgb_tensor shape = {rgb_tensor.shape}")
    bbs_preds = bbs_preds[0]
    bbs_coords = bbs_preds['rois']
    bbs_scores = bbs_preds['scores']
    valid_bbs_index = np.argwhere(bbs_scores > 0.85)
    if len(valid_bbs_index) != 0:
        bbs_coords = [bbs_coords[i] for i in valid_bbs_index]
    # print(f"Valid bb = {bbs_coords}")
    pub = rospy.Publisher('/bbsinfo', Float32, queue_size=10)
    rate = rospy.Rate(10)  # 10hz
    # print('bbs', bbs)
    pub.publish(bbs_coords)

def start_node():
    rospy.init_node('bbs_publisher')
    rospy.loginfo('bbs_publisher node started')
    img_sub = rospy.Subscriber('/freicar_1/sim/camera/rgb/front/image', Image, callback=img_callback, queue_size=10)
    rospy.spin()

def generate_predictionBox(imgs):

    # obj_list = params['obj_list']
    threshold = 0.2
    imgs = imgs.float()
    with torch.no_grad():
        features, regression, classification, anchors = model(imgs)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    preds = postprocess(imgs, anchors, regression, classification, regressBoxes, clipBoxes, threshold,
                        nms_threshold)
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    imgs = (imgs * 255).astype(np.uint8)
    imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
    display(preds, imgs, imshow=True, imwrite=False, obj_list=obj_list)
    return preds

if __name__ == '__main__':
    try:
        model = EfficientDetBackbone(compound_coef=compound_coef,
                                     num_classes=len(obj_list),
                                     ratios=eval(params['anchors_ratios']),
                                     scales=eval(params['anchors_scales']))
        # load model weights file from disk
        if args.weights:
            if os.path.isfile(os.path.join(os.path.abspath(os.getcwd()), args.weights)):
                model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

        model.requires_grad_(False)
        if params['num_gpus'] > 0:
            model = model.cuda()
        start_node()
    except rospy.ROSInterruptException:
        pass
