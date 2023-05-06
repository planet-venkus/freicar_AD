#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np
from torch.backends import cudnn
import argparse
import yaml
import torch
import os
import sys
from rospy_tutorials.msg import Floats
from cv_bridge import CvBridge, CvBridgeError
from torchvision import transforms
# TODO : Remove the hard coding of the path.
sys.path.append('/home/freicar/freicar_ws/src/freicar-2020-exercises/01-01-object-detection-roboracers')
from rospy.numpy_msg import numpy_msg
from image_boundingboxinfo_publisher.msg import box, boxes
from model.efficientdet.backbone import EfficientDetBackbone
from model.efficientdet.utils import BBoxTransform, ClipBoxes

from utils import postprocess, boolean_string, STANDARD_COLORS, standard_to_bgr
from utils import display
from dataloader.freicar_dataloader import FreiCarDataset
from model.efficientdet.dataset import collater
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='freicar-detection', help='project file that contains parameters')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')

args = ap.parse_args()
project_name = args.project
weights_path = args.weights


force_input_size = None

threshold = 0.2
nms_threshold = 0.2

use_cuda = True
cudnn.fastest = True
cudnn.benchmark = True
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
obj_list = ['freicar']
compound_coef = 0

def GenBBOverlay(bbs, draw_image=None):
    bb_image = np.zeros((draw_image.shape[0], draw_image.shape[1], 3)).astype(np.uint8)
    overlay_img = (draw_image.copy()).astype(np.uint8)
    for bb in bbs:
        if draw_image is not None:
            cv2.rectangle(bb_image, (bb[0], bb[1]), (bb[2], bb[3]), color=(0, 255, 255), thickness=3)

    overlay = cv2.addWeighted(bb_image, 0.3, overlay_img, 0.7, 0)
    return overlay

def TensorImage3ToCV(data):
    cv = np.transpose(data.cpu().data.numpy().squeeze(), (1, 2, 0))
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    return cv

def img_callback(msg):
    np_img = np.frombuffer(msg.data, dtype=np.uint8).reshape((720, 1280, 3))
    bgr = np.zeros((np_img.shape[0], np_img.shape[1], 3), dtype=np.uint8)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR, bgr, 3)
    rgb = np.zeros((np_img.shape[0], np_img.shape[1], 3), dtype=np.uint8)
    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, rgb, 3)
    rgb = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
    rgb = cv2.copyMakeBorder(rgb, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    rgb = (torch.from_numpy(rgb)).permute(2, 0, 1)
    rgb_tensor = torch.unsqueeze(rgb, 0)
    rgb_tensor = rgb_tensor.cuda().float()
    imgs = rgb_tensor.float()

    with torch.no_grad():
        _, regression, classification, anchors = model(imgs)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    preds = postprocess(imgs, anchors, regression, classification, regressBoxes, clipBoxes, threshold,
                        nms_threshold)

    #for further use
    # imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    # imgs = (imgs * 255).astype(np.uint8)
    # imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
    # display(preds, imgs, imshow=True, imwrite=False, obj_list=obj_list)

    bbs_preds = preds[0]
    bbs = bbs_preds['rois']
    #Filtering out the boxes with lower scores
    bbs_scores = bbs_preds['scores']
    valid_bbs_index = np.argwhere(bbs_scores > 0.85).flatten()
    bbs_num = len(valid_bbs_index)
    # cv2.imshow("bbsimage", bbs)
    # bbs = np.squeeze(bbs)
    b_b_msg = box()
    b_b_s_msg = boxes()

    cv_rgb = TensorImage3ToCV(imgs)
    bbs_img_overlay = GenBBOverlay(bbs, cv_rgb)
    bridge = CvBridge()
    # cv2.imshow("overlaid image", bbs_img_overlay)
    # cv2.waitKey(2)
    bbs_img_overlay_msg = bridge.cv2_to_imgmsg(bbs_img_overlay, "bgr8")
    bbs_img_overlay_msg.header.stamp = msg.header.stamp
    bbs_pub.publish(bbs_img_overlay_msg)
    if bbs_num != 0:
        bbs = bbs[valid_bbs_index]
        for bb in bbs:
            # print('bbs', bb)
            b_b_msg.xmin = bb[0]
            b_b_msg.ymin = bb[1]
            b_b_msg.xmax = bb[2]
            b_b_msg.ymax = bb[3]
            xcenter = ((bb[2] - bb[0])/2.0) + bb[0]
            ycenter = ((bb[3] - bb[1])/2.0) + bb[1]
            b_b_msg.xcenter = xcenter
            b_b_msg.ycenter = ycenter

            b_b_s_msg.bounding_boxes.append(b_b_msg)

        pub.publish(b_b_s_msg)
        rospy.Rate(10)  # 10hz

if __name__ == '__main__':
    try:
        # start_node()
        rospy.init_node('bbs_publisher')
        rospy.loginfo('bbs_publisher node started')
        # pub = rospy.Publisher('bbsinfo', numpy_msg(Floats), queue_size=1)
        pub = rospy.Publisher('bbsarray', boxes, queue_size=10)
        bbs_pub = rospy.Publisher('/boundingbox/overlaid_image', Image, queue_size=1)
        model = EfficientDetBackbone(compound_coef=compound_coef,
                                     num_classes=len(obj_list),
                                     ratios=anchor_ratios,
                                     scales=anchor_scales)

        if args.weights:
            if os.path.isfile(os.path.join(os.path.abspath(os.getcwd()), args.weights)):
                model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model = model.cuda()  # Using cuda True
        img_sub = rospy.Subscriber('/freicar_1/sim/camera/rgb/front/image', Image, callback=img_callback, queue_size=1)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
