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
import geometry_msgs.msg
import visualization_msgs.msg
import pdb
# from raiscar_msgs.msg import ControlCommand
from cv_bridge import CvBridge, CvBridgeError

sys.path.append('/home/freicar/freicar_ws/src/freicar_base/freicar_exercises/01-02-semantic-segmentation-roboracers/')
from model import fast_scnn_model
from dataset_helper import color_coder
import birdsEyeT

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='freicar-detection', help='project file that contains parameters')
ap.add_argument('--weights', type=str, default=None, help='/path/to/weights')
# ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
args = ap.parse_args()

# use_cuda = args.cuda
gpu = args.device

project_name = args.project
weights_path = args.weights

model = fast_scnn_model.Fast_SCNN(3, 4)
if args.weights:
    if os.path.isfile(os.path.join(os.path.abspath(os.getcwd()), args.weights)):
        # print("=> loading checkpoint '{}'".format(args.weights))
        model.load_state_dict(torch.load(os.path.join(os.path.abspath(os.getcwd()), args.weights))['state_dict'])
model = model.cuda()  # Using cuda True


def img_callback(msg):
    np_img = np.fromstring(msg.data, dtype=np.uint8).reshape((720, 1280, 3))
    # np_img = np_img[:, :, :3]
    bgr = np.zeros((np_img.shape[0], np_img.shape[1], 3), dtype=np.uint8)
    cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR, bgr, 3)
    # cv2.imshow("Test", bgr)
    # cv2.waitKey()
    rgb = np.zeros((np_img.shape[0], np_img.shape[1], 3), dtype=np.uint8)
    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, rgb, 3)

    # print(f"rgb image size before resizing= {rgb.shape}")
    rgb = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
    rgb = cv2.copyMakeBorder(rgb, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # print(f"rgb image size after resizing= {rgb.shape}")

    rgb = (torch.from_numpy(rgb)).permute(2, 0, 1)

    # bgr_tensor = torch.from_numpy(bgr)
    rgb_tensor = torch.unsqueeze(rgb, 0)
    rgb_tensor = rgb_tensor.cuda().float()

    # load model weights file from disk
    seg_eval, lreg_eval = model(rgb_tensor)
    img = lreg_eval.detach().cpu().squeeze().numpy()
    # cv2.imshow('Lane Regression', img)
    # cv2.waitKey(10)

    lreg_eval = lreg_eval[:, :, 12:372, :]  # Unpadding from top and bottom
    lreg_cv = TensorImage1ToCV(lreg_eval)
    # print(f"lreg_cv shape = {lreg_cv.shape}")
    lreg_norm = cv2.normalize(lreg_cv, None, alpha=0.00, beta=1.00, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ret, lreg_thresh = cv2.threshold(lreg_norm, 0.50, 1.00, cv2.THRESH_TOZERO)

    # cv2.imshow('Lreg before birds eye', lreg_norm)

    # cv2.waitKey()
    ############### Birds Eye Transform ############
    hom_conv = birdsEyeT.birdseyeTransformer('./dataset_helper/freicar_homography.yaml', 3, 3, 100,
                                             2)  # 3mX3m, 200 pixel per meter
    lreg_bev = hom_conv.birdseye(lreg_thresh)
    lreg_bev = cv2.flip(lreg_bev, -1)
    lreg_bev = lreg_bev * 255
    lreg_bev = lreg_bev.astype(np.uint8)
    # cv2.imshow('BEV', lreg_bev)
    # cv2.waitKey()
    ros_bev_img = CvBridge().cv2_to_imgmsg(lreg_bev, 'mono8')
    # print(f"lreg_bev shape = {lreg_bev.shape}")

    ros_bev_img.header.stamp = msg.header.stamp

    bev_pub = rospy.Publisher('/freicar_1/sim/camera/rgb/front/reg_bev', Image, queue_size=10)

    bev_pub.publish(ros_bev_img)

    threshold_pixels = lreg_bev  # taking only red channel pixels
    idxs = np.transpose((threshold_pixels > 0).nonzero())  # only pixels with non zero red value
    n_samples = 1000
    if (idxs.shape[0] - 1) > 0:
        sample = np.random.choice((idxs.shape[0] - 1), size=n_samples, replace=True)
        sampled_idxs = idxs[sample]
        # sub = threshold_pixels.shape[1] / 100.0
        marker_array = visualization_msgs.msg.MarkerArray()
        count = 0
        for pt in sampled_idxs:
            marker = visualization_msgs.msg.Marker()
            pose = geometry_msgs.msg.Pose()
            point = geometry_msgs.msg.Point()
            point.y = pt[1] / 100.0 - 1.5
            # point.y = pt[1] / 100.0 - 3.1
            point.x = pt[0] / 100.0 + 1.6
            # point.x = pt[0] / 100.0 - sub + 2
            point.z = 0.0
            pose.position = point
            # marker.type = visualization_msgs.msg.Marker.POINTS
            # marker.action = visualization_msgs.msg.Marker.ADD
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.id = count
            marker.pose = pose
            marker.color.g = 1.0
            marker.color.r = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker_array.markers.append(marker)
            # print(f"marker x = {point.x}")
            count += 1

        pub = rospy.Publisher('/lreg', visualization_msgs.msg.MarkerArray, queue_size=10)

        pub.publish(marker_array)

    # marker.color.g = 1.0
    # marker.color.r = 0.0
    # marker.color.b = 0.0

    # cv2.imshow('BEV', bev)
    # cv2.waitKey()


def TensorImage1ToCV(data):
    cv = data.cpu().data.numpy().squeeze()
    return cv


# def visJetColorCoding(name, img):
#     color_img = np.zeros(img.shape, dtype=img.dtype)
#     cv2.normalize(img, color_img, 0, 255, cv2.NORM_MINMAX)
#     color_img = color_img.astype(np.uint8)
#     color_img = cv2.applyColorMap(color_img, cv2.COLORMAP_JET, color_img)
#     color_img = cv2.rotate(color_img, cv2.ROTATE_180)
#     cv2.imshow(name, color_img)
def visJetColorCoding(img):
    color_img = np.zeros(img.shape, dtype=img.dtype)
    cv2.normalize(img, color_img, 0, 255, cv2.NORM_MINMAX)
    color_img = color_img.astype(np.uint8)
    color_img = cv2.applyColorMap(color_img, cv2.COLORMAP_JET, color_img)
    return color_img


def start_node():
    rospy.init_node('lreg_birdseye')
    rospy.loginfo('Lane regression node started')
    img_sub = rospy.Subscriber('/freicar_1/sim/camera/rgb/front/image', Image, callback=img_callback, queue_size=10)
    # pub = rospy.Publisher('/freicar_1/control', ControlCommand, queue_size=10)
    # rate = rospy.Rate(10)  # 10hz
    # com = input("enter command")
    # start_time = rospy.get_time()
    # while not rospy.is_shutdown():
    #     # com = input("enter next command")
    #     ctrl_msg = ControlCommand()
    #     if rospy.get_time() - start_time < 4:
    #         ctrl_msg.steering = 0.5
    #         ctrl_msg.throttle = 0.1
    #     elif 4 < rospy.get_time() - start_time < 7:
    #         ctrl_msg.steering = -0.5
    #         ctrl_msg.throttle = 0.1
    #     else:
    #         ctrl_msg.steering = 0.0
    #         ctrl_msg.throttle = 0.1
    #         # if "w" == str(com):
    #         #     ctrl_msg.throttle = 0.2
    #         #
    #         # elif "a" == str(com):
    #         #     ctrl_msg.steering = 0.5
    #         #     ctrl_msg.throttle = 0.1
    #         # elif "d" == str(com):
    #         #     ctrl_msg.steering = 0.5
    #         #     ctrl_msg.throttle = 0.1
    #         # elif "s" == str(com):
    #         #     ctrl_msg.throttle = -0.2
    #         #     ctrl_msg.throttle = 0.0
    #     pub.publish(ctrl_msg)
    # com = ""

    rospy.spin()


if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
