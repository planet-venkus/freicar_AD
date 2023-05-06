import birdsEyeT
import cv2
import numpy as np
import torch

def visJetColorCoding(img):
    color_img = np.zeros(img.shape, dtype=img.dtype)
    cv2.normalize(img, color_img, 0, 255, cv2.NORM_MINMAX)
    color_img = color_img.astype(np.uint8)
    color_img = cv2.applyColorMap(color_img, cv2.COLORMAP_JET, color_img)
    return color_img

def TensorImage1ToCV(data):
    cv = data.cpu().data.numpy().squeeze()
    return cv

lane_reg_example = cv2.imread('../../data/seg_reg_data/lane_dt/1603450738_189883683.png', cv2.IMREAD_GRAYSCALE)
corr_rgb_image = cv2.imread('../../data/detection_data/synth/1603450738_189883683.png')
lane_reg_example = torch.from_numpy(lane_reg_example).float().unsqueeze(0)


hom_conv = birdsEyeT.birdseyeTransformer('freicar_homography.yaml', 3, 3, 200, 2)  # 3mX3m, 200 pixel per meter
bev = visJetColorCoding(hom_conv.birdseye(TensorImage1ToCV(lane_reg_example)))
original = visJetColorCoding(TensorImage1ToCV(lane_reg_example))

cv2.imshow('BEV', bev)
cv2.imshow('Original perspective', original)
cv2.imshow('Corresponding RGB image', corr_rgb_image)
cv2.waitKey()