import cv2
import torch
from face_detection.face_ssd_infer import SSD
from face_detection.utils import vis_detections
import os 
import sys
from tqdm import tqdm

def face_detection(img_path,device="cuda"):
    device = torch.device(device)
    conf_thresh = 0.9
    target_size = (300, 300)

    net = SSD("test")
    net.load_state_dict(torch.load('face_detection/weights/WIDERFace_DSFD_RES152.pth'))
    net.to(device).eval();
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    detections = net.detect_on_image(img, target_size, device, is_pad=False, keep_thresh=conf_thresh)
    bbox_list = vis_detections(img, detections, conf_thresh)#, show_text=False
    
    return bbox_list


bbox_list = face_detection('rcb.jpg')


print(bbox_list)