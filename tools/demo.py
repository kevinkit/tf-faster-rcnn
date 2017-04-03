#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test_vgg16 import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16_depre import vgg16

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt', 'vgg16.weights')}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def vis_detectionOnCam(im, class_name,dets,thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
#        print(bbox)
#        print(score)
#	print(class_name)
        rgb = colors[i];
        rgb = colors[CLASSES.index(class_name)]
	print(rgb,rgb[0],rgb[1],rgb[2])
	cv2.rectangle(im, (bbox[0],bbox[1]),(bbox[2],bbox[3]),(rgb[0],rgb[1],rgb[2]),2)
#	cv2.rectangle(im, (bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,255,255),2)	
        cv2.putText(im,class_name,(bbox[0],bbox[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,255)
    return im


#Will return the indices to be used
def Threshing(class_names, dets,thresh=0.5):
	inds = np.where(dets[:,-1] >= thresh)[0];
	print(class_names)
	for i in inds:
		print(i)
		print(class_names[i]);



def camdemo(sess,net,im,resize=(500,375)):
    #Resize iamge so it fullfills the requirements needed by Faster-RCNN   
    im = cv2.resize(im,resize)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

#    return scores,boxes
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
#	print(cls,dets)
#	Threshing(cls,dets,thresh=CONF_THRESH)
	Img = vis_detectionOnCam(im, cls, dets, thresh=CONF_THRESH)


    return Img
#    cv2.imshow('frame',im)




def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))


    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):

        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
	vis_detections(im, cls, dets, thresh=CONF_THRESH)

def init_colors(bias=50,maxi=255):

 color = [];
 step = 255./len(CLASSES[1,:])
 for i in range(0,len(CLASSES[1,:])):
	np.append(color,[bias + step*i,bias + step*i,bias + step*i]);

 return color;	


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')


    args = parser.parse_args()

    return args




def getColors(Classes):
 per_channel = Classes/9.
 step_per_channel = 255/per_channel;
 cnt = 0
 rgb = []
 for j in range(0,int(per_channel)+1):
  rgb.append(int(j*step_per_channel))
 cols = []
 for i in range(0,len(rgb)):
  for j in range(0,len(rgb)):
   for k in range(0,len(rgb)):
     cols.append([rgb[i],rgb[j],rgb[k]])
     cnt = cnt +1

 return cols 









if __name__ == '__main__':

    global colors
    colors = getColors(len(CLASSES[1:]))
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    tfmodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', 'voc_2007_trainval', 'default',
                              NETS[demonet][0])
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(tfmodel + '.meta'))

    # weight path
    tfweight = os.path.join(cfg.DATA_DIR, 'imagenet_weights', NETS[demonet][1])
    if not os.path.isfile(tfweight):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_imagenet_weights.sh?').format(tfweight))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    else:
        raise NotImplementedError

    net.create_architecture(sess, "TEST", 21, caffe_weight_path=tfweight, 
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']

    
    #Crate Capture Object
    cap0 = cv2.VideoCapture(0);
    cap1 = cv2.VideoCapture(1);

    #Get Frame
    cv2.namedWindow('captured 0')
    cv2.namedWindow('captured 1')
    cv2.namedWindow('detected 0')
    cv2.namedWindow('detected 1')
    while(True): 
     ret0, frame0 = cap0.read(); 
     ret1, frame1 = cap1.read();
     if ret0:
	cv2.imshow('captured 0',frame0);
     	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
     	Img = camdemo(sess,net,frame0);

	if Img is not None:
        	cv2.imshow('detected 0',Img);

     else:
	print("Could not recieve data from camera 0")

     if ret1:
	cv2.imshow('captured 1',frame1);
	if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        Img = camdemo(sess,net,frame1);

	if Img is not None:
        	cv2.imshow('detected 1',Img);

     else:
        print("Could not recieve data from camera 1");


     if ret0 and ret1:
	print("both cameras working, trying some matching...")


#    for im_name in im_names:
#        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#        print('Demo for data/demo/{}'.format(im_name))
#        demo(sess, net, im_name)

#    plt.show()
