#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import Image

class OcclusionSimulation:
    def __init__(self):

        self.rect = (0,0,0,0)
        self.startPoint = False
        self.endPoint = False
        self.start_moving = False
        self.rect_center = None
        self.offsets = None
        self.resting = False

        # update the mask each iteration
        self.mouse_mask = None

        self.rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        self.occlusion_mask_img_pub = rospy.Publisher('/mask_with_occlusion', Image, queue_size=100)

    def callback(self,rgb):

        # # debug
        # print("start moving =", start_moving)
        # print("resting =", resting)

        cur_image = ros_numpy.numpify(rgb)

        # convert color for opencv display
        cur_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_BGR2RGB)

        # resize for smaller window
        height, width, layers = cur_image.shape
        new_h = int(height / 1.5)
        new_w = int(width / 1.5)
        frame = cv2.resize(cur_image, (new_w, new_h))

        # initialize mask if none
        if self.mouse_mask is None:
            self.mouse_mask = np.ones(frame.shape)

        # filter with mask
        frame = (frame * np.clip(self.mouse_mask, 0.5, 1)).astype('uint8')

        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self.on_mouse)    

        key = cv2.waitKey(10)

        if key == 114: # r
            # reset everyhting
            frame = cv2.resize(cur_image, (new_w, new_h))
            self.startPoint = False
            self.endPoint = False
            self.start_moving = False
            self.mouse_mask = np.ones(frame.shape)
            cv2.imshow('frame',frame)
        elif self.start_moving == True and self.resting == False:
            # first reset
            self.mouse_mask = np.ones(frame.shape)
            self.mouse_mask[self.rect[1]:self.rect[3], self.rect[0]:self.rect[2], :] = 0
            cv2.imshow('frame', frame)
        else:
            #drawing rectangle
            if self.startPoint == True and self.endPoint != True:
                cv2.rectangle(frame, (self.rect[0], self.rect[1]), (self.rect[2], self.rect[3]), (0, 0, 255), 2)
            
            # if another rectangle is drawn, update mask
            if self.startPoint == True and self.endPoint == True:
                self.mouse_mask[self.rect[1]:self.rect[3], self.rect[0]:self.rect[2], :] = 0

            cv2.imshow('frame', frame)

        # publish mask
        occlusion_mask = (self.mouse_mask*255).astype('uint8')

        # resize back for pub
        occlusion_mask = cv2.resize(occlusion_mask, (width, height))

        occlusion_mask_img_msg = ros_numpy.msgify(Image, occlusion_mask, 'rgb8')
        self.occlusion_mask_img_pub.publish(occlusion_mask_img_msg)

    def on_mouse(self,event, x, y, flags, params):

        # get mouse click
        if event == cv2.EVENT_LBUTTONDOWN:

            if self.startPoint == True and self.endPoint == True:
                self.startPoint = False
                self.endPoint = False
                self.rect = (0, 0, 0, 0)

            if self.startPoint == False:
                self.rect = (x, y, x, y)
                self.startPoint = True
            elif self.endPoint == False:
                self.rect = (self.rect[0], self.rect[1], x, y)
                self.endPoint = True
        
        # draw rectangle when mouse hovering
        elif event == cv2.EVENT_MOUSEMOVE and self.startPoint == True and self.endPoint == False:
            self.rect = (self.rect[0], self.rect[1], x, y)

        elif event == cv2.EVENT_MBUTTONDOWN and self.start_moving == False and np.sum(self.mouse_mask[y, x]) == 0:
            self.start_moving = True
            # record rect center
            self.rect_center = (x, y)
            # offsets: left, up, right, down
            self.offsets = (self.rect[0]-self.rect_center[0], self.rect[1]-self.rect_center[1], self.rect[2]-self.rect_center[0], self.rect[3]-self.rect_center[1])
        
        elif event == cv2.EVENT_MOUSEMOVE and self.start_moving == True:
            self.rect = (x+self.offsets[0], y+self.offsets[1], x+self.offsets[2], y+self.offsets[3])
            self.resting = False

        elif event == cv2.EVENT_MBUTTONDOWN and self.start_moving == True:
            self.start_moving = False
        
        elif not event == cv2.EVENT_MOUSEMOVE and self.start_moving == True:
            self.resting = True

if __name__=='__main__':
    rospy.init_node("test")
    t = OcclusionSimulation()
    try:
        rospy.spin()
    except:
        print("Shutting down")