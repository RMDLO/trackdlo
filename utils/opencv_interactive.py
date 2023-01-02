#!/usr/bin/env python3

import rospy
import ros_numpy
from sensor_msgs.msg import Image
import cv2
import numpy as np

rect = (0,0,0,0)
startPoint = False
endPoint = False
start_moving = False
rect_center = None
offsets = None
resting = False

# this mask will get updated each iteration
mouse_mask = None

def on_mouse(event, x, y, flags, params):

    global rect, startPoint, endPoint, start_moving, rect_center, offsets, resting

    # get mouse click
    if event == cv2.EVENT_LBUTTONDOWN:

        if startPoint == True and endPoint == True:
            startPoint = False
            endPoint = False
            rect = (0, 0, 0, 0)

        if startPoint == False:
            rect = (x, y, x, y)
            startPoint = True
        elif endPoint == False:
            rect = (rect[0], rect[1], x, y)
            endPoint = True
    
    # draw rectangle when mouse hovering
    elif event == cv2.EVENT_MOUSEMOVE and startPoint == True and endPoint == False:
        rect = (rect[0], rect[1], x, y)

    elif event == cv2.EVENT_MBUTTONDOWN and start_moving == False and np.sum(mouse_mask[y, x]) == 0:
        start_moving = True
        # record rect center
        rect_center = (x, y)
        # offsets: left, up, right, down
        offsets = (rect[0]-rect_center[0], rect[1]-rect_center[1], rect[2]-rect_center[0], rect[3]-rect_center[1])
    
    elif event == cv2.EVENT_MOUSEMOVE and start_moving == True:
        rect = (x+offsets[0], y+offsets[1], x+offsets[2], y+offsets[3])
        resting = False

    elif event == cv2.EVENT_MBUTTONDOWN and start_moving == True:
        start_moving = False
    
    elif not event == cv2.EVENT_MOUSEMOVE and start_moving == True:
        resting = True

def callback (rgb):
    global rect, startPoint, endPoint, mouse_mask, start_moving, resting

    # # debug
    # print("start moving =", start_moving)
    # print("resting =", resting)

    cur_image = ros_numpy.numpify(rgb)

    # convert color for opencv display
    cur_image = cv2.cvtColor(cur_image.copy(), cv2.COLOR_BGR2RGB)

    # frame = cur_image.copy()
    # resize for smaller window
    height, width, layers = cur_image.shape
    new_h = int(height / 1.5)
    new_w = int(width / 1.5)
    frame = cv2.resize(cur_image, (new_w, new_h))

    # initialize mask if none
    if mouse_mask is None:
        mouse_mask = np.ones(frame.shape)

    # filter with mask
    frame = (frame * np.clip(mouse_mask, 0.5, 1)).astype('uint8')

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)    

    key = cv2.waitKey(10)

    if key == 114: # r
        # reset everyhting
        frame = cv2.resize(cur_image, (new_w, new_h))
        startPoint = False
        endPoint = False
        start_moving = False
        mouse_mask = np.ones(frame.shape)
        cv2.imshow('frame',frame)
    elif start_moving == True and resting == False:
        # first reset
        mouse_mask = np.ones(frame.shape)
        mouse_mask[rect[1]:rect[3], rect[0]:rect[2], :] = 0
        cv2.imshow('frame', frame)
    else:
        #drawing rectangle
        if startPoint == True and endPoint != True:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
        
        # if another rectangle is drawn, update mask
        if startPoint == True and endPoint == True:
            mouse_mask[rect[1]:rect[3], rect[0]:rect[2], :] = 0

        cv2.imshow('frame', frame)

    # publish mask
    occlusion_mask = (mouse_mask*255).astype('uint8')

    # resize back for pub
    occlusion_mask = cv2.resize(occlusion_mask, (width, height))

    occlusion_mask_img_msg = ros_numpy.msgify(Image, occlusion_mask, 'rgb8')
    occlusion_mask_img_pub.publish(occlusion_mask_img_msg)


if __name__=='__main__':
    rospy.init_node('test', anonymous=True)

    rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, callback)
    occlusion_mask_img_pub = rospy.Publisher('/mask_with_occlusion', Image, queue_size=10)

    rospy.spin()