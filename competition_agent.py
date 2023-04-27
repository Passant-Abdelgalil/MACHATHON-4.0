#!/usr/bin/env python3

# ROS imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32

# Solution imports
from solution.utils import translate
from solution.perception import get_center_lane_slope

import cv2
import numpy as np
import time


class FPSCounter:
    def __init__(self):
        self.frames = []

    def step(self):
        self.frames.append(time.monotonic())

    def get_fps(self):
        n_seconds = 5

        count = 0
        cur_time = time.monotonic()
        for f in self.frames:
            if cur_time - f < n_seconds:  # Count frames in the past n_seconds
                count += 1

        return count / n_seconds


def callback(msg):
    global throttle_pub, steering_pub, num, fps_counter, speed_count, last_steer

    original_image = CvBridge().imgmsg_to_cv2(msg)
    original_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    image = original_image[70:, :]
    fps_counter.step()
    fps = fps_counter.get_fps()

    # draw fps on image
    cv2.putText(
        original_image,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("original image", original_image)
    cv2.waitKey(1)

    slope = get_center_lane_slope(image)
    angle = translate(slope, -90, 90, -40, 40)

    if angle > 0:
        if angle < 15:
            angle += 5
        angle = min(angle, 20)
    elif angle < 0:
        if angle > -15:
            angle -= 5
        angle = max(angle, -20)
    else:
        angle = np.sign(last_steer) * 8

    speed = 105
    steering_pub.publish(angle)
    throttle_pub.publish(speed)
    last_steer = angle


def listener():
    global throttle_pub, steering_pub

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('agent', anonymous=True)

    rospy.Subscriber('camera_image', Image, callback)

    throttle_pub = rospy.Publisher("/throttle", Float32, queue_size=1)
    steering_pub = rospy.Publisher("/steering", Float32, queue_size=1)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    fps_counter = FPSCounter()
    listener()
