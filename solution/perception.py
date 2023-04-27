import cv2
import numpy as np
from types import tuple
import math
from scipy.stats import linregress


def get_center_lane(image: np.ndarray) -> np.ndarray:
    """
    Get the center lane of the road by transforming the image to HSV then filter the white color
    Parameters
    ----------
    image : np.ndarray
        The image to process
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # filter non white color
    lower_red = np.array([0, 0, 190])
    upper_red = np.array([0, 3, 255])

    # filter by white color which is just the grey region of the road
    res = cv2.inRange(hsv, lower_red, upper_red)

    # get the contours
    contours, _ = cv2.findContours(
        res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # remove the small contours
    mask = np.zeros(res.shape, np.uint8)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 28000:
            mask = cv2.drawContours(mask, [contour], -1, 255, -1)

    # remove the top part of the image
    mask[-70:, :] = 0
    return mask


def get_center(img: np.ndarray) -> tuple(int, int):
    """
    Get the center of the lane
    Parameters
    ----------
    img : np.ndarray
        The image to process
    """
    M = cv2.moments(img)
    cxm, cxy = None, None
    if M['m00'] > 0:
        cxm = int(M['m10']/M['m00'])
        cym = int(M['m01']/M['m00'])
        cv2.circle(img, (cxm, cym), 20, (0, 0, 0), -1)

    return cxm, cxy


def get_center_lane_slope(img: np.ndarray) -> float:
    """
    Get the slope of the center lane by finding the slope of the largest red contour
    Parameters
    ----------
    img : np.ndarray
        The image to process
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([150, 70, 0])
    upper_red = np.array([255, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([0, 100, 0])
    upper_red = np.array([25, 255, 255])

    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    result = cv2.bitwise_or(mask1, mask2)

    # remove noise by performing opening morph ex
    kernel = np.ones((5, 5))
    result = cv2.dilate(result, kernel)

    # find contours corresponding to different red areas of the road
    contours, _ = cv2.findContours(
        result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_contour = -1
    slope = 0

    # get the slope of the largest contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_contour:
            area_contour = area
            slope, _, _, _, _ = linregress(contour[:, 0, :])
            slope = math.atan(slope)/np.pi * 180.

    return slope
